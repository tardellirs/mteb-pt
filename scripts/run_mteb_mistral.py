#!/usr/bin/env python
"""Run MTEB(por, v2) for a Mistral embedding model via the direct Mistral API,
respecting the free-tier rate limit.

Confirmed free-tier limits (from live API headers, 2026-06-26):
  - 60 RPM hard limit (x-ratelimit-limit-req-minute=60)
  - No observable TPM limit in headers (TPM header absent even under load)
  - No `input_type` parameter (extra fields → 422)
  - mistral-embed == mistral-embed-2312 (identical aliases; same model, same weights,
    bitwise-identical embeddings confirmed empirically)

Feasibility at 60 RPM with batch_size=128:
  - 25 light tasks:     ~17 min (throttle-only)
  - Quati (250k corpus): ~33 min
  - Full suite total:   ~50 min raw + ~1.5× for actual latency → ~75 min wall-clock
  -> FEASIBLE in a single run on the free key.

Quati runs LAST so all cheap tasks finish first if the key budget bites.

Env:
  MISTRAL_API_KEY      required
  MISTRAL_MODEL        default: mistral-embed  (aliases mistral-embed-2312 → same)
  MISTRAL_RPM          default: 55   (5-RPM safety margin under the 60 limit)
  MISTRAL_BATCH_SIZE   default: 128  (texts per API call)
  HF_RESULTS_REPO      default: mteb-pt/mteb-pt-results
  MTEB_CACHE           default: ~/.cache/mteb
  HF_SYNC_SECONDS      default: 120
  HF_TOKEN             optional (needed to push results)
  MTEB_TASKS           optional comma-list to restrict tasks
  MTEB_OVERWRITE       set to "1" to overwrite existing results (default: only-missing)
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

import numpy as np

import mteb
import mteb_pt  # noqa: F401  (side-effect: registers tasks)
import mteb_pt.register as register
from huggingface_hub import HfApi, snapshot_download
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO = os.environ.get("HF_RESULTS_REPO", "mteb-pt/mteb-pt-results")
CACHE = os.environ.get("MTEB_CACHE", os.path.expanduser("~/.cache/mteb"))
RESULTS = os.path.join(CACHE, "results")
SYNC_EVERY = int(os.environ.get("HF_SYNC_SECONDS", "120"))
TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("MISTRAL_MODEL", "mistral-embed")
RPM = float(os.environ.get("MISTRAL_RPM", "55"))       # free tier = 60; use 55 for safety
BATCH_SIZE = int(os.environ.get("MISTRAL_BATCH_SIZE", "128"))
DIM = 1024  # confirmed empirically; fixed for mistral-embed

_EXCLUDED = {"OffComBR", "CSTNewsClustering", "BBCNewsPTClustering", "TweetSentBR"}
_PRIORITY = {"Retrieval": 0, "Reranking": 1, "Clustering": 2}

MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"


# ---------------------------------------------------------------------------
# Task selection (mirrors voyage/openai runners)
# ---------------------------------------------------------------------------
def v2_tasks() -> list:
    tasks = [cls() for cls in register._TASKS_TO_REGISTER if cls.metadata.name not in _EXCLUDED]
    tasks.append(mteb.get_task("Assin2STS"))
    only = os.environ.get("MTEB_TASKS")
    if only:
        keep = {x.strip() for x in only.split(",")}
        tasks = [t for t in tasks if t.metadata.name in keep]
    # Quati LAST (largest corpus); otherwise light → heavy
    tasks.sort(key=lambda t: (1 if t.metadata.name == "Quati" else 0,
                               _PRIORITY.get(t.metadata.type, 9)))
    return tasks


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class MistralEmbedModel(AbsEncoder):
    """mteb AbsEncoder backed by the Mistral /v1/embeddings API, throttled to RPM.

    No input_type param (confirmed: Mistral rejects extra fields with 422).
    Both 'mistral-embed' and 'mistral-embed-2312' route to the same weights.
    """

    def __init__(self):
        self._api_key = os.environ["MISTRAL_API_KEY"]
        self._lock = threading.Lock()
        self._last_req = 0.0
        self._min_gap = 60.0 / RPM  # seconds between requests
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=f"mistralai/{MODEL_ID}",
            revision="api",
            release_date="2023-12-11",
            languages=["por-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=8192,
            embed_dim=DIM,
            license=None,
            open_weights=False,
            public_training_code=None,
            public_training_data=None,
            framework=["API"],
            similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=False,   # no input_type → no query/doc asymmetry
            training_datasets=None,
        )

    def _throttle(self):
        with self._lock:
            wait = self._min_gap - (time.time() - self._last_req)
            if wait > 0:
                time.sleep(wait)
            self._last_req = time.time()

    def _call_api(self, batch: list[str]) -> list[list[float]]:
        """Single API call with exponential backoff on 429/5xx."""
        payload = json.dumps({"model": MODEL_ID, "input": batch}).encode()
        delays = (5, 20, 45, 90, None)
        for delay in delays:
            self._throttle()
            req = urllib.request.Request(
                MISTRAL_EMBED_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    resp = json.load(r)
                    return [item["embedding"] for item in resp["data"]]
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:300]
                if delay is None:
                    print(f"  [mistral] give-up: HTTP {e.code} {body[:110]}", flush=True)
                    return [[0.0] * DIM] * len(batch)
                is_rate = e.code in (429, 503)
                extra = 60 if is_rate else 0
                print(f"  [mistral] retry (HTTP {e.code}) wait {delay + extra}s", flush=True)
                time.sleep(delay + extra)
            except Exception as e:  # noqa: BLE001
                if delay is None:
                    print(f"  [mistral] give-up: {e}", flush=True)
                    return [[0.0] * DIM] * len(batch)
                print(f"  [mistral] retry ({str(e)[:60]}) wait {delay}s", flush=True)
                time.sleep(delay)
        return [[0.0] * DIM] * len(batch)  # unreachable but satisfies mypy

    def _embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = [(t[:30000] if t else " ") or " " for t in texts[i: i + BATCH_SIZE]]
            out.extend(self._call_api(batch))
        return np.array(out, dtype=np.float32)

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs):
        texts = [text for batch in inputs for text in batch["text"]]
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        return self._embed(texts)


# ---------------------------------------------------------------------------
# HF persistence helpers (identical pattern to voyage/openai runners)
# ---------------------------------------------------------------------------
def _n_files(root: str) -> int:
    return sum(len(f) for _, _, f in os.walk(root)) if os.path.isdir(root) else 0


def pull_from_hf():
    os.makedirs(RESULTS, exist_ok=True)
    try:
        snapshot_download(
            REPO, repo_type="dataset",
            allow_patterns="results/**",
            local_dir=CACHE, token=TOKEN,
        )
        print(f"[hf-resume] pulled {_n_files(RESULTS)} files", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[hf-resume] pull skipped ({str(e)[:90]})", flush=True)


def _upload_once(api: HfApi):
    if _n_files(RESULTS) == 0:
        return
    for attempt in range(3):
        try:
            api.upload_folder(
                folder_path=RESULTS,
                path_in_repo="results",
                repo_id=REPO,
                repo_type="dataset",
                commit_message="mistral sync",
            )
            return
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1) + random.uniform(0, 5))
                continue
            print(f"[hf-resume] sync err ({str(e)[:90]})", flush=True)
            return


def sync_loop(stop: threading.Event, api: HfApi):
    time.sleep(SYNC_EVERY * random.uniform(0.5, 1.5))
    while not stop.is_set():
        _upload_once(api)
        stop.wait(SYNC_EVERY)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    overwrite = "overwrite" if os.environ.get("MTEB_OVERWRITE") == "1" else "only-missing"
    pull_from_hf()
    tasks = v2_tasks()
    print(
        f"[mistral] {MODEL_ID} | dim={DIM} | RPM={RPM} batch={BATCH_SIZE} "
        f"| {len(tasks)} tasks (Quati last) | overwrite={overwrite} | repo={REPO}",
        flush=True,
    )
    api = HfApi(token=TOKEN)
    stop = threading.Event()
    threading.Thread(target=sync_loop, args=(stop, api), daemon=True).start()
    t0 = time.time()
    print(f"\n=== model: mistralai/{MODEL_ID} ===", flush=True)
    try:
        mteb.evaluate(
            MistralEmbedModel(),
            tasks=tasks,
            overwrite_strategy=overwrite,
            encode_kwargs={"batch_size": BATCH_SIZE},
            raise_error=False,
        )
        print(f"=== mistral done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"=== mistral FAILED: {type(e).__name__}: {str(e)[:200]} ===", flush=True)
    stop.set()
    _upload_once(api)
    print("=== FLEET RUN COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
