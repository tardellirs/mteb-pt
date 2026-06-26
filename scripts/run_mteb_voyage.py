#!/usr/bin/env python
"""Run MTEB(por, v2) for Voyage's CONTEXTUALIZED model (voyage-context-4) via the
DIRECT Voyage API, respecting the free-tier rate limit.

voyage-context-4 is a contextualized chunk-embedding model: endpoint
/v1/contextualizedembeddings, SDK client.contextualized_embed(inputs, model,
input_type). We embed each text as a single-chunk document -> one vector per text.
inputs MUST be List[List[str]] (each text wrapped) for input_type="document";
that nested form also works for "query", so we always wrap.

Rate limit (free, no payment method): 3 RPM / 10K TPM -> very slow (Quati ~hours).
Add a payment method on Voyage to get Tier 1 (2000 RPM / 3M TPM) WITHOUT being
charged until the 200M free tokens run out. Throttle via VOYAGE_RPM / VOYAGE_TPM.

Quati runs LAST (the 250k-passage corpus is the biggest single cost) so the cheap
tasks finish first and only Quati is exposed if the budget/limit bites.

Env: VOYAGE_API_KEY, VOYAGE_MODEL (default voyage-context-4), VOYAGE_RPM (default 3),
     VOYAGE_TPM (default 10000), VOYAGE_DIM (default 1024), HF_RESULTS_REPO,
     MTEB_CACHE, HF_SYNC_SECONDS, MTEB_TASKS (optional comma-list).
"""
from __future__ import annotations

import os
import random
import sys
import threading
import time

import numpy as np

import mteb
import mteb_pt
import mteb_pt.register as register
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.abs_encoder import AbsEncoder
from huggingface_hub import HfApi, snapshot_download
import voyageai

REPO = os.environ.get("HF_RESULTS_REPO", "mteb-pt/mteb-pt-results")
CACHE = os.environ.get("MTEB_CACHE", os.path.expanduser("~/.cache/mteb"))
RESULTS = os.path.join(CACHE, "results")
SYNC_EVERY = int(os.environ.get("HF_SYNC_SECONDS", "120"))
TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("VOYAGE_MODEL", "voyage-context-4")
DIM = int(os.environ.get("VOYAGE_DIM", "1024"))
RPM = float(os.environ.get("VOYAGE_RPM", "3"))
TPM = float(os.environ.get("VOYAGE_TPM", "10000"))
_EXCLUDED = {"OffComBR", "CSTNewsClustering", "BBCNewsPTClustering", "TweetSentBR"}
_PRIORITY = {"Retrieval": 0, "Reranking": 1, "Clustering": 2}


def v2_tasks() -> list:
    tasks = [cls() for cls in register._TASKS_TO_REGISTER if cls.metadata.name not in _EXCLUDED]
    tasks.append(mteb.get_task("Assin2STS"))
    only = os.environ.get("MTEB_TASKS")
    if only:
        keep = {x.strip() for x in only.split(",")}
        tasks = [t for t in tasks if t.metadata.name in keep]
    # Quati LAST (biggest corpus); everything else first, by light->heavy priority.
    tasks.sort(key=lambda t: (1 if t.metadata.name == "Quati" else 0,
                              _PRIORITY.get(t.metadata.type, 9)))
    return tasks


class VoyageContextModel(AbsEncoder):
    """mteb encoder backed by Voyage's contextualized-embeddings API, throttled."""

    def __init__(self):
        self.client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self._lock = threading.Lock()
        self._last_req = 0.0
        self._min_gap = 60.0 / RPM            # seconds between requests (RPM cap)
        self._max_tok_per_req = max(1000, int(TPM / max(RPM, 1)))  # stay under TPM
        self.mteb_model_meta = ModelMeta(
            loader=None, name=f"voyage/{MODEL_ID}", revision="api",
            release_date="2026-06-01", languages=["por-Latn"], n_parameters=None,
            memory_usage_mb=None, max_tokens=32000, embed_dim=DIM, license=None,
            open_weights=False, public_training_code=None, public_training_data=None,
            framework=["API"], similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=True, training_datasets=None,
        )

    def _throttle(self):
        with self._lock:
            wait = self._min_gap - (time.time() - self._last_req)
            if wait > 0:
                time.sleep(wait)
            self._last_req = time.time()

    def _call(self, batch, input_type):
        # batch: list[str]; always wrap each text as a single-chunk document.
        wrapped = [[t[:30000] if t else " "] for t in batch] or [[" "]]
        for delay in (5, 20, 45, 90, None):
            self._throttle()
            try:
                r = self.client.contextualized_embed(
                    inputs=wrapped, model=MODEL_ID, input_type=input_type,
                    output_dimension=DIM,
                )
                return [res.embeddings[0] for res in r.results]
            except Exception as e:  # noqa: BLE001 -- 429 / transient -> back off
                msg = str(e).lower()
                if delay is None:
                    print(f"  [voyage] give-up batch: {str(e)[:110]}", flush=True)
                    return [[0.0] * DIM] * len(batch)
                extra = 45 if ("rate" in msg or "429" in msg or "limit" in msg) else 0
                print(f"  [voyage] retry ({str(e)[:60]}) wait {delay + extra}s", flush=True)
                time.sleep(delay + extra)

    def _embed(self, texts, input_type):
        # batch so each request stays under ~max_tok_per_req (est. 4 chars/token).
        out, cur, cur_tok = [], [], 0
        for t in texts:
            tok = max(1, len(t or " ") // 4)
            if cur and (cur_tok + tok > self._max_tok_per_req or len(cur) >= 128):
                out.extend(self._call(cur, input_type)); cur, cur_tok = [], 0
            cur.append(t); cur_tok += tok
        if cur:
            out.extend(self._call(cur, input_type))
        return np.array(out, dtype=np.float32)

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs):
        texts = [text for batch in inputs for text in batch["text"]]
        itype = "query" if (prompt_type is not None and "query" in str(prompt_type).lower()) else "document"
        return self._embed(texts, itype) if texts else np.zeros((0, DIM), dtype=np.float32)


def _n_files(root):
    return sum(len(f) for _, _, f in os.walk(root)) if os.path.isdir(root) else 0


def pull_from_hf():
    os.makedirs(RESULTS, exist_ok=True)
    try:
        snapshot_download(REPO, repo_type="dataset", allow_patterns="results/**", local_dir=CACHE, token=TOKEN)
        print(f"[hf-resume] pulled {_n_files(RESULTS)} files", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[hf-resume] pull skipped ({str(e)[:90]})", flush=True)


def _upload_once(api):
    if _n_files(RESULTS) == 0:
        return
    for attempt in range(3):
        try:
            api.upload_folder(folder_path=RESULTS, path_in_repo="results", repo_id=REPO,
                              repo_type="dataset", commit_message="voyage sync")
            return
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1) + random.uniform(0, 5)); continue
            print(f"[hf-resume] sync err ({str(e)[:90]})", flush=True); return


def sync_loop(stop, api):
    time.sleep(SYNC_EVERY * random.uniform(0.5, 1.5))
    while not stop.is_set():
        _upload_once(api); stop.wait(SYNC_EVERY)


def main():
    pull_from_hf()
    tasks = v2_tasks()
    print(f"[voyage] {MODEL_ID} | dim={DIM} | RPM={RPM} TPM={TPM} | {len(tasks)} tasks "
          f"(Quati last) | repo={REPO}", flush=True)
    api = HfApi(token=TOKEN)
    stop = threading.Event()
    threading.Thread(target=sync_loop, args=(stop, api), daemon=True).start()
    t0 = time.time()
    print(f"\n=== model: voyage/{MODEL_ID} ===", flush=True)
    try:
        mteb.evaluate(VoyageContextModel(), tasks=tasks, overwrite_strategy="only-missing",
                      encode_kwargs={"batch_size": 128}, raise_error=False)
        print(f"=== voyage done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"=== voyage FAILED: {type(e).__name__}: {str(e)[:200]} ===", flush=True)
    stop.set()
    _upload_once(api)
    print("=== FLEET RUN COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
