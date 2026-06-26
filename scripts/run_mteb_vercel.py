#!/usr/bin/env python
"""Run MTEB(por, v2) for CLOSED API embedding models via the Vercel AI Gateway.

OpenAI-compatible endpoint (https://ai-gateway.vercel.sh/v1/embeddings) wrapped as
an mteb model, so the API models are scored with the EXACT SAME mteb logic as the
open-weight fleet -> directly comparable in the ranking. No GPU; runs locally.

Resume = HF Hub (same as run_mteb_hfresume.py): pull existing results at start, a
daemon thread syncs new results every ~HF_SYNC_SECONDS, only-missing skips done work.

Env:
  VERCEL_AI_GATEWAY_KEY                      gateway key
  VERCEL_MODEL_IDS  "voyage/voyage-4-lite,voyage/voyage-4,..."   comma-list
  HF_RESULTS_REPO   (default mteb-pt/mteb-pt-results)
  MTEB_CACHE        (default ~/.cache/mteb)
  HF_SYNC_SECONDS   (default 90)
  VERCEL_BATCH_SIZE (default 64)
  MTEB_TASKS        (optional comma-list)    restrict tasks (testing)
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
from openai import OpenAI

REPO = os.environ.get("HF_RESULTS_REPO", "mteb-pt/mteb-pt-results")
CACHE = os.environ.get("MTEB_CACHE", os.path.expanduser("~/.cache/mteb"))
RESULTS = os.path.join(CACHE, "results")
SYNC_EVERY = int(os.environ.get("HF_SYNC_SECONDS", "90"))
TOKEN = os.environ.get("HF_TOKEN")
BASE_URL = "https://ai-gateway.vercel.sh/v1"
MAX_CHARS = 24000  # ~6k tokens defensive
_EXCLUDED = {"OffComBR", "CSTNewsClustering", "BBCNewsPTClustering", "TweetSentBR"}
_PRIORITY = {"Retrieval": 0, "Reranking": 1, "Clustering": 2}


def v2_tasks() -> list:
    tasks = [cls() for cls in register._TASKS_TO_REGISTER if cls.metadata.name not in _EXCLUDED]
    tasks.append(mteb.get_task("Assin2STS"))
    only = os.environ.get("MTEB_TASKS")
    if only:
        keep = {x.strip() for x in only.split(",")}
        tasks = [t for t in tasks if t.metadata.name in keep]
    tasks.sort(key=lambda t: _PRIORITY.get(t.metadata.type, 9))
    return tasks


class VercelGatewayModel(AbsEncoder):
    """mteb encoder (AbsEncoder) backed by the Vercel AI Gateway."""

    def __init__(self, model_id: str, dim: int = 1024):
        self.model_id = model_id
        self.client = OpenAI(api_key=os.environ["VERCEL_AI_GATEWAY_KEY"], base_url=BASE_URL)
        self.batch = int(os.environ.get("VERCEL_BATCH_SIZE", "64"))
        self._dim = None  # auto-detected from the first successful embedding
        self.mteb_model_meta = ModelMeta(
            loader=None, name=model_id, revision="api", release_date="2025-06-01",
            languages=["por-Latn"], n_parameters=None, memory_usage_mb=None,
            max_tokens=32000, embed_dim=dim, license=None, open_weights=False,
            public_training_code=None, public_training_data=None, framework=["API"],
            similarity_fn_name=ScoringFunction.COSINE, use_instructions=False,
            training_datasets=None,
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        texts = [(t[:MAX_CHARS] if t else " ") or " " for t in texts]
        if len(texts) > 1 and sum(len(t) for t in texts) // 3 > 90000:
            mid = len(texts) // 2
            return np.concatenate([self._embed(texts[:mid]), self._embed(texts[mid:])], axis=0)
        for delay in (2, 5, 15, 30, 60, None):
            try:
                r = self.client.embeddings.create(model=self.model_id, input=texts)
                vecs = [d.embedding for d in r.data]
                if self._dim is None:
                    self._dim = len(vecs[0])  # auto-detect so give-up zeros always match
                return np.array(vecs, dtype=np.float32)
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if any(s in msg for s in ("too long", "too large", "maximum", "exceed", "token", "input length")):
                    if len(texts) > 1:
                        mid = len(texts) // 2
                        return np.concatenate([self._embed(texts[:mid]), self._embed(texts[mid:])], axis=0)
                    texts = [texts[0][: max(1, len(texts[0]) // 2)]]  # single text too long -> halve + retry
                    continue
                if delay is None:
                    print(f"  [vercel] giving up on a batch: {str(e)[:100]}", flush=True)
                    return np.zeros((len(texts), self._dim or 1536), dtype=np.float32)
                time.sleep(delay)

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs):
        # inputs is a DataLoader of batches; each batch has a "text" field.
        texts = [text for batch in inputs for text in batch["text"]]
        out = []
        for i in range(0, len(texts), self.batch):
            out.append(self._embed(texts[i:i + self.batch]))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self._dim), dtype=np.float32)


def _n_files(root):
    return sum(len(f) for _, _, f in os.walk(root)) if os.path.isdir(root) else 0


def pull_from_hf():
    os.makedirs(RESULTS, exist_ok=True)
    try:
        snapshot_download(REPO, repo_type="dataset", allow_patterns="results/**", local_dir=CACHE, token=TOKEN)
        print(f"[hf-resume] pulled {_n_files(RESULTS)} files from {REPO}", flush=True)
    except Exception as e:
        print(f"[hf-resume] pull skipped ({str(e)[:100]})", flush=True)


def _upload_once(api):
    if _n_files(RESULTS) == 0:
        return
    for attempt in range(3):
        try:
            api.upload_folder(folder_path=RESULTS, path_in_repo="results", repo_id=REPO,
                              repo_type="dataset", commit_message="vercel sync")
            return
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1) + random.uniform(0, 5)); continue
            print(f"[hf-resume] sync err ({str(e)[:90]})", flush=True); return


def sync_loop(stop, api):
    time.sleep(SYNC_EVERY * random.uniform(0.5, 1.5))
    while not stop.is_set():
        _upload_once(api); stop.wait(SYNC_EVERY)


def main(model_ids):
    pull_from_hf()
    tasks = v2_tasks()
    print(f"[vercel] {len(model_ids)} models x {len(tasks)} tasks | repo={REPO}", flush=True)
    api = HfApi(token=TOKEN)
    stop = threading.Event()
    threading.Thread(target=sync_loop, args=(stop, api), daemon=True).start()
    for mid in model_ids:
        t0 = time.time()
        print(f"\n=== model: {mid} ===", flush=True)
        try:
            model = VercelGatewayModel(mid)
            mteb.evaluate(model, tasks=tasks, overwrite_strategy="only-missing",
                          encode_kwargs={"batch_size": model.batch}, raise_error=False)
            print(f"=== {mid} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"=== {mid} FAILED: {type(e).__name__}: {str(e)[:200]} ===", flush=True)
    stop.set()
    _upload_once(api)
    print("=== FLEET RUN COMPLETE ===", flush=True)


if __name__ == "__main__":
    ids = sys.argv[1:] or [x.strip() for x in os.environ.get("VERCEL_MODEL_IDS", "").split(",") if x.strip()]
    if not ids:
        print("usage: run_mteb_vercel.py <model_id> [...]  (or VERCEL_MODEL_IDS=...)"); sys.exit(1)
    main(ids)
