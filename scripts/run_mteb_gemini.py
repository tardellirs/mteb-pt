#!/usr/bin/env python
"""Run MTEB(por, v2) for gemini-embedding-001 via the Google GenAI API.

Cost-minimal HYBRID encoding (Flex inference does NOT support embeddings, so the
only 50%-discount path is Batch Mode):
  - big encode calls (retrieval/reranking corpora, > BATCH_THRESHOLD texts) -> Batch
    Mode ($0.075/1M, async, rate-limit-safe), split into inline jobs under the 20MB /
    5M-token (Tier 2) per-job caps;
  - small encode calls (queries, classification, STS, clustering) -> standard sync
    ($0.150/1M, batch=100/request).

Wrapped as an mteb AbsEncoder so scores match the open-weight + Voyage fleets exactly.
task_type is set per MTEB task (RETRIEVAL_DOCUMENT/QUERY/CLASSIFICATION/CLUSTERING/
SEMANTIC_SIMILARITY); output_dimensionality=3072 (native, auto-normalized).

Resume = HF Hub (pull at start, daemon sync every ~HF_SYNC_SECONDS, only-missing).

Env: GEMINI_API_KEY/GOOGLE_API_KEY, HF_RESULTS_REPO, MTEB_CACHE, HF_SYNC_SECONDS,
     GEMINI_BATCH_THRESHOLD (default 2000), MTEB_TASKS (testing).
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
from google import genai
from google.genai import types

REPO = os.environ.get("HF_RESULTS_REPO", "mteb-pt/mteb-pt-results")
CACHE = os.environ.get("MTEB_CACHE", os.path.expanduser("~/.cache/mteb"))
RESULTS = os.path.join(CACHE, "results")
SYNC_EVERY = int(os.environ.get("HF_SYNC_SECONDS", "120"))
TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-embedding-001")
DIM = 3072
BATCH_THRESHOLD = int(os.environ.get("GEMINI_BATCH_THRESHOLD", "100000000"))  # Batch mode produces broken embeddings -> sync by default
# ~4MB/job (~1.2M tokens); MAX_PENDING jobs concurrent keeps enqueued < Tier-2 5M-token cap.
CHUNK_CHARS = int(os.environ.get("GEMINI_CHUNK_CHARS", "4000000"))
MAX_PENDING = int(os.environ.get("GEMINI_MAX_PENDING", "4"))
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


def _task_type(task_metadata, prompt_type) -> str:
    pt = str(prompt_type) if prompt_type is not None else ""
    if "query" in pt:
        return "RETRIEVAL_QUERY"
    if "document" in pt or "passage" in pt:
        return "RETRIEVAL_DOCUMENT"
    ttype = getattr(task_metadata, "type", "") or ""
    if ttype in ("Retrieval", "Reranking"):
        return "RETRIEVAL_DOCUMENT"
    if ttype == "Classification" or "Classification" in ttype:
        return "CLASSIFICATION"
    if ttype == "Clustering":
        return "CLUSTERING"
    return "SEMANTIC_SIMILARITY"  # STS, PairClassification, default


class GeminiModel(AbsEncoder):
    """mteb encoder for gemini-embedding-001 (hybrid Batch + sync)."""

    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
        self.mteb_model_meta = ModelMeta(
            loader=None, name=f"google/{MODEL_ID}", revision="api",
            release_date="2025-06-01", languages=["por-Latn"], n_parameters=None,
            memory_usage_mb=None, max_tokens=2048, embed_dim=DIM, license=None,
            open_weights=False, public_training_code=None, public_training_data=None,
            framework=["API"], similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=True, training_datasets=None,
        )

    def _cfg(self, task_type):
        return types.EmbedContentConfig(task_type=task_type, output_dimensionality=DIM)

    def _embed_sync(self, texts, task_type):
        out = []
        for i in range(0, len(texts), 100):
            chunk = [(t[:30000] if t else " ") or " " for t in texts[i:i + 100]]
            for delay in (5, 15, 30, 60, 120, 240, 480, None):
                try:
                    r = self.client.models.embed_content(model=MODEL_ID, contents=chunk, config=self._cfg(task_type))
                    out.extend([e.values for e in r.embeddings]); break
                except Exception as e:  # noqa: BLE001
                    if delay is None:
                        print(f"  [gemini] sync FAILED after ~16min retries -> RAISE (never zero-vec): {str(e)[:80]}", flush=True)
                        raise  # fail the task cleanly; mteb skips it -> re-run, instead of corrupting with zero-vec
                    time.sleep(delay)
        return np.array(out, dtype=np.float32)

    def _submit(self, chunk, task_type):
        return self.client.batches.create_embeddings(
            model=MODEL_ID,
            src=types.EmbeddingsBatchJobSource(
                inlined_requests=types.EmbedContentBatch(contents=chunk, config=self._cfg(task_type))
            ),
            config=types.CreateEmbeddingsBatchJobConfig(display_name="mteb-pt"),
        )

    def _embed_batch(self, texts, task_type):
        """Batch Mode (50% off) with a concurrency window (up to MAX_PENDING jobs at once,
        within the Tier-2 5M enqueued-token cap)."""
        texts = [(t[:30000] if t else " ") or " " for t in texts]
        chunks, cur, n = [], [], 0
        for t in texts:
            if cur and n + len(t) > CHUNK_CHARS:
                chunks.append(cur); cur, n = [], 0
            cur.append(t); n += len(t)
        if cur:
            chunks.append(cur)
        print(f"  [gemini] Batch Mode: {len(texts)} texts -> {len(chunks)} job(s), up to {MAX_PENDING} concurrent", flush=True)
        results = [None] * len(chunks)
        pending = {}  # ci -> job handle
        nxt = 0
        while nxt < len(chunks) or pending:
            while nxt < len(chunks) and len(pending) < MAX_PENDING:
                try:
                    pending[nxt] = self._submit(chunks[nxt], task_type); nxt += 1
                except Exception as e:  # noqa: BLE001
                    if any(s in str(e).upper() for s in ("RESOURCE_EXHAUSTED", "QUOTA", "429")):
                        break  # enqueue budget full — wait for a job to finish, retry later
                    print(f"  [gemini] submit err -> sync: {str(e)[:70]}", flush=True)
                    results[nxt] = self._embed_sync(chunks[nxt], task_type); nxt += 1
            if not pending:
                continue
            time.sleep(20)
            for ci in list(pending):
                try:
                    job = self.client.batches.get(name=pending[ci].name)
                except Exception:  # noqa: BLE001
                    continue
                if "SUCCEEDED" in str(job.state):
                    resps = job.dest.inlined_embed_content_responses
                    results[ci] = np.array([r.response.embedding.values for r in resps], dtype=np.float32)
                    del pending[ci]
                elif "FAILED" in str(job.state):
                    print(f"  [gemini] job {ci} FAILED -> sync fallback", flush=True)
                    results[ci] = self._embed_sync(chunks[ci], task_type); del pending[ci]
        return np.concatenate(results, axis=0)

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
        texts = [text for batch in inputs for text in batch["text"]]
        tt = _task_type(task_metadata, prompt_type)
        if len(texts) > BATCH_THRESHOLD:
            return self._embed_batch(texts, tt)
        return self._embed_sync(texts, tt)


def _n_files(root):
    return sum(len(f) for _, _, f in os.walk(root)) if os.path.isdir(root) else 0


def pull_from_hf():
    os.makedirs(RESULTS, exist_ok=True)
    try:
        snapshot_download(REPO, repo_type="dataset", allow_patterns="results/**", local_dir=CACHE, token=TOKEN)
        print(f"[hf-resume] pulled {_n_files(RESULTS)} files", flush=True)
    except Exception as e:
        print(f"[hf-resume] pull skipped ({str(e)[:90]})", flush=True)


def _upload_once(api):
    if _n_files(RESULTS) == 0:
        return
    for attempt in range(3):
        try:
            api.upload_folder(folder_path=RESULTS, path_in_repo="results", repo_id=REPO,
                              repo_type="dataset", commit_message="gemini sync")
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
    print(f"[gemini] gemini-embedding-001 x {len(tasks)} tasks | batch>{BATCH_THRESHOLD} texts", flush=True)
    api = HfApi(token=TOKEN)
    stop = threading.Event()
    threading.Thread(target=sync_loop, args=(stop, api), daemon=True).start()
    t0 = time.time()
    print("\n=== model: google/gemini-embedding-001 ===", flush=True)
    try:
        mteb.evaluate(GeminiModel(), tasks=tasks, overwrite_strategy=os.environ.get("MTEB_OVERWRITE", "only-missing"),
                      encode_kwargs={"batch_size": 100}, raise_error=False)
        print(f"=== gemini done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"=== gemini FAILED: {type(e).__name__}: {str(e)[:200]} ===", flush=True)
    stop.set()
    _upload_once(api)
    print("=== FLEET RUN COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
