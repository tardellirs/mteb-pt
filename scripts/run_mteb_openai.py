#!/usr/bin/env python
"""Run MTEB(por, v2) for an OpenAI embedding model via the OpenAI API (DIRECT),
using the Batch API (50% off) for big corpora.

Hybrid: big encode calls (corpora > OPENAI_BATCH_THRESHOLD texts) -> OpenAI Batch API
($0.065/1M for 3-large, 50% off; file-based JSONL, concurrent batches); small encode
calls -> sync embeddings (input list up to 2048). mteb AbsEncoder so scores match the
open-weight + gateway fleets. HF-resume (pull at start, daemon sync, only-missing).

Env: OPENAI_API_KEY, OPENAI_MODEL (default text-embedding-3-large), HF_RESULTS_REPO,
     MTEB_CACHE, HF_SYNC_SECONDS, OPENAI_BATCH_THRESHOLD (default 2000), MTEB_TASKS.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
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
SYNC_EVERY = int(os.environ.get("HF_SYNC_SECONDS", "120"))
TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("OPENAI_MODEL", "text-embedding-3-large")
DIM = 3072 if "large" in MODEL_ID else 1536
BATCH_THRESHOLD = int(os.environ.get("OPENAI_BATCH_THRESHOLD", "2000"))
# OpenAI org enqueued-token limit is 3M for 3-large. Chunk by chars (~3.5M -> ~0.9M tok)
# and keep MAX_PENDING * chunk under 3M enqueued.
CHUNK_CHARS = int(os.environ.get("OPENAI_CHUNK_CHARS", "3500000"))
MAX_PENDING = int(os.environ.get("OPENAI_MAX_PENDING", "2"))
_EXCLUDED = {"OffComBR", "CSTNewsClustering", "BBCNewsPTClustering", "TweetSentBR"}
_PRIORITY = {"Retrieval": 0, "Reranking": 1, "Clustering": 2}


def v2_tasks() -> list:
    tasks = [cls() for cls in register._TASKS_TO_REGISTER if cls.metadata.name not in _EXCLUDED]
    tasks.append(mteb.get_task("Assin2STS"))
    only = os.environ.get("MTEB_TASKS")
    if only:
        keep = {x.strip() for x in only.split(",")}
        tasks = [t for t in tasks if t.metadata.name in keep]
    excl = os.environ.get("MTEB_EXCLUDE")
    if excl:
        drop = {x.strip() for x in excl.split(",")}
        tasks = [t for t in tasks if t.metadata.name not in drop]
    tasks.sort(key=lambda t: _PRIORITY.get(t.metadata.type, 9))
    return tasks


class OpenAIBatchModel(AbsEncoder):
    """mteb encoder for an OpenAI embedding model (sync small, Batch API for big)."""

    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120.0, max_retries=2)  # timeout pra não pendurar
        self.mteb_model_meta = ModelMeta(
            loader=None, name=f"openai/{MODEL_ID}", revision="api",
            release_date="2024-01-25", languages=["por-Latn"], n_parameters=None,
            memory_usage_mb=None, max_tokens=8191, embed_dim=DIM, license=None,
            open_weights=False, public_training_code=None, public_training_data=None,
            framework=["API"], similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=False, training_datasets=None,
        )

    def _embed_sync(self, texts):
        out = []
        prepped = [(t[:30000] if t else " ") or " " for t in texts]
        i = 0
        while i < len(prepped):
            # batch por token estimado (chars/4) até ~250K, sob o limite de 300K/request da OpenAI
            chunk, toks = [], 0
            while i < len(prepped) and (not chunk or (len(chunk) < 2048 and toks + len(prepped[i]) // 4 < 250000)):
                chunk.append(prepped[i]); toks += len(prepped[i]) // 4; i += 1
            for delay in (2, 5, 15, 30, 60, None):
                try:
                    r = self.client.embeddings.create(model=MODEL_ID, input=chunk)
                    out.extend([d.embedding for d in r.data]); break
                except Exception as e:  # noqa: BLE001
                    if delay is None:
                        print(f"  [openai] sync give-up: {str(e)[:90]}", flush=True)
                        out.extend([[0.0] * DIM] * len(chunk)); break
                    time.sleep(delay)
        return np.array(out, dtype=np.float32)

    def _submit_chunk(self, texts, start):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for j, t in enumerate(texts):
            f.write(json.dumps({
                "custom_id": str(start + j), "method": "POST", "url": "/v1/embeddings",
                "body": {"model": MODEL_ID, "input": (t[:30000] if t else " ") or " "},
            }) + "\n")
        f.close()
        for delay in (5, 15, 30, 60, None):
            try:
                up = self.client.files.create(file=open(f.name, "rb"), purpose="batch")
                b = self.client.batches.create(input_file_id=up.id, endpoint="/v1/embeddings",
                                                completion_window="24h")
                os.unlink(f.name)
                return b.id
            except Exception as e:  # noqa: BLE001
                if delay is None:
                    os.unlink(f.name); raise
                print(f"  [openai] submit retry ({str(e)[:70]})", flush=True); time.sleep(delay)

    def _embed_batch(self, texts):
        """OpenAI Batch API (50% off) with a concurrency window under the org 3M
        enqueued-token limit; failed/expired batches fall back to sync (no zero vecs)."""
        texts = [(t[:30000] if t else " ") or " " for t in texts]
        chunks, starts, cur, cur_start, n = [], [], [], 0, 0
        for i, t in enumerate(texts):
            if cur and n + len(t) > CHUNK_CHARS:
                chunks.append(cur); starts.append(cur_start); cur, n, cur_start = [], 0, i
            cur.append(t); n += len(t)
        if cur:
            chunks.append(cur); starts.append(cur_start)
        print(f"  [openai] Batch: {len(texts)} texts -> {len(chunks)} chunk(s), <= {MAX_PENDING} concurrent (3M-tok cap)", flush=True)
        results = {}
        pending = {}  # ci -> batch_id

        def _sync_fill(ci):
            emb = self._embed_sync(chunks[ci])
            for j in range(len(chunks[ci])):
                results[str(starts[ci] + j)] = emb[j]

        nxt = 0
        while nxt < len(chunks) or pending:
            while nxt < len(chunks) and len(pending) < MAX_PENDING:
                try:
                    pending[nxt] = self._submit_chunk(chunks[nxt], starts[nxt]); nxt += 1
                except Exception as e:  # noqa: BLE001
                    if any(s in str(e).lower() for s in ("token_limit", "enqueued", "rate", "429")):
                        break  # enqueue budget full -> wait for a batch to finish
                    print(f"  [openai] submit err -> sync: {str(e)[:60]}", flush=True)
                    _sync_fill(nxt); nxt += 1
            if not pending:
                continue
            time.sleep(30)
            for ci in list(pending):
                try:
                    b = self.client.batches.retrieve(pending[ci])
                except Exception:  # noqa: BLE001
                    continue
                if b.status == "completed" and b.output_file_id:
                    content = self.client.files.content(b.output_file_id).read()
                    for line in content.decode().splitlines():
                        o = json.loads(line); resp = o.get("response", {})
                        if resp.get("status_code") == 200:
                            data = resp.get("body", {}).get("data", [])
                            if data:
                                results[o["custom_id"]] = np.array(data[0]["embedding"], dtype=np.float32)
                    del pending[ci]
                elif b.status in ("failed", "expired", "cancelled"):
                    print(f"  [openai] batch ci={ci} {b.status} -> sync fallback", flush=True)
                    _sync_fill(ci); del pending[ci]
        return np.array([results.get(str(i), np.zeros(DIM, dtype=np.float32)) for i in range(len(texts))], dtype=np.float32)

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
        texts = [text for batch in inputs for text in batch["text"]]
        if len(texts) > BATCH_THRESHOLD:
            return self._embed_batch(texts)
        return self._embed_sync(texts)


def _n_files(root):
    return sum(len(f) for _, _, f in os.walk(root)) if os.path.isdir(root) else 0


def pull_from_hf():
    os.makedirs(RESULTS, exist_ok=True)
    try:
        snapshot_download(REPO, repo_type="dataset", allow_patterns=f"results/openai__{MODEL_ID}/**", local_dir=CACHE, token=TOKEN)
        print(f"[hf-resume] pulled {_n_files(RESULTS)} files", flush=True)
    except Exception as e:
        print(f"[hf-resume] pull skipped ({str(e)[:90]})", flush=True)


def _upload_once(api):
    if _n_files(RESULTS) == 0:
        return
    for attempt in range(3):
        try:
            api.upload_folder(folder_path=RESULTS, path_in_repo="results", repo_id=REPO,
                              repo_type="dataset", commit_message=f"openai sync ({MODEL_ID})",
                              allow_patterns=[f"openai__{MODEL_ID}/**"])
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
    print(f"[openai] {MODEL_ID} x {len(tasks)} tasks | batch>{BATCH_THRESHOLD} texts | dim={DIM}", flush=True)
    api = HfApi(token=TOKEN)
    stop = threading.Event()
    threading.Thread(target=sync_loop, args=(stop, api), daemon=True).start()
    t0 = time.time()
    print(f"\n=== model: openai/{MODEL_ID} ===", flush=True)
    try:
        mteb.evaluate(OpenAIBatchModel(), tasks=tasks, overwrite_strategy="only-missing",
                      encode_kwargs={"batch_size": 256}, raise_error=False)
        print(f"=== openai done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"=== openai FAILED: {type(e).__name__}: {str(e)[:200]} ===", flush=True)
    stop.set()
    _upload_once(api)
    print("=== FLEET RUN COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
