#!/usr/bin/env python
"""Run MTEB(por, v2) reranking tasks for Voyage cross-encoder rerankers.

Supports rerank-2.5 and rerank-2.5-lite via the Voyage /v1/rerank endpoint.

Cross-encoder rerankers score (query, document) pairs; they cannot produce
embeddings, so they ONLY run the Reranking tasks:
  - QuatiReranking    (web / general PT-BR, 50 queries × ~100 candidates)
  - JurisTCUReranking (legal / TCU, 150 queries × ~100 candidates)

mteb 2.12 interface for cross-encoders (CrossEncoderProtocol):
  predict(inputs1, inputs2, *, task_metadata, hf_split, hf_subset) -> np.Array
  - inputs1: DataLoader yielding batches for queries  (batch["text"] or batch["query"])
  - inputs2: DataLoader yielding batches for documents (batch["text"])
  - Returns: flat 1-D array of relevance scores, one per (query, doc) pair,
             in the SAME ORDER as the pairs from inputs1/inputs2.

Voyage rerank(query, documents, model, top_k) returns results sorted by
relevance_score descending with .index referencing the original doc position.
We reconstruct the original-order score array.

Throttle: Tier 1 defaults 900 RPM (conservative). Each call is 1 request
(1 query + batch of candidates). JurisTCU = 150 calls, Quati = 50 calls → cheap.

Env vars:
  VOYAGE_API_KEY        (required)
  VOYAGE_RERANK_MODEL   rerank-2.5 (default) | rerank-2.5-lite
  VOYAGE_RPM            requests per minute cap (default 900)
  HF_TOKEN              HuggingFace token for result upload
  HF_RESULTS_REPO       dataset repo for results (default mteb-pt/mteb-pt-results)
  MTEB_CACHE            local cache dir (default ~/.cache/mteb)
  HF_SYNC_SECONDS       upload interval in seconds (default 120)
  MTEB_TASKS            comma-list to restrict tasks (e.g. QuatiReranking)

Usage:
  VOYAGE_RERANK_MODEL=rerank-2.5 python scripts/run_mteb_voyage_rerank.py
  VOYAGE_RERANK_MODEL=rerank-2.5-lite python scripts/run_mteb_voyage_rerank.py
"""
from __future__ import annotations

import os
import random
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

import mteb
import mteb_pt
import mteb_pt.register as register
from mteb.models.model_meta import ModelMeta, ScoringFunction
from huggingface_hub import HfApi, snapshot_download
import voyageai

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
REPO = os.environ.get("HF_RESULTS_REPO", "mteb-pt/mteb-pt-results")
CACHE = os.environ.get("MTEB_CACHE", os.path.expanduser("~/.cache/mteb"))
RESULTS = os.path.join(CACHE, "results")
SYNC_EVERY = int(os.environ.get("HF_SYNC_SECONDS", "120"))
TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2.5")
RPM = float(os.environ.get("VOYAGE_RPM", "900"))  # Tier 1 conservative default

# Only these two tasks can use a cross-encoder reranker
_RERANK_TASKS = {"QuatiReranking", "JurisTCUReranking"}


# ---------------------------------------------------------------------------
# Task list
# ---------------------------------------------------------------------------
def rerank_tasks() -> list:
    """Return the reranking task instances, filtered by MTEB_TASKS if set."""
    tasks = [cls() for cls in register._TASKS_TO_REGISTER
             if cls.metadata.name in _RERANK_TASKS]
    only = os.environ.get("MTEB_TASKS")
    if only:
        keep = {x.strip() for x in only.split(",")}
        tasks = [t for t in tasks if t.metadata.name in keep]
    # JurisTCU (150 queries) before Quati (50 queries) — more queries first
    tasks.sort(key=lambda t: (0 if t.metadata.name == "JurisTCUReranking" else 1))
    return tasks


# ---------------------------------------------------------------------------
# Voyage Cross-Encoder Wrapper
# ---------------------------------------------------------------------------
class VoyageReranker:
    """mteb CrossEncoderProtocol backed by the Voyage /v1/rerank API.

    mteb wraps this in SearchCrossEncoderWrapper which calls .predict() with
    two DataLoaders: inputs1 (queries) and inputs2 (documents). The pairs are
    already aligned (inputs1[i] goes with inputs2[i]). We group by query,
    call voyageai.Client().rerank(query, candidates) once per unique query,
    then reconstruct the flat score array in the original pair order.
    """

    def __init__(self, model_id: str = "rerank-2.5"):
        self.model_id = model_id
        self.client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self._lock = threading.Lock()
        self._last_req = 0.0
        self._min_gap = 60.0 / RPM  # seconds between requests (RPM cap)

        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=f"voyage/{model_id}",
            revision="api",
            release_date="2024-10-01",
            languages=["por-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=32000,
            embed_dim=None,
            license=None,
            open_weights=False,
            public_training_code=None,
            public_training_data=None,
            framework=["API"],
            similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=False,
            training_datasets=None,
        )

    def _throttle(self) -> None:
        with self._lock:
            wait = self._min_gap - (time.time() - self._last_req)
            if wait > 0:
                time.sleep(wait)
            self._last_req = time.time()

    def _rerank_one(self, query: str, docs: list[str]) -> list[float]:
        """Call Voyage rerank for one query + its candidate documents.

        Returns scores in the ORIGINAL document order (same index as docs list).
        """
        for delay in (5, 20, 45, 90, None):
            self._throttle()
            try:
                response = self.client.rerank(
                    query=query,
                    documents=docs,
                    model=self.model_id,
                    top_k=len(docs),  # score ALL candidates
                )
                # response.results: sorted by relevance_score DESC,
                # each result has .index (original doc position) and .relevance_score
                scores = [0.0] * len(docs)
                for result in response.results:
                    scores[result.index] = float(result.relevance_score)
                return scores
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if delay is None:
                    print(f"  [voyage-rerank] give-up: {str(e)[:110]}", flush=True)
                    return [0.0] * len(docs)
                extra = 45 if ("rate" in msg or "429" in msg or "limit" in msg) else 0
                print(f"  [voyage-rerank] retry ({str(e)[:60]}) wait {delay+extra}s",
                      flush=True)
                time.sleep(delay + extra)
        return [0.0] * len(docs)  # unreachable, but satisfies type checker

    def predict(
        self,
        inputs1: "DataLoader[BatchedInput]",
        inputs2: "DataLoader[BatchedInput]",
        *,
        task_metadata: "TaskMetadata",
        hf_split: str,
        hf_subset: str,
        prompt_type: "PromptType | None" = None,
        **kwargs: Any,
    ) -> "Array":
        """Score all (query, document) pairs passed in by SearchCrossEncoderWrapper.

        inputs1 and inputs2 are aligned: inputs1[i] is the query for inputs2[i].
        We group consecutive identical queries (since mteb expands each query
        against all its candidates), make one Voyage API call per unique query,
        and return a flat score array in the original pair order.
        """
        # Extract raw text from dataloaders
        # queries loader: may have "query" key (from _combine_queries_with_instruction_text)
        # or just "text" (when passed through _corpus_to_dict with document prompt_type).
        # We try "query" first, fall back to "text".
        queries_raw: list[str] = []
        for batch in inputs1:
            if "query" in batch:
                queries_raw.extend(batch["query"])
            else:
                queries_raw.extend(batch["text"])

        docs_raw: list[str] = []
        for batch in inputs2:
            docs_raw.extend(batch["text"])

        assert len(queries_raw) == len(docs_raw), (
            f"Mismatched queries ({len(queries_raw)}) and docs ({len(docs_raw)})"
        )

        n_pairs = len(queries_raw)
        print(f"  [voyage-rerank] scoring {n_pairs} (query, doc) pairs "
              f"with {self.model_id}", flush=True)

        # Group consecutive pairs by query text to make one API call per query
        all_scores = np.zeros(n_pairs, dtype=np.float32)
        i = 0
        q_idx = 0
        while i < n_pairs:
            query = queries_raw[i]
            # Find the end of this query's block
            j = i + 1
            while j < n_pairs and queries_raw[j] == query:
                j += 1
            # docs for this query are indices [i, j)
            candidate_docs = docs_raw[i:j]
            scores = self._rerank_one(query, candidate_docs)
            all_scores[i:j] = scores
            print(f"  [voyage-rerank] query {q_idx} → {len(candidate_docs)} docs scored",
                  flush=True)
            i = j
            q_idx += 1

        return all_scores


# ---------------------------------------------------------------------------
# HF persistence helpers (identical pattern to run_mteb_voyage.py)
# ---------------------------------------------------------------------------
def _n_files(root: str) -> int:
    return sum(len(f) for _, _, f in os.walk(root)) if os.path.isdir(root) else 0


def pull_from_hf() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    try:
        snapshot_download(
            REPO, repo_type="dataset",
            allow_patterns=f"results/voyage__{MODEL_ID}/**",
            local_dir=CACHE,
            token=TOKEN,
        )
        print(f"[hf-resume] pulled {_n_files(RESULTS)} files", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[hf-resume] pull skipped ({str(e)[:90]})", flush=True)


def _upload_once(api: HfApi) -> None:
    if _n_files(RESULTS) == 0:
        return
    for attempt in range(3):
        try:
            api.upload_folder(
                folder_path=RESULTS,
                path_in_repo="results",
                repo_id=REPO,
                repo_type="dataset",
                commit_message=f"voyage-rerank {MODEL_ID} sync",
                allow_patterns=[f"voyage__{MODEL_ID}/**"],
            )
            return
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1) + random.uniform(0, 5))
                continue
            print(f"[hf-resume] sync err ({str(e)[:90]})", flush=True)
            return


def sync_loop(stop: threading.Event, api: HfApi) -> None:
    time.sleep(SYNC_EVERY * random.uniform(0.5, 1.5))
    while not stop.is_set():
        _upload_once(api)
        stop.wait(SYNC_EVERY)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pull_from_hf()
    tasks = rerank_tasks()
    if not tasks:
        print("[voyage-rerank] No tasks to run (check MTEB_TASKS filter)", flush=True)
        sys.exit(0)

    model = VoyageReranker(MODEL_ID)
    print(
        f"[voyage-rerank] model={MODEL_ID} | RPM={RPM} | tasks={[t.metadata.name for t in tasks]} | repo={REPO}",
        flush=True,
    )

    api = HfApi(token=TOKEN)
    stop = threading.Event()
    threading.Thread(target=sync_loop, args=(stop, api), daemon=True).start()

    t0 = time.time()
    print(f"\n=== voyage-rerank: {MODEL_ID} ===", flush=True)
    try:
        mteb.evaluate(
            model,
            tasks=tasks,
            overwrite_strategy="only-missing",
            encode_kwargs={"batch_size": 32},
            raise_error=False,
        )
        print(f"=== voyage-rerank done in {(time.time() - t0) / 60:.1f} min ===",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(
            f"=== voyage-rerank FAILED: {type(e).__name__}: {str(e)[:200]} ===",
            flush=True,
        )

    stop.set()
    _upload_once(api)
    print("=== VOYAGE RERANK RUN COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
