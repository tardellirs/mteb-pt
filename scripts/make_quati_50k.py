#!/usr/bin/env python
"""Create tardellirs/mteb-pt-quati-50k: subsample the 250k Quati corpus down to 50k.

Keeps ALL judged passages (every corpus-id referenced in qrels) plus a fixed
random sample (seed 42) of the remaining passages, to 50k total. Queries and
qrels are carried over UNCHANGED. The 50k pool is therefore a strict subset of
the pinned 250k pool (which is itself a subset of the original 1M ClueWeb22-PT).

Why: Quati is ~95% of the per-model encoding load at 250k (1.1B chars vs ~63M for
all other 21 tasks combined). Cutting to 50k makes the whole MTEB(por) suite ~4x
cheaper to run and re-run, while a 50k pool is still a large, well-discriminating
retrieval corpus (all gold preserved -> recall ceiling unchanged; ~48k distractors
keep it hard).

Run on a POD (not the Mac) — it loads the 250k corpus into memory:

    set -a; source .env; set +a
    python scripts/make_quati_50k.py

At the end it prints the new dataset revision SHA — paste it into
src/mteb_pt/tasks/retrieval/por/quati.py (_REVISION) and the task is live.
"""
from __future__ import annotations

import random

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

SRC = "tardellirs/mteb-pt-quati-250k"
SRC_REV = "7440d16aa3a53c037e63a16591c461210b72dd82"
DST = "mteb-pt/quati-50k"
TARGET = 50_000
SEED = 42


def main() -> None:
    corpus = list(load_dataset(SRC, "corpus", split="test", revision=SRC_REV))
    queries = load_dataset(SRC, "queries", split="test", revision=SRC_REV)
    qrels = load_dataset(SRC, "qrels", split="test", revision=SRC_REV)

    gold = {str(r["corpus-id"]) for r in qrels}
    gold_rows = [r for r in corpus if str(r["_id"]) in gold]
    rest = [r for r in corpus if str(r["_id"]) not in gold]
    rng = random.Random(SEED)
    rng.shuffle(rest)
    n_distract = max(0, TARGET - len(gold_rows))
    sub = gold_rows + rest[:n_distract]

    print(
        f"corpus 250k={len(corpus)} | gold(judged)={len(gold_rows)} "
        f"| distractors={n_distract} | -> 50k pool={len(sub)}"
    )
    assert gold.issubset({str(r["_id"]) for r in sub}), "gold passage dropped!"

    # corpus subsampled; queries + qrels carried over unchanged
    Dataset.from_list(sub).push_to_hub(DST, "corpus", split="test")
    Dataset.from_list(list(queries)).push_to_hub(DST, "queries", split="test")
    Dataset.from_list(list(qrels)).push_to_hub(DST, "qrels", split="test")

    sha = HfApi().dataset_info(DST).sha
    print(f"\nDONE -> {DST}")
    print(f'  _REVISION = "{sha}"   # <- cole em quati.py')


if __name__ == "__main__":
    main()
