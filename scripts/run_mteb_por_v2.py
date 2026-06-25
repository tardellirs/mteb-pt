#!/usr/bin/env python
"""Run the MTEB(por, v2) native suite for one or more embedding models.

Designed for Verda SPOT instances + a mounted BLOCK VOLUME, so a preemption
costs almost nothing:

  - HF_HOME on the volume   -> model weights (8B ~= 16GB!) + datasets cached;
                               a spot restart re-mounts the volume, no re-download.
  - MTEB_CACHE on the volume -> results cached. overwrite_strategy="only-missing"
                               means re-running the SAME command RESUMES: every
                               finished (model, task) is skipped; only unfinished
                               work recomputes. A spot kill loses at most the one
                               task in progress.

Heavy tasks (retrieval, incl. Quati 250k) run FIRST so the expensive work is
checkpointed early -- a later preemption then only re-does a cheap task.

--- Verda setup (block volume mounted at /mnt/vol) ---
    export HF_HOME=/mnt/vol/hf             # model + dataset cache (persisted)
    export MTEB_CACHE=/mnt/vol/mteb_cache  # results cache (persisted, resumable)
    export HF_TOKEN=...                    # gated/private datasets
    export MTEB_BATCH_SIZE=64              # tune per GPU
    uv pip install -e ".[dev]"            # installs mteb_pt + mteb
    python scripts/run_mteb_por_v2.py intfloat/multilingual-e5-small Qwen/Qwen3-Embedding-8B

Resume after a spot kill: just re-run the SAME command. Nothing re-downloads and
finished (model, task) pairs are skipped.

NOTE: this runs the 26 license-clean native tasks. Gated tasks (PortuLex, PAGICO,
Ulysses, SICK-Br) are added to the suite only once their licenses clear.
"""

from __future__ import annotations

import os
import sys
import time

import mteb

import mteb_pt
import mteb_pt.register as register

# Registered for backward-compat but NOT part of the v2 native suite.
_EXCLUDED = {"OffComBR", "CSTNewsClustering", "BBCNewsPTClustering", "TweetSentBR"}
# Heavy categories first -> checkpointed early (spot resilience). Lower = earlier.
_PRIORITY = {"Retrieval": 0, "Reranking": 1, "Clustering": 2}


def v2_tasks():
    """The MTEB(por, v2) suite: the 26 headline tasks + any PENDING tasks (e.g.
    PortuLexRRIP, gated/license-pending) registered in mteb_pt.register."""
    tasks = [cls() for cls in register._TASKS_TO_REGISTER if cls.metadata.name not in _EXCLUDED]
    tasks.append(mteb.get_task("Assin2STS"))
    tasks.sort(key=lambda t: _PRIORITY.get(t.metadata.type, 9))
    return tasks


def main(model_names: list[str]) -> None:
    tasks = v2_tasks()
    bs = int(os.environ.get("MTEB_BATCH_SIZE", "64"))
    n_pending = sum(1 for t in tasks if t.metadata.name in mteb_pt.PENDING_TASKS)
    print(
        f"MTEB(por,v2): {len(tasks)} tasks ({len(tasks) - n_pending} headline + {n_pending} pending: "
        f"{mteb_pt.PENDING_TASKS}) | HF_HOME={os.environ.get('HF_HOME', 'default')} "
        f"| MTEB_CACHE={os.environ.get('MTEB_CACHE', '~/.cache/mteb')} | batch_size={bs}",
        flush=True,
    )
    for mname in model_names:
        t0 = time.time()
        print(f"\n=== model: {mname} ===", flush=True)
        model = mteb.get_model(mname)
        # only-missing => resumable; raise_error=False => one bad task doesn't kill the model run.
        mteb.evaluate(
            model,
            tasks=tasks,
            overwrite_strategy="only-missing",
            encode_kwargs={"batch_size": bs},
            raise_error=False,
        )
        print(f"=== {mname} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    models = sys.argv[1:]
    if not models:
        print("usage: python scripts/run_mteb_por_v2.py <model_name> [<model_name> ...]")
        print(
            "       (models read from a file: python scripts/run_mteb_por_v2.py $(cat models.txt))"
        )
        sys.exit(1)
    main(models)
