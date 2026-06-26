#!/usr/bin/env python
"""Run MTEB(por, v2) with HF-based resume -- NO SFS, runs in ANY location.

Why: a Shared Filesystem (SFS) is location-locked, which pins the whole fleet to
one DataCrunch/Verda region. If that region lacks the cheap A6000 ($0.214) and is
scarce, the fleet is forced onto pricier 80/96GB cards and starves for GPUs. Using
the HF Hub as the resume layer frees every instance to grab the cheapest GPU in ANY
location.

How (spot-resilient, task-level granularity):
  start : pull existing results from the HF results repo -> local MTEB_CACHE
  during: a daemon thread syncs new result files to HF every ~HF_SYNC_SECONDS
          (jittered, so parallel instances don't 429 each other)
  resume: a fresh instance re-pulls from HF; only-missing skips done (model, task)
          pairs. A preemption loses AT MOST the one task in progress (<= ~60s).

Heavy tasks (retrieval, incl. Quati 250k) run FIRST so the expensive work is
checkpointed + uploaded early; the mid-Quati exposure window is just the first few
minutes of each model. (mteb's only-missing is task-atomic -- it cannot resume
*within* a task, so a mid-Quati preemption re-encodes; heavy-first bounds that.)

Env:
  HF_RESULTS_REPO  (default mteb-pt/mteb-pt-results)  results dataset repo
  MTEB_CACHE       (default ~/.cache/mteb)            results cache root
  HF_SYNC_SECONDS  (default 60)                       sync cadence
  HF_TOKEN                                            write access to the repo
  MTEB_BATCH_SIZE  (default 64)
  MTEB_TASKS       (optional comma-list)              restrict tasks (testing)
"""

from __future__ import annotations

import os
import random
import sys
import shutil
import threading
import time

import mteb

import mteb_pt
import mteb_pt.register as register
from huggingface_hub import HfApi, snapshot_download

REPO = os.environ.get("HF_RESULTS_REPO", "mteb-pt/mteb-pt-results")
CACHE = os.environ.get("MTEB_CACHE", os.path.expanduser("~/.cache/mteb"))
RESULTS = os.path.join(CACHE, "results")
SYNC_EVERY = int(os.environ.get("HF_SYNC_SECONDS", "60"))
TOKEN = os.environ.get("HF_TOKEN")

_EXCLUDED = {"OffComBR", "CSTNewsClustering", "BBCNewsPTClustering", "TweetSentBR"}
_PRIORITY = {"Retrieval": 0, "Reranking": 1, "Clustering": 2}


def v2_tasks() -> list:
    tasks = [cls() for cls in register._TASKS_TO_REGISTER if cls.metadata.name not in _EXCLUDED]
    tasks.append(mteb.get_task("Assin2STS"))
    only = os.environ.get("MTEB_TASKS")
    if only:
        keep = {x.strip() for x in only.split(",")}
        tasks = [t for t in tasks if t.metadata.name in keep]
    _first = os.environ.get("MTEB_FIRST_TASK", "HateBR")  # quick task before Quati for early validation
    tasks.sort(key=lambda t: (-1 if t.metadata.name == _first else _PRIORITY.get(t.metadata.type, 9)))
    return tasks


def _n_files(root: str) -> int:
    return sum(len(f) for _, _, f in os.walk(root)) if os.path.isdir(root) else 0


def pull_from_hf() -> None:
    """Download existing results -> MTEB_CACHE so only-missing skips them."""
    os.makedirs(RESULTS, exist_ok=True)
    try:
        snapshot_download(
            REPO, repo_type="dataset", allow_patterns="results/**",
            local_dir=CACHE, token=TOKEN,
        )
        print(f"[hf-resume] pulled existing results from {REPO} ({_n_files(RESULTS)} files cached)", flush=True)
    except Exception as e:  # new/empty repo, offline, etc. -> start fresh
        print(f"[hf-resume] pull skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)


def _upload_once(api: HfApi) -> None:
    if _n_files(RESULTS) == 0:
        return
    for attempt in range(3):
        try:
            api.upload_folder(
                folder_path=RESULTS, path_in_repo="results",
                repo_id=REPO, repo_type="dataset", commit_message="hf-resume sync",
            )
            return
        except Exception as e:  # noqa: BLE001 -- 429 / transient -> back off + retry
            if "429" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1) + random.uniform(0, 5))
                continue
            print(f"[hf-resume] sync err ({type(e).__name__}: {str(e)[:100]})", flush=True)
            return


def sync_loop(stop: threading.Event, api: HfApi) -> None:
    time.sleep(SYNC_EVERY * random.uniform(0.5, 1.5))  # de-phase parallel instances
    while not stop.is_set():
        _upload_once(api)
        stop.wait(SYNC_EVERY * random.uniform(0.85, 1.15))


def main(model_names: list[str]) -> None:
    bs = int(os.environ.get("MTEB_BATCH_SIZE", "64"))
    pull_from_hf()
    tasks = v2_tasks()
    print(
        f"[hf-resume] {len(tasks)} tasks | repo={REPO} | cache={CACHE} "
        f"| sync={SYNC_EVERY}s | batch={bs}",
        flush=True,
    )
    api = HfApi(token=TOKEN)
    stop = threading.Event()
    threading.Thread(target=sync_loop, args=(stop, api), daemon=True).start()
    hub = os.environ.get("HF_HUB_CACHE", "")
    for mname in model_names:
        t0 = time.time()
        model = None
        print(f"\n=== model: {mname} ===", flush=True)
        try:
            try:
                model = mteb.get_model(mname)
            except Exception as ge:  # noqa: BLE001
                # KaLM-Gemma3 is text-only (Gemma3TextModel) but tokenizer_config.json
                # carries a stray processor_class=Gemma3Processor; ST 5.x unconditionally
                # calls AutoProcessor -> OSError. Strip that key in the snapshot, then load
                # the patched local path (pooling+normalize come from modules.json).
                print(f"  [load] get_model failed ({str(ge)[:55]}); patching + SentenceTransformer", flush=True)
                import json as _json
                import torch as _torch
                from huggingface_hub import snapshot_download as _sd
                from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
                _local = _sd(mname)
                _tc = os.path.join(_local, "tokenizer_config.json")
                if os.path.exists(_tc):
                    _cfg = _json.load(open(_tc))
                    _changed = _cfg.pop("processor_class", None) is not None
                    if _cfg.get("padding_side") != "left":  # Gemma3 last-token pooling needs LEFT-pad
                        _cfg["padding_side"] = "left"
                        _changed = True
                    if _changed:
                        _json.dump(_cfg, open(_tc, "w"), ensure_ascii=False, indent=2)
                        print("  [load] patched tokenizer_config (processor_class strip + padding_side=left)", flush=True)
                model = SentenceTransformerEncoderWrapper(
                    _local, trust_remote_code=True,
                    model_kwargs={"torch_dtype": _torch.bfloat16},
                )
            mteb.evaluate(
                model, tasks=tasks, overwrite_strategy="only-missing",
                encode_kwargs={"batch_size": bs}, raise_error=False,
            )
            print(f"=== {mname} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
        except Exception as e:  # a bad model must not halt the fleet
            print(f"=== {mname} FAILED: {type(e).__name__}: {str(e)[:200]} ===", flush=True)
        finally:
            del model
            if hub and os.path.isdir(hub):  # free ~model-size disk for the next model
                shutil.rmtree(hub, ignore_errors=True)
    stop.set()
    print("[hf-resume] final sync...", flush=True)
    _upload_once(api)
    print("=== FLEET RUN COMPLETE ===", flush=True)


if __name__ == "__main__":
    models = sys.argv[1:]
    if not models:
        print("usage: run_mteb_hfresume.py <model> [<model> ...]")
        sys.exit(1)
    main(models)
