# Running MTEB(por, v2) on Verda (spot + SFS)

Exact recipe for evaluating models on the 22-task suite,
resumable across spot preemptions, with results persisted on a Shared Filesystem.

## Instances
- **Eval (GPU):** `1L40S.20V` spot — L40S 48GB handles every model (incl. 8B at fp16) and runs the 300M `embeddinggemma` in minutes.
- **Upload (CPU):** `CPU.4V.16G` spot — cheap, used only after the run to push results to HF.
- **Storage:** one **Shared Filesystem (SFS), 50 GB** (multi-attach, so it can be reused by both instances and by parallel GPU instances later). Mounts survive spot preemption (`on_spot_discontinue=keep_detached` is the default).

`HF_TOKEN` must have: (1) **Gemma license accepted** (for embeddinggemma), (2) **PortuLex gated access** (for `rrip`), (3) **write** to `mteb-pt/mteb-pt-results` (for the upload).

---

## Phase 1 — GPU eval (`1L40S.20V` spot)

Attach the SFS to the instance, then (SSH in):

```bash
# --- point HF cache + mteb results at the SFS (both persist across preemption) ---
export SFS=/mnt/sfs                 # <-- set to the SFS's actual mount point (check: df -h)
export HF_HOME=$SFS/hf              # model weights + datasets cache (no re-download on restart)
export MTEB_CACHE=$SFS/mteb_cache   # results cache (resumable: only-missing skips finished work)
export HF_TOKEN=hf_xxx              # Gemma + PortuLex(rrip) access
export MTEB_BATCH_SIZE=256          # L40S 48GB; for 8B models later drop to ~32-64
mkdir -p "$HF_HOME" "$MTEB_CACHE"

# --- code (the 22-task MTEB(por) suite) ---
git clone https://github.com/tardellirs/mteb-pt.git
cd mteb-pt
pip install -e .          # pyproject pins datasets>=3, sentence-transformers>=5, transformers>=4.57 (Gemma3-ready)

# --- run embeddinggemma on all 22 tasks ---
python scripts/run_mteb_por_v2.py google/embeddinggemma-300m
```

**Resume after a spot kill:** re-provision, re-mount the SFS, `cd mteb-pt`, re-run the **same** command. Finished `(model, task)` pairs are skipped; nothing re-downloads (cache is on the SFS).

**More models later:** same command, list them — `python scripts/run_mteb_por_v2.py model-a model-b ...`. API models (Gemini/OpenAI) are evaluated separately (no GPU).

---

## Phase 2 — CPU upload (`CPU.4V.16G` spot), after the run

Attach the **same SFS**, then:

```bash
export SFS=/mnt/sfs
export MTEB_CACHE=$SFS/mteb_cache
export HF_TOKEN=hf_xxx
git clone https://github.com/tardellirs/mteb-pt.git && cd mteb-pt && pip install -e .

python scripts/upload_results_to_hf.py --dry-run    # list what would upload
python scripts/upload_results_to_hf.py              # -> mteb-pt/mteb-pt-results (one batched commit, no 429s)
```

Never upload during the GPU run — it wastes GPU time. Results live on the SFS until this step.

---

## Notes
- **PortuLexRRIP** is a headline Classification task (8-way rhetorical-role). Its source dataset () is gated on HF and its license is unspecified — accept gated access before running.
- **Parallelism:** to go faster, provision several `1L40S.20V` spots, all attached to the same SFS, each running a different subset of models. The shared `MTEB_CACHE` + `only-missing` dedupe automatically. ~50 models: ~8-12h on one L40S, ~2-3h across 4.
- **embeddinggemma sanity check:** v1 baseline was `mean_16 = 0.7202` (#6). The v2 scores on the overlapping tasks should land near this.
