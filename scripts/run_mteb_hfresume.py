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

# ─── Nemotron-8B compat shim (guarded by MTEB_NEMOTRON_SHIM) ────────────────────
# nvidia/llama-embed-nemotron-8b's loader pins transformers==4.51.0 exact because its
# custom LlamaBidirectionalModel overrides _update_causal_mask() for BIDIRECTIONAL
# attention. transformers 5.x REMOVED that method, so the override is silently ignored
# → model runs CAUSAL → wrong embeddings (no crash). Fix: spoof the version guard AND
# set is_causal=False so transformers 5.x routes to its native create_bidirectional_mask().
if os.environ.get("MTEB_NEMOTRON_SHIM"):
    import transformers as _tf
    _tf.__version__ = "4.51.0"
    from transformers.models.llama.configuration_llama import LlamaConfig as _LC
    _orig_lc_init = _LC.__init__
    def _patched_lc_init(self, **kw):
        _orig_lc_init(self, **kw)
        if getattr(self, "use_bidirectional_attention", False):
            self.is_causal = False
    _LC.__init__ = _patched_lc_init
    print("[shim] nemotron bidirectional shim active (tf->4.51.0 spoof + is_causal=False)", flush=True)

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
_MODEL_SLUGS: list[str] = []  # scoped upload (anti-clobber): so os modelos desta run


def v2_tasks() -> list:
    tasks = [cls() for cls in register._TASKS_TO_REGISTER if cls.metadata.name not in _EXCLUDED]
    tasks.append(mteb.get_task("Assin2STS"))
    only = os.environ.get("MTEB_TASKS")
    if only:
        keep = {x.strip() for x in only.split(",")}
        tasks = [t for t in tasks if t.metadata.name in keep]
    excl = os.environ.get("MTEB_EXCLUDE")  # comma-list to drop (e.g. Quati now, run it separately later)
    if excl:
        drop = {x.strip() for x in excl.split(",")}
        tasks = [t for t in tasks if t.metadata.name not in drop]
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
            REPO, repo_type="dataset",
            allow_patterns=[f"results/{sl}/**" for sl in _MODEL_SLUGS],
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
                allow_patterns=[sl + "/**" for sl in _MODEL_SLUGS],
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
    global _MODEL_SLUGS
    _MODEL_SLUGS = [m.replace("/", "__") for m in model_names]
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
    import base64 as _b64mp, json as _jsonmp
    _mpe = os.environ.get("MTEB_MODEL_PROMPTS_B64")
    _MP = _jsonmp.loads(_b64mp.b64decode(_mpe)) if _mpe else None
    for mname in model_names:
        t0 = time.time()
        model = None
        print(f"\n=== model: {mname} ===", flush=True)
        try:
            try:
                # fp16 overflow fix: Mistral/Qwen2 decoder-LLM embedders emit inf/NaN on
                # long inputs (EnemEssay/AssinRTE/InferBR) under fp16 (logits > 65504).
                # bf16 has fp32 exponent range -> no overflow. Patch loader_kwargs->bfloat16.
                _FB16 = {"intfloat/e5-mistral-7b-instruct", "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                         "Salesforce/SFR-Embedding-Mistral", "Salesforce/SFR-Embedding-2_R"}
                _FB16 |= {m for m in os.environ.get("MTEB_FORCE_BF16", "").split(",") if m}
                if mname in _FB16:
                    import torch as _torch
                    try:
                        _meta = mteb.get_model_meta(mname)
                        _lk = dict(getattr(_meta, "loader_kwargs", None) or {})
                        _mk = dict(_lk.get("model_kwargs", {}))
                        _mk["dtype"] = _torch.bfloat16
                        _lk["model_kwargs"] = _mk
                        _meta.loader_kwargs = _lk
                        print(f"  [bf16] dtype=bfloat16 forced for {mname} (fp16 overflow fix)", flush=True)
                    except Exception as _be:  # noqa: BLE001
                        print(f"  [bf16] patch failed: {str(_be)[:60]}", flush=True)
                model = mteb.get_model(mname)
                if _MP and hasattr(model, "model_prompts"):
                    model.model_prompts = _MP
                    if hasattr(getattr(model, "model", None), "prompts"):
                        model.model.prompts = _MP
                    print(f"  [load] get_model model_prompts override: {list(_MP.keys())}", flush=True)
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
                import base64 as _b64mp
                _mpe = os.environ.get("MTEB_MODEL_PROMPTS_B64")
                _mp = _json.loads(_b64mp.b64decode(_mpe)) if _mpe else None
                if _mp:
                    print(f"  [load] model_prompts aplicado: {list(_mp.keys())}", flush=True)
                model = SentenceTransformerEncoderWrapper(
                    _local, trust_remote_code=True,
                    model_kwargs={"torch_dtype": _torch.bfloat16},
                    model_prompts=_mp,
                )
                try:
                    import re as _re2
                    _msha = _re2.search(r"snapshots[/_]+([0-9a-f]{40})", _local)
                    if getattr(model, "mteb_model_meta", None) is not None:
                        model.mteb_model_meta.name = mname
                        if _msha:
                            model.mteb_model_meta.revision = _msha.group(1)
                        print(f"  [load] meta limpo -> {mname} @ {(_msha.group(1)[:8] if _msha else '?')}", flush=True)
                except Exception as _e2:
                    print(f"  [load] meta-fix falhou: {str(_e2)[:50]}", flush=True)
            # (removido o wrapper _te de truncacao do BRTaxQAR: reconstruia o input do encode
            #  como lista e quebrava o RETRIEVAL com "'list' object has no attribute 'dataset'".
            #  BRTaxQAR roda sem truncacao -- nao causa OOM na pratica.)
            # Tasks de texto LONGO (BRTaxQAR docs-monstro, JurisTCUClustering corpus grande)
            # estouram VRAM em batch grande MESMO em modelo pequeno (atencao O(seq^2)) e
            # ate em A100-80. -> cap de batch <=16 SO nessas; o resto mantem o batch padrao.
            _HEAVY = {"BRTaxQAR", "JurisTCUClusteringP2P"}
            _ov = os.environ.get("MTEB_OVERWRITE", "only-missing")
            _light = [t for t in tasks if t.metadata.name not in _HEAVY]
            _heavy = [t for t in tasks if t.metadata.name in _HEAVY]
            if _light:
                mteb.evaluate(model, tasks=_light, overwrite_strategy=_ov,
                              encode_kwargs={"batch_size": bs}, raise_error=False)
            if _heavy:
                mteb.evaluate(model, tasks=_heavy, overwrite_strategy=_ov,
                              encode_kwargs={"batch_size": min(bs, 16)}, raise_error=False)
            # ROOT-FIX anti-LIXO: o ST-fallback grava no slug-path-local
            # (__root__hfmodels__models--ORG--MODEL__snapshots__SHA/no_revision_available/).
            # mteb_model_meta.name NAO redireciona -> renomeia o LIXO pro slug limpo aqui,
            # no rev ja existente (do pull) p/ nao criar rev duplicado. Sem isso o upload
            # escopado nao casa e os resultados nunca sobem (falha silenciosa).
            try:
                import re as _rl
                clean_slug = mname.replace("/", "__")
                for d in list(os.listdir(RESULTS)):
                    mm = _rl.search(r"models--(.+?)--(.+?)__snapshots__([0-9a-f]{40})", d)
                    if not mm or ("%s__%s" % (mm.group(1), mm.group(2))) != clean_slug:
                        continue
                    src = os.path.join(RESULTS, d, "no_revision_available")
                    if not os.path.isdir(src):
                        continue
                    cdir = os.path.join(RESULTS, clean_slug)
                    revs = [r for r in os.listdir(cdir) if os.path.isdir(os.path.join(cdir, r))] if os.path.isdir(cdir) else []
                    target_rev = revs[0] if revs else mm.group(3)  # rev do pull, senao o SHA
                    dst = os.path.join(RESULTS, clean_slug, target_rev)
                    os.makedirs(dst, exist_ok=True)
                    for fn in os.listdir(src):
                        shutil.move(os.path.join(src, fn), os.path.join(dst, fn))
                    shutil.rmtree(os.path.join(RESULTS, d), ignore_errors=True)
                    print(f"  [anti-LIXO] {d[:46]} -> {clean_slug}/{target_rev[:8]}", flush=True)
            except Exception as _le:
                print(f"  [anti-LIXO] falhou: {str(_le)[:70]}", flush=True)
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
