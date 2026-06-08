<div align="center">

# MTEB-PT

**Massive Text Embedding Benchmark for Brazilian Portuguese**

[![Leaderboard](https://img.shields.io/badge/🤗-Leaderboard-yellow)](https://huggingface.co/spaces/mteb-pt/leaderboard)
[![Results](https://img.shields.io/badge/🤗-Dataset-blue)](https://huggingface.co/datasets/mteb-pt/mteb-pt-results)
[![Org](https://img.shields.io/badge/🤗-mteb--pt-green)](https://huggingface.co/mteb-pt)
[![License: Apache 2.0](https://img.shields.io/badge/License_(code)-Apache_2.0-blue)](LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/License_(data)-CC--BY--4.0-orange)](https://creativecommons.org/licenses/by/4.0/)

</div>

A public benchmark for evaluating text embedding models on **Brazilian Portuguese**, built as a thin extension on top of the [`mteb`](https://github.com/embeddings-benchmark/mteb) library.

**Live leaderboard**: <https://huggingface.co/spaces/mteb-pt/leaderboard>

## What it is

- **54 models** evaluated · open-weights + commercial APIs
- **16 tasks** from native PT-BR sources (no machine-translated benchmarks)
- Headline metric **mean_16** = average across all 16 tasks
- Per-task scores, per-query parquets, and reproduction scripts all public

## Quickstart

Install the package directly from GitHub:

```bash
pip install git+https://github.com/tardellirs/mteb-pt.git
```

Evaluate any sentence-transformers model on a single MTEB-PT task:

```python
import mteb_pt.register   # registers our 16 tasks with the global mteb registry
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
task  = mteb.get_task("HateBR")
mteb.MTEB(tasks=[task]).run(model, output_folder="./results")
```

Or run the **full 16-task headline evaluation** in one command:

```bash
python examples/run_headline.py \
    --model intfloat/multilingual-e5-large-instruct \
    --output ./results
```

This reproduces a row of the public leaderboard. Total runtime is approximately 30–60 minutes on a single A10G GPU depending on model size.

Compute the **paired-bootstrap p-value** between two model evaluations (reproduces paper Finding F1):

```bash
python examples/compute_bootstrap_ci.py \
    --results-a ./results/intfloat__multilingual-e5-large-instruct/mean_16.json \
    --results-b ./results/Qwen__Qwen3-Embedding-8B/mean_16.json
```

## Package layout

```
mteb_pt/
├── __init__.py                       # HEADLINE_TASKS list + TASKS_BY_CATEGORY map
├── register.py                       # side-effect: registers all 16 tasks with mteb
├── stats.py                          # bootstrap CIs + paired significance helpers
└── tasks/
    ├── classification/por/           # HateBR, OffComBR, ToxSynPT, TweetSentBR
    ├── pair_classification/por/      # AssinRTE, InferBR
    ├── sts/por/                      # AssinSTS
    ├── clustering/por/               # MedPTClustering, WikipediaPTCategoriesClusteringP2P
    └── retrieval/por/                # Quati, JurisTCU, BRTaxQAR, FaQuADIR, MedPTRetrieval
                                      # + reranking variants for Quati and JurisTCU

examples/
├── quickstart.py                     # 1 model × 1 task smoke test (~5 min CPU)
├── run_headline.py                   # full 16-task evaluation on one model
└── compute_bootstrap_ci.py           # paired-bootstrap p-value between two models

tests/
└── test_register.py                  # smoke tests: all 16 tasks resolve via mteb.get_task
```

## Task suite (16 headline tasks)

Each task wrapper uses upstream data pinned to a specific revision. Click a source for the
original dataset / paper.

| Task | Category | Source |
|---|---|---|
| [HateBR](https://aclanthology.org/2022.lrec-1.777/) | Classification | Vargas et al. 2022 (LREC) — [HF dataset](https://huggingface.co/datasets/franciellevargas/HateBR) |
| [OffComBR](https://huggingface.co/datasets/fernandabufon/offcombr) | Classification | Pelle & Moreira 2017 — HF mirror |
| [TweetSentBR](http://www.lrec-conf.org/proceedings/lrec2018/summaries/389.html) | Classification | Brum & Nunes 2018 (LREC) |
| [ToxSynPT](https://huggingface.co/datasets/AKCIT/ToxSyn-PT) | Classification | AKCIT |
| [AssinRTE](https://aclanthology.org/2020.lrec-1.319/) | Pair-classification (NLI) | Real et al. 2020 (LREC) — [HF dataset](https://huggingface.co/datasets/nilc-nlp/assin) |
| [InferBR](https://aclanthology.org/2024.lrec-main.788/) | Pair-classification (NLI) | Rodrigues et al. 2024 (LREC) |
| [AssinSTS](https://aclanthology.org/2020.lrec-1.319/) | STS | Real et al. 2020 (LREC) — [HF dataset](https://huggingface.co/datasets/nilc-nlp/assin) |
| [MedPTClustering](https://huggingface.co/datasets/AKCIT/MedPT) | Clustering | AKCIT |
| [WikipediaPTCategoriesClusteringP2P](https://huggingface.co/datasets/tardellirs/mteb-pt-wikipedia-categories) | Clustering | Wikipedia-derived (this benchmark) |
| [Quati](https://aclanthology.org/2024.stil-1.19/) | Retrieval (Pool) | Bueno et al. 2024 (STIL) — [HF dataset](https://huggingface.co/datasets/unicamp-dl/quati) |
| [JurisTCU](https://huggingface.co/datasets/LeandroRibeiro/JurisTCU) | Retrieval | Ribeiro et al. — TCU rulings |
| [BRTaxQAR](https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R) | Retrieval | UNICAMP-DL |
| [FaQuADIR](https://github.com/liafacom/faquad) | Retrieval | Sayama et al. 2019 — [HF mirror](https://huggingface.co/datasets/eraldoluis/faquad) |
| [MedPTRetrieval](https://huggingface.co/datasets/AKCIT/MedPT) | Retrieval | AKCIT |
| [QuatiReranking](https://aclanthology.org/2024.stil-1.19/) | Reranking | Bueno et al. 2024 — [HF dataset](https://huggingface.co/datasets/unicamp-dl/quati) |
| [JurisTCUReranking](https://huggingface.co/datasets/LeandroRibeiro/JurisTCU) | Reranking | Ribeiro et al. — TCU rulings |

Two additional tasks are evaluated but **excluded from the headline `mean_16`**: CSTNewsClustering (degenerate — all models score 1.000) and BBCNewsPTClustering (BBC copyright concern). They remain in the raw results for transparency.

If you cite a specific task in your work, please cite the **original task source** alongside this benchmark.

## Submit a new model

Two channels, pick whichever fits:

- **HF Discussion** on the leaderboard Space → [open a thread here](https://huggingface.co/spaces/mteb-pt/leaderboard/discussions/new) and attach the eval JSONs
- **GitHub Issue** → use the [model submission template](https://github.com/tardellirs/mteb-pt/issues/new?template=submit-model.yml)

Required for a submission:
1. `model_id` (HF repo path or vendor product name)
2. Per-task result JSONs for the 16 headline tasks
3. Reproducible evaluation command (e.g. the `mteb` library invocation or a script that downloads + evaluates the model)

We re-run a sample of submissions to verify before merging into the leaderboard. Closed-API models accepted (we'll verify with the vendor's official endpoint).

## Propose a new task

A task is a candidate for inclusion if it:
- Sources its data from native PT-BR (not machine-translated)
- Has clear licensing
- Discriminates across embedding models (i.e. not degenerate)

Open an [issue using the task proposal template](https://github.com/tardellirs/mteb-pt/issues/new?template=propose-task.yml) describing the dataset, license, size, and discrimination evidence.

## Maintainer

**Tardelli Stekel** — IFSP, São Paulo, Brazil
Email: <stekel@ifsp.edu.br>

Contributions, corrections, and discussion all welcome via Issues or HF Discussions.

## Citation

```bibtex
@misc{mteb-portuguese-2026,
  title  = {MTEB Portuguese: A Massive Text Embedding Benchmark for Brazilian Portuguese},
  author = {Stekel, Tardelli},
  year   = {2026},
  url    = {https://huggingface.co/spaces/mteb-pt/leaderboard}
}
```

If you used a specific task novel to this benchmark, please also cite the original task dataset.

## License

- Benchmark code: Apache-2.0
- Results dataset: CC-BY-4.0
- Individual task datasets: see each dataset's original license (linked in `docs/datasheet_novel_tasks.md`)
- Models evaluated: see each model card

## Acknowledgments

Built on top of the [`mteb`](https://github.com/embeddings-benchmark/mteb) library (Muennighoff et al., 2023). The multilingual sub-benchmark methodology follows MMTEB (Enevoldsen et al., 2025). Task datasets contributed by their original authors — see the [Task suite](#task-suite-16-headline-tasks) table above for sources and citations.
