"""MTEB Portuguese (mteb-pt) — a public benchmark for evaluating text embedding
models on **native** Brazilian Portuguese (created or mined in PT-BR, no machine
translation).

Usage
-----

Register the MTEB(por, v2) tasks with the upstream ``mteb`` library and evaluate::

    import mteb_pt.register  # side-effect: registers the tasks with mteb
    import mteb

    model = mteb.get_model("intfloat/multilingual-e5-large-instruct")
    tasks = mteb.get_tasks(tasks=mteb_pt.HEADLINE_TASKS)
    results = mteb.evaluate(model, tasks=tasks)

Or run the whole suite (resumable; spot / block-volume friendly)::

    python scripts/run_mteb_por_v2.py intfloat/multilingual-e5-small

See ``examples/`` for paired-bootstrap significance testing and headline ranking.

Resources
---------

- Live leaderboard:  https://huggingface.co/spaces/mteb-pt/leaderboard
- Results dataset:   https://huggingface.co/datasets/mteb-pt/mteb-pt-results
- Organization:      https://huggingface.co/mteb-pt
- Source code:       https://github.com/tardellirs/mteb-pt
"""

from __future__ import annotations

__version__ = "2.0.0"

#: The 26 MTEB(por, v2) tasks — all native Brazilian Portuguese, no translation,
#: spanning 8 MTEB task-types.
HEADLINE_TASKS: list[str] = [
    # Classification (3)
    "HateBR",
    "ToxSynPT",
    "FactckBrClassification",
    # Multi-label classification (2)
    "BrighterEmotionMultilabelClassification",
    "OlidBrMultilabelClassification",
    # Regression (3)
    "EnemEssayRegression",
    "NarrativeEssaysBRRegression",
    "BrighterEmotionIntensityRegression",
    # Pair classification / NLI (2)
    "AssinRTE",
    "InferBR",
    # Semantic textual similarity (2)
    "AssinSTS",
    "Assin2STS",
    # Clustering (6)
    "WikipediaPTCategoriesClusteringP2P",
    "MedPTClustering",
    "JurisTCUClusteringP2P",
    "SciELOClusteringP2P",
    "CamaraProposicoesClustering",
    "StackoverflowPtClustering",
    # Retrieval (6)
    "Quati",
    "JurisTCU",
    "BRTaxQAR",
    "FaQuADIR",
    "MedPTRetrieval",
    "FaqBacenRetrieval",
    # Reranking (2)
    "QuatiReranking",
    "JurisTCUReranking",
]

#: Per-category groupings (the 8 MTEB task-types covered).
TASKS_BY_CATEGORY: dict[str, list[str]] = {
    "Classification": ["HateBR", "ToxSynPT", "FactckBrClassification"],
    "MultilabelClassification": [
        "BrighterEmotionMultilabelClassification",
        "OlidBrMultilabelClassification",
    ],
    "Regression": [
        "EnemEssayRegression",
        "NarrativeEssaysBRRegression",
        "BrighterEmotionIntensityRegression",
    ],
    "PairClassification": ["AssinRTE", "InferBR"],
    "STS": ["AssinSTS", "Assin2STS"],
    "Clustering": [
        "WikipediaPTCategoriesClusteringP2P",
        "MedPTClustering",
        "JurisTCUClusteringP2P",
        "SciELOClusteringP2P",
        "CamaraProposicoesClustering",
        "StackoverflowPtClustering",
    ],
    "Retrieval": [
        "Quati",
        "JurisTCU",
        "BRTaxQAR",
        "FaQuADIR",
        "MedPTRetrieval",
        "FaqBacenRetrieval",
    ],
    "Reranking": ["QuatiReranking", "JurisTCUReranking"],
}

#: License-pending / gated tasks — evaluated but kept OUT of the headline suite
#: until access + license clear (PortuLexRRIP: gated, license unconfirmed).
PENDING_TASKS: list[str] = ["PortuLexRRIP"]

__all__ = ["HEADLINE_TASKS", "PENDING_TASKS", "TASKS_BY_CATEGORY", "__version__"]
