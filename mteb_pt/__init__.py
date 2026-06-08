"""MTEB Portuguese (mteb-pt) — a public benchmark for evaluating text embedding
models on Brazilian Portuguese.

Usage
-----

Auto-register all 16 headline tasks with the upstream ``mteb`` library and run
an evaluation on your model::

    import mteb_pt.register  # side-effect: registers our 16 tasks with mteb
    import mteb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    tasks = mteb.get_tasks(tasks=mteb_pt.HEADLINE_TASKS)
    results = mteb.MTEB(tasks=tasks).run(model, output_folder="./results")

See ``examples/`` in the GitHub repository for full evaluation, paired-bootstrap
significance testing, and headline ranking reproduction.

Resources
---------

- Live leaderboard:  https://huggingface.co/spaces/mteb-pt/leaderboard
- Results dataset:   https://huggingface.co/datasets/mteb-pt/mteb-pt-results
- Organization:      https://huggingface.co/mteb-pt
- Source code:       https://github.com/tardellirs/mteb-pt
"""

from __future__ import annotations

__version__ = "1.0.0"

#: The 16 headline MTEB-PT tasks reported in the paper's ``mean_16`` metric.
HEADLINE_TASKS: list[str] = [
    # Classification (4)
    "HateBR",
    "OffComBR",
    "TweetSentBR",
    "ToxSynPT",
    # Pair classification / NLI (2)
    "AssinRTE",
    "InferBR",
    # Semantic textual similarity (1)
    "AssinSTS",
    # Clustering (2)
    "MedPTClustering",
    "WikipediaPTCategoriesClusteringP2P",
    # Retrieval (5)
    "Quati",
    "JurisTCU",
    "BRTaxQAR",
    "FaQuADIR",
    "MedPTRetrieval",
    # Reranking (2)
    "QuatiReranking",
    "JurisTCUReranking",
]

#: Per-category groupings, matching the paper's Table 2.
TASKS_BY_CATEGORY: dict[str, list[str]] = {
    "Classification":      ["HateBR", "OffComBR", "TweetSentBR", "ToxSynPT"],
    "Pair classification": ["AssinRTE", "InferBR"],
    "STS":                 ["AssinSTS"],
    "Clustering":          ["MedPTClustering", "WikipediaPTCategoriesClusteringP2P"],
    "Retrieval":           ["Quati", "JurisTCU", "BRTaxQAR", "FaQuADIR", "MedPTRetrieval"],
    "Reranking":           ["QuatiReranking", "JurisTCUReranking"],
}

__all__ = ["HEADLINE_TASKS", "TASKS_BY_CATEGORY", "__version__"]
