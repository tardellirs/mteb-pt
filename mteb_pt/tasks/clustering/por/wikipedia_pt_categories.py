"""WikipediaPTCategoriesClusteringP2P — cluster PT Wikipedia article paragraphs by category.

Dataset: mteb-pt/wikipedia-categories
Built from MediaWiki API category traversal of 15 broad PT categories
(História, Geografia, Política, Esporte, Música, Cinema, Literatura, Religião,
Ciência, Tecnologia, Animais, Plantas, Medicina, Filosofia, Astronomia).

2,873 articles total, recursive category descent depth 2 from each root.
For each article: first paragraph from MediaWiki `extracts` API (≤500 chars,
≥80 chars minimum), labeled by the root category.

Phase 0 E4 experiment with heuristic regex on titles produced v_measure 0.72
with e5-large; this v1 version uses the real category graph and should
match or exceed that.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClustering

_REPO = "mteb-pt/wikipedia-categories"
_REVISION = "ff433edc23c7e2dd3304da3fb7d0edba6356001a"


class WikipediaPTCategoriesClusteringP2P(AbsTaskClustering):
    """Cluster PT Wikipedia article paragraphs into 15 broad subject categories."""

    metadata = TaskMetadata(
        name="WikipediaPTCategoriesClusteringP2P",
        description=(
            "Cluster the first paragraph of Brazilian Portuguese Wikipedia "
            "articles into 15 broad subject categories (História, Geografia, "
            "Política, Esporte, Música, Cinema, Literatura, Religião, Ciência, "
            "Tecnologia, Animais, Plantas, Medicina, Filosofia, Astronomia). "
            "Articles sampled via MediaWiki API category traversal (depth 2) "
            "of curated root categories; first paragraph from the `extracts` API."
        ),
        reference="https://huggingface.co/datasets/mteb-pt/wikipedia-categories",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["por-Latn"],
        main_score="v_measure",
        date=("2026-05-24", "2026-05-24"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@misc{mtebpt2026wikicat,
  title  = {Wikipedia-PT Categories: a clustering benchmark for Brazilian Portuguese},
  author = {Stekel, Tardelli Ronan Coelho},
  year   = {2026},
  howpublished = {HuggingFace Dataset},
  note   = {Available at \url{https://huggingface.co/datasets/mteb-pt/wikipedia-categories}},
}""",
    )

    input_column_name = "sentences"
    label_column_name = "label"

    # mteb new AbsTaskClustering takes the dataset row-level as-is:
    # one (sentences: str, label: str) per row. Our dataset is already in
    # that shape from the build script, so no dataset_transform needed.
