"""SciELOClusteringP2P — cluster Brazilian scientific abstracts by research area.

Portuguese abstracts from the SciELO Brazil open-access library
(eduagarcia/scielo_abstracts), filtered to domain=scielo.br, lang=pt, and
license = pure CC-BY-4.0 (the per-row license field carries 21 variants; we
keep only Creative Commons Attribution 4.0, dropping NC/ND). The WoS
first_category (298 fine categories) is consolidated into 8 broad research
areas (Health Sciences, Social Sciences, Agricultural Sciences,
Biological/Life Sciences, Humanities & Arts, Engineering & Technology,
Physical Sciences & Chemistry, Mathematics & Computer Science). Capped to
500 abstracts/area for balance; built as mteb-pt/scielo-clustering.

Adds a formal-academic register orthogonal to the encyclopedic (Wikipedia),
legal (JurisTCU) and biomedical-QA (MedPT) clustering tasks.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClustering

_REPO = "mteb-pt/scielo-clustering"
_REVISION = "c84cafc4789b384b8cdd1d6bc2c22966e7e36f03"


class SciELOClusteringP2P(AbsTaskClustering):
    """Cluster Brazilian Portuguese scientific abstracts into 8 broad research areas."""

    metadata = TaskMetadata(
        name="SciELOClusteringP2P",
        description=(
            "Cluster Brazilian Portuguese scientific abstracts from the SciELO "
            "Brazil open-access library into 8 broad research areas (Health "
            "Sciences, Social Sciences, Agricultural Sciences, Biological/Life "
            "Sciences, Humanities & Arts, Engineering & Technology, Physical "
            "Sciences & Chemistry, Mathematics & Computer Science). Areas are "
            "consolidated from the Web-of-Science subject categories of each "
            "article; only pure CC-BY-4.0 articles are included."
        ),
        reference="https://huggingface.co/datasets/mteb-pt/scielo-clustering",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="v_measure",
        date=("2026-06-24", "2026-06-24"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=["brazilian"],
        sample_creation="found",
        adapted_from=None,
        bibtex_citation=r"""@misc{scielo_abstracts,
    title        = {{SciELO} Abstracts: a corpus of open-access Brazilian scientific abstracts},
    author       = {{SciELO}},
    howpublished = {HuggingFace dataset \texttt{eduagarcia/scielo\_abstracts}, derived from the SciELO open-access scientific library},
    url          = {https://huggingface.co/datasets/eduagarcia/scielo_abstracts},
}""",
    )

    input_column_name = "sentences"
    label_column_name = "label"
