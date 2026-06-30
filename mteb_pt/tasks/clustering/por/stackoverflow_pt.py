"""StackoverflowPtClustering — cluster Portuguese Stack Overflow questions by tag.

Question titles from pt.stackoverflow.com (fetched via the Stack Exchange API),
labelled by their technology tag (python, java, php, javascript, android, mysql,
c#, html, css, c). Native PT-BR technical Q&A — the Programming domain, absent
from the rest of the suite. Content is CC-BY-SA-4.0 (Stack Exchange). Built as
mteb-pt/stackoverflow-clustering.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClustering

_REPO = "mteb-pt/stackoverflow-clustering"
_REVISION = "baea103af775d126bedd7e51e350c6e193557d0e"


class StackoverflowPtClustering(AbsTaskClustering):
    """Cluster pt.stackoverflow question titles into 10 technology tags."""

    metadata = TaskMetadata(
        name="StackoverflowPtClustering",
        description=(
            "Cluster native Brazilian-Portuguese technical question titles from "
            "the Portuguese Stack Overflow (pt.stackoverflow.com) into 10 "
            "technology tags (python, java, php, javascript, android, mysql, c#, "
            "html, css, c). Programming domain."
        ),
        reference="https://pt.stackoverflow.com/",
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
        date=("2014-01-01", "2024-12-31"),
        domains=["Programming", "Web", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@misc{ptstackoverflow,
    title        = {Portuguese Stack Overflow (pt.stackoverflow.com)},
    author       = {{Stack Exchange Inc.}},
    howpublished = {Community Q\&A site; data via the Stack Exchange API, content licensed CC-BY-SA-4.0},
    url          = {https://pt.stackoverflow.com/},
}""",
    )

    input_column_name = "sentences"
    label_column_name = "label"
