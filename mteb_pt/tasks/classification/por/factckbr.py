"""FactckBrClassification — Brazilian Portuguese fake-news / fact-check classification.

FACTCK.BR (github.com/jghm-f/FACTCK.BR): native PT-BR claims fact-checked by
Brazilian agencies (Lupa, Publica, Aos Fatos) via the ClaimReview schema.
Veracity labels are normalised to 3 classes: falso / impreciso / verdadeiro.

Quality notes: the input is the CLAIM text only (`claimReviewed`), NOT the
fact-check article title (which leaks the verdict). The dataset is heavily
imbalanced (~72% falso), reflecting the fact-checking process itself. News /
fact-check domain (absent from the suite). Repackaged + pinned at
mteb-pt/factckbr.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

_REPO = "mteb-pt/factckbr"
_REVISION = "a57f556588718d0af1a81ef89cf10e8719f8454a"


class FactckBrClassification(AbsTaskClassification):
    """Classify a Brazilian-Portuguese claim as falso / impreciso / verdadeiro."""

    metadata = TaskMetadata(
        name="FactckBrClassification",
        description=(
            "Classify the veracity of a native Brazilian-Portuguese claim "
            "fact-checked by Brazilian agencies (Lupa, Publica, Aos Fatos), into "
            "3 classes: falso, impreciso, verdadeiro. The input is the claim text "
            "only (not the fact-check title, which would leak the verdict). "
            "News / fact-checking domain; heavily imbalanced toward falso."
        ),
        reference="https://github.com/jghm-f/FACTCK.BR",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2019-12-31"),
        domains=["News", "Social", "Written"],
        task_subtypes=["Claim verification"],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@inproceedings{moreno2019factckbr,
    title     = {{FACTCK.BR}: a new dataset to study fake news},
    author    = {Moreno, Jo{\~a}o and Bressan, Giovanni},
    booktitle = {Proceedings of the 25th Brazillian Symposium on Multimedia and the Web (WebMedia)},
    year      = {2019},
    doi       = {10.1145/3323503.3361698},
    url       = {https://doi.org/10.1145/3323503.3361698},
}""",
    )
