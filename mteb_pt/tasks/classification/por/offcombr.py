"""OffComBR — Brazilian Portuguese offensive-comment classification.

Reference: https://huggingface.co/datasets/fernandabufon/offcombr
Original paper: de Pelle & Moreira (2017), BRACIS proceedings

OffComBR-3 is a 1,033-comment corpus of Brazilian web/news comments
annotated for offensive language. Binary classification: offensive
(label=1) vs non-offensive (label=0). Class imbalance: ~80% non-
offensive (label=0 → 831 rows), ~20% offensive (label=1 → 202 rows).

Complements HateBR (Instagram domain) with web/news-comment domain
offensive-language detection. Tests generalization of PT-BR hate/
offensive classifiers across different platforms.

License: not declared in dataset README. We record "not specified" and
have queued an email to the dataset authors to confirm.
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

_OFFCOMBR_REPO = "fernandabufon/offcombr"
_OFFCOMBR_REVISION = "d89ef519d14a0e6c6bd28a013ef3e792a909faa0"


class OffComBR(AbsTaskClassification):
    """OffComBR — Brazilian Portuguese offensive-comment binary classification."""

    metadata = TaskMetadata(
        name="OffComBR",
        description=(
            "OffComBR: Brazilian Portuguese offensive-language detection over "
            "web/news comments. 1,033 comments, binary annotation (offensive vs "
            "non-offensive). Class-imbalanced (~80/20). Complements HateBR's "
            "Instagram domain with web/news-comment domain coverage."
        ),
        reference="https://huggingface.co/datasets/fernandabufon/offcombr",
        dataset={
            "path": _OFFCOMBR_REPO,
            "revision": _OFFCOMBR_REVISION,
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2017-01-01", "2017-12-31"),
        domains=["News", "Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@inproceedings{de-pelle-moreira-2017-offcombr,
    title = {Offensive Comments in the {B}razilian Web: a Dataset and Baseline Results},
    author = {de Pelle, Rog{\'e}rio P. and Moreira, Viviane P.},
    booktitle = {Brazilian Workshop on Social Network Analysis and Mining (BraSNAM)},
    year = {2017},
}""",
    )

    input_column_name = "text"
    label_column_name = "label"
    samples_per_label = 32

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """OffComBR ships only a 'train' split; build an 80/20 stratified test.

        Unlike HateBR (where label is a raw int), OffComBR already has a
        ClassLabel dtype upstream, so no class_encode_column is needed.
        """
        assert self.dataset is not None, "load_data() must run before dataset_transform"
        if "test" in self.dataset:
            return
        if "train" not in self.dataset:
            raise RuntimeError(
                f"Unexpected OffComBR splits {list(self.dataset.keys())}; expected 'train'."
            )
        split: Any = self.dataset["train"].train_test_split(
            test_size=0.2,
            seed=self.seed,
            stratify_by_column=self.label_column_name,
        )
        self.dataset = split
