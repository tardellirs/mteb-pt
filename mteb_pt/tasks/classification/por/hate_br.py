"""HateBR — Brazilian Instagram offensive-language classification (Vargas et al., LREC 2022).

Reference: https://aclanthology.org/2022.lrec-1.777/
Dataset:   https://huggingface.co/datasets/franciellevargas/HateBR

This is the **binary offensive-language detection** variant. HateBR also ships
multilabel offense-category annotations (antisemitism, fatphobia, etc.), but
those belong in a separate MultilabelClassification task — out of scope for v1.

Note: as of writing, the upstream `mteb` package ships HateSpeechPortugueseClassification
(Fortuna 2019 tweets), which is a different dataset. HateBR is distinct.
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

# Pinned to the SHA observed during 2026-05-23 preflight. Refresh via:
#   from huggingface_hub import HfApi
#   HfApi().dataset_info("franciellevargas/HateBR").sha
_HATEBR_REVISION = "077db456f6c4b376540ef07fbcfdec61e7806cc7"


class HateBR(AbsTaskClassification):
    """HateBR binary offensive-language classification."""

    metadata = TaskMetadata(
        name="HateBR",
        description=(
            "HateBR: Brazilian Portuguese Instagram comments annotated for "
            "offensive language and hate speech. Binary classification of "
            "offensive vs non-offensive comments. 7,000 comments, expert-"
            "annotated by linguists with kappa agreement >0.83."
        ),
        reference="https://aclanthology.org/2022.lrec-1.777/",
        dataset={
            "path": "franciellevargas/HateBR",
            "revision": _HATEBR_REVISION,
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2021-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@inproceedings{vargas-etal-2022-hatebr,
    title = "{H}ate{BR}: A Large Expert Annotated Corpus of {B}razilian {I}nstagram Comments for Offensive Language and Hate Speech Detection",
    author = "Vargas, Francielle  and
      Carvalho, Isabelle  and
      Rodrigues de G{\'o}es, Fabiana  and
      Pardo, Thiago  and
      Benevenuto, Fabr{\'\i}cio",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.777",
    pages = "7174--7183",
}""",
    )

    # Real HateBR CSV columns (verified 2026-05-23 preflight):
    # id, comentario, anotator1, anotator2, anotator3, label_final, links_post, account_post
    # `label_final` is the canonical binary label (0/1, balanced 3500/3500).
    input_column_name = "comentario"
    label_column_name = "label_final"
    samples_per_label = 32

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """HateBR ships only a 'train' split; build an 80/20 stratified test.

        `label_final` is a plain int Value, so we first cast it to ClassLabel
        (required by `train_test_split(stratify_by_column=...)`).
        """
        assert self.dataset is not None, "load_data() must run before dataset_transform"
        # `self.dataset` is a DatasetDict keyed by split.
        if "test" in self.dataset:  # already split upstream → nothing to do
            return
        if "train" not in self.dataset:
            raise RuntimeError(
                f"Unexpected HateBR splits {list(self.dataset.keys())}; expected 'train'."
            )
        train_with_label: Any = self.dataset["train"].class_encode_column(self.label_column_name)
        split: Any = train_with_label.train_test_split(
            test_size=0.2,
            seed=self.seed,
            stratify_by_column=self.label_column_name,
        )
        self.dataset = split
