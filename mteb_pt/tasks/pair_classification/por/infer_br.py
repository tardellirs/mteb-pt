"""InferBR — Brazilian Portuguese Natural Language Inference (Bencke et al., 2024).

Reference: https://huggingface.co/datasets/hapaxlegomenon/InferBR
Original paper: Bencke et al. (2024), LREC-COLING

InferBR is a 3-way NLI dataset (CONTRADICTION/ENTAILMENT/NEUTRAL) in
Brazilian Portuguese, with quality-flagged samples removed. Total 10,528
samples across train/val/test (8,190/633/1,705).

For PairClassification in MTEB we binarize: ENTAILMENT (label=1) vs
{CONTRADICTION, NEUTRAL} (label=0). This mirrors AssinRTE's binary
recipe and tests whether models detect strict entailment vs everything
else. Complements AssinRTE (which has only 500 dev/2000 test pairs)
with a larger and modern NLI sample pool.

License: MIT (declared in dataset README).
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskPairClassification

_REPO = "hapaxlegomenon/InferBR"
_REVISION = "304b2ca358c4f315a908b69d3d0c607ee101176d"


class InferBR(AbsTaskPairClassification):
    """InferBR — Brazilian Portuguese NLI, binarized as Entailment vs other."""

    metadata = TaskMetadata(
        name="InferBR",
        description=(
            "InferBR: Brazilian Portuguese Natural Language Inference. 10,528 "
            "premise-hypothesis pairs annotated as CONTRADICTION / ENTAILMENT / "
            "NEUTRAL, with quality-flagged samples removed from the original "
            "release. For PairClassification we binarize: ENTAILMENT (label=1) "
            "vs other (label=0). Complements AssinRTE with a larger, modern "
            "sample pool."
        ),
        reference="https://aclanthology.org/2024.lrec-main.788/",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="max_ap",
        date=("2023-01-01", "2024-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""@inproceedings{bencke-etal-2024-inferbr-natural,
    title = "{I}nfer{BR}: A Natural Language Inference Dataset in {P}ortuguese",
    author = "Bencke, Luciana  and
              Pereira, Francielle Vasconcellos  and
              Santos, Moniele Kunrath  and
              Moreira, Viviane",
    booktitle = "Proceedings of LREC-COLING 2024",
    year = "2024",
}""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Binarize 3-way label (Entailment=1 only) and reshape to mteb pair-class format."""
        from datasets import Dataset, DatasetDict

        assert self.dataset is not None, "load_data() must run before dataset_transform"

        new: dict[str, Dataset] = {}
        for split_name, ds in self.dataset.items():
            premises = list(ds["premise"])
            hypotheses = list(ds["hypothesis"])
            # ENTAILMENT label is 1 in upstream schema; binarize: 1 stays 1, 0/2 → 0.
            labels = [1 if int(x) == 1 else 0 for x in ds["label"]]
            new[split_name] = Dataset.from_dict(
                {
                    "sentence1": [premises],
                    "sentence2": [hypotheses],
                    "labels": [labels],
                }
            )
        self.dataset = DatasetDict(new)
