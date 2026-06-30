"""PortuLex RRIP — rhetorical-role classification of Brazilian legal sentences.

Dataset: https://huggingface.co/datasets/eduagarcia/PortuLex_benchmark (config `rrip`)

Each sentence of a Brazilian legal petition/decision is labelled with one of 8
rhetorical roles. Native PT-BR legal text (named entities anonymised). This is
the only PortuLex config usable as an embedding task — the other four are
token-level NER.

NOTE: `eduagarcia/PortuLex_benchmark` is GATED (manual approval) and declares no
license. This task is PENDING license/un-gating clearance (see the outreach to
E. Garcia) and is therefore kept OUT of the public headline suite until it
clears — it can be evaluated but should not ship in the published benchmark
while gated/unlicensed.
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

_REPO = "eduagarcia/PortuLex_benchmark"
_REVISION = "cddad88112fb6d0bb4f66cb58a362f08b85033e1"


class PortuLexRRIP(AbsTaskClassification):
    """PortuLex RRIP — 8-way rhetorical-role classification of PT-BR legal sentences."""

    metadata = TaskMetadata(
        name="PortuLexRRIP",
        description=(
            "Rhetorical Role Identification (RRIP) from the PortuLex legal benchmark: "
            "each sentence of a Brazilian legal document is classified into one of 8 "
            "rhetorical roles. Native PT-BR legal text with anonymised named entities."
        ),
        reference="https://huggingface.co/datasets/eduagarcia/PortuLex_benchmark",
        dataset={"path": _REPO, "name": "rrip", "revision": _REVISION},
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2023-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@misc{portulex_benchmark,
    title = {PortuLex Benchmark},
    author = {Garcia, Eduardo A. S.},
    howpublished = {\url{https://huggingface.co/datasets/eduagarcia/PortuLex_benchmark}},
}""",
    )

    def dataset_transform(self, **kwargs: Any) -> None:
        self.dataset = self.dataset.rename_column("sentence", "text").select_columns(
            ["text", "label"]
        )
