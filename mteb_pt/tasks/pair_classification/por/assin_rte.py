"""ASSIN v1 RTE (Recognizing Textual Entailment) — Brazilian Portuguese.

Reference: Fonseca, E., Santos, L., Criscuolo, M., & Aluísio, S. (2016).
"ASSIN: Avaliação de Similaridade Semântica e Inferência Textual."
PROPOR 2016. http://propor2016.di.fc.ul.pt/?page_id=381

Dataset: https://huggingface.co/datasets/nilc-nlp/assin

ASSIN ships three configs: `full` (PT-BR + PT-PT), `ptbr` (Brazilian only),
`ptpt` (European only). For MTEB-PT we use **`ptbr`** to keep the dialect
clean. Splits: train/validation/test = 2500/500/2000 (Brazilian subset).

Note: ASSIN v2 (`Assin2RTE`) is already in upstream mteb. ASSIN v1 is the
older sibling with different (smaller) sentence pool; we include it for
historical comparison + extra data point in the pair-classification suite.

Labels: `entailment_judgment` is one of {"None", "Entailment", "Paraphrase"}.
For binary RTE we map None→0 and {Entailment, Paraphrase}→1, following the
upstream Assin2RTE convention.
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskPairClassification

_ASSIN_REVISION = "6535e48351178e07ade013b05b69f0e35cb28bbb"


class AssinRTE(AbsTaskPairClassification):
    """ASSIN v1 RTE binarized: None=0, Entailment+Paraphrase=1. Brazilian subset."""

    metadata = TaskMetadata(
        name="AssinRTE",
        description=(
            "ASSIN (Avaliação de Similaridade Semântica e Inferência Textual) v1, "
            "Brazilian Portuguese subset. Recognizing Textual Entailment task: "
            "given a premise-hypothesis pair, predict whether the hypothesis "
            "is entailed by the premise. Labels binarized as {None: 0, "
            "Entailment+Paraphrase: 1} following Assin2RTE convention."
        ),
        reference="https://aclanthology.org/2020.lrec-1.319/",  # ASSIN follow-up survey
        dataset={
            "path": "nilc-nlp/assin",
            "revision": _ASSIN_REVISION,
            "name": "ptbr",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="max_ap",
        date=("2016-01-01", "2016-12-31"),
        domains=["News", "Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""@inproceedings{fonseca-etal-2016-assin,
    title = "{ASSIN}: {A}valia{\c{c}}{\~a}o de {S}imilaridade {S}em{\^a}ntica e {I}nfer{\^e}ncia {T}extual",
    author = "Fonseca, Erick  and
      Santos, Leandro  and
      Criscuolo, Marcelo  and
      Alu{\'\i}sio, Sandra",
    booktitle = "Proceedings of the 12th International Conference on Computational Processing of the Portuguese Language",
    year = "2016",
    url = "http://propor2016.di.fc.ul.pt/",
}""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Binarize entailment_judgment and reshape into mteb pair-classification format.

        mteb's PairClassification expects each split to be a Dataset with columns
        `sentence1`, `sentence2`, `labels`, where each row's value is a *list*
        spanning the whole split (batch-style).
        """
        from datasets import Dataset, DatasetDict

        def _binarize(lab: Any) -> int:
            if isinstance(lab, int):
                return 1 if lab > 0 else 0
            s = str(lab).lower()
            return 0 if s in {"none", "0"} else 1

        assert self.dataset is not None, "load_data() must run before dataset_transform"
        new: dict[str, Dataset] = {}
        for split_name, ds in self.dataset.items():
            premises = list(ds["premise"])
            hypotheses = list(ds["hypothesis"])
            labels = [_binarize(x) for x in ds["entailment_judgment"]]
            new[split_name] = Dataset.from_dict(
                {
                    "sentence1": [premises],
                    "sentence2": [hypotheses],
                    "labels": [labels],
                }
            )
        self.dataset = DatasetDict(new)
