"""ASSIN v1 STS (Semantic Textual Similarity) — Brazilian Portuguese.

Reference: Fonseca et al., PROPOR 2016 (see assin_rte.py).
Dataset: https://huggingface.co/datasets/nilc-nlp/assin

ASSIN v1 `ptbr` config: ~2500 train / 500 validation / 2000 test pairs with
human-annotated `relatedness_score` on a 1-5 Likert scale (5 = most similar).

Note: ASSIN v2 STS is already in upstream mteb as `Assin2STS`. We include
ASSIN v1 for historical coverage + a separate data point in the STS suite.
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskSTS

_ASSIN_REVISION = "6535e48351178e07ade013b05b69f0e35cb28bbb"


class AssinSTS(AbsTaskSTS):
    """ASSIN v1 STS, Brazilian subset, relatedness 1-5."""

    metadata = TaskMetadata(
        name="AssinSTS",
        description=(
            "ASSIN (Avaliação de Similaridade Semântica e Inferência Textual) v1, "
            "Brazilian Portuguese subset. Semantic textual similarity task: each "
            "pair of sentences is annotated with a relatedness score on a "
            "1 (no similarity) to 5 (paraphrase) Likert scale by human experts."
        ),
        reference="https://aclanthology.org/2020.lrec-1.319/",
        dataset={
            "path": "nilc-nlp/assin",
            "revision": _ASSIN_REVISION,
            "name": "ptbr",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="cosine_spearman",
        date=("2016-01-01", "2016-12-31"),
        domains=["News", "Web", "Written"],
        task_subtypes=[],
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
}""",
    )
    min_score = 1
    max_score = 5

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Rename ASSIN columns to mteb STS expected: sentence1, sentence2, score."""
        from datasets import Dataset, DatasetDict

        assert self.dataset is not None, "load_data() must run before dataset_transform"
        keep = {"sentence1", "sentence2", "score"}
        new: dict[str, Dataset] = {}
        for split_name, ds in self.dataset.items():
            ds = ds.rename_columns(
                {
                    "premise": "sentence1",
                    "hypothesis": "sentence2",
                    "relatedness_score": "score",
                }
            )
            drop = [c for c in ds.column_names if c not in keep]
            new[split_name] = ds.remove_columns(drop)
        self.dataset = DatasetDict(new)
