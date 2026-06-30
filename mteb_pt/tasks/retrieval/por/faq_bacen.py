"""FaqBacenRetrieval — Brazilian Central Bank FAQ retrieval (Financial domain).

Itau-Unibanco/FAQ_BACEN: native PT-BR question/answer pairs from the Banco
Central do Brasil (BACEN) public FAQ on financial and banking regulation.
Reformulated as retrieval: given a citizen question, retrieve the correct
regulatory answer from the pool of 1673 unique answers (373 test questions).
Fills the Financial domain (absent from the suite) and a consumer/regulatory
FAQ-retrieval gap. Repackaged + pinned at mteb-pt/faq-bacen.
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

_REPO = "mteb-pt/faq-bacen"
_REVISION = "076d89a68a8b8d2f14e3161631c416ffe29b8463"


class FaqBacenRetrieval(AbsTaskRetrieval):
    """Retrieve the correct BACEN regulatory answer for a citizen question."""

    metadata = TaskMetadata(
        name="FaqBacenRetrieval",
        description=(
            "Retrieve the correct answer to a citizen question about Brazilian "
            "financial/banking regulation, from the Banco Central do Brasil "
            "(BACEN) public FAQ. 373 test questions over a pool of 1673 unique "
            "regulatory answers. Native PT-BR; financial/government domain."
        ),
        reference="https://huggingface.co/datasets/Itau-Unibanco/FAQ_BACEN",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2024-12-31"),
        domains=["Financial", "Government", "Written"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@misc{faq_bacen,
    title        = {{FAQ-BACEN}: a question-answering dataset from the {B}razilian {C}entral {B}ank public FAQ},
    author       = {{Ita\'u Unibanco}},
    howpublished = {HuggingFace dataset \texttt{Itau-Unibanco/FAQ\_BACEN}, sourced from the Banco Central do Brasil FAQ},
    url          = {https://huggingface.co/datasets/Itau-Unibanco/FAQ_BACEN},
}""",
    )

    def load_data(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Load corpus/queries/qrels from the pinned MTEB-format HF dataset."""
        if getattr(self, "data_loaded", False):
            return
        from datasets import load_dataset

        corpus = {
            r["_id"]: {"text": r["text"], "title": r["title"]}
            for r in load_dataset(_REPO, "corpus", split="test", revision=_REVISION)
        }
        queries = {
            r["_id"]: r["text"]
            for r in load_dataset(_REPO, "queries", split="test", revision=_REVISION)
        }
        relevant_docs: dict[str, dict[str, int]] = {}
        for r in load_dataset(_REPO, "qrels", split="test", revision=_REVISION):
            relevant_docs.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.data_loaded = True
