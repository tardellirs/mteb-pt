"""FaQuAD-IR — Brazilian Portuguese academic retrieval, reformulated from FaQuAD SQuAD.

Reference: https://huggingface.co/datasets/eraldoluis/faquad
Original paper: Sayama et al., BRACIS 2019

FaQuAD is originally a SQuAD-style extractive QA dataset (Sayama et al.,
BRACIS 2019) covering Brazilian higher-education questions and answer
spans within source paragraphs. The corpus is built from 18 official
documents of a federal university CS program + 21 Wikipedia articles on
the Brazilian higher-education system.

We *reformulate* it as a passage retrieval task: given a question,
retrieve the paragraph that contains the answer. corpus = 244 unique
paragraphs (each indexed by (article_title, paragraph_index)); queries
= 900 questions; qrels: each question's source paragraph has relevance
score 1.

Complements the LEGAL retrieval suite (Quati web + JurisTCU
jurisprudence + BR-TaxQA-R tax law) with ACADEMIC/educational domain
retrieval, completing the multi-domain story.

Repackaged into MTEB retrieval format (corpus / queries / qrels configs) and
pinned at mteb-pt/faquad-ir for reproducibility (the original ships
as SQuAD JSON on GitHub, not Hub-resolvable; this is the same data reshaped).

License: CC-BY-4.0 (declared in the FaQuAD HF mirror README).
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

_REPO = "mteb-pt/faquad-ir"
_REVISION = "51fd9e7707bb4971229a0189560379992c3adce2"


class FaQuADIR(AbsTaskRetrieval):
    """FaQuAD-IR — Brazilian Portuguese academic retrieval, 900 queries / 244 paragraphs."""

    metadata = TaskMetadata(
        name="FaQuADIR",
        description=(
            "FaQuAD reformulated as PT-BR academic retrieval: given a question "
            "about Brazilian higher education, retrieve the source paragraph "
            "containing the answer. 900 questions over 244 unique paragraphs, "
            "drawn from 18 official documents of a federal-university CS "
            "program plus 21 Wikipedia articles about Brazil's higher-education "
            "system. Complements legal/web retrieval domains with academic/"
            "educational content."
        ),
        reference="https://github.com/liafacom/faquad",
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
        date=("2019-01-01", "2019-12-31"),
        domains=["Academic", "Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""@INPROCEEDINGS{sayama-etal-2019-faquad,
    author = {Sayama, H{\'e}lio Fonseca and Araujo, Anderson Vi{\c{c}}oso and Fernandes, Eraldo Rezende},
    booktitle = {2019 8th Brazilian Conference on Intelligent Systems (BRACIS)},
    title = {{F}a{Q}u{AD}: Reading Comprehension Dataset in the Domain of {B}razilian Higher Education},
    year = {2019},
    pages = {443--448},
    doi = {10.1109/BRACIS.2019.00084},
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
