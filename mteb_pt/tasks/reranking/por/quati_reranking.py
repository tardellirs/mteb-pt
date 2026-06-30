"""QuatiReranking — genuine web-domain reranking with BM25 hard negatives.

Reference: https://huggingface.co/datasets/unicamp-dl/quati

Genuine reranking (not the qrels-union shortcut): for each of the 50 queries we
take the BM25 top-100 first-stage candidates over the FULL 1M-passage Quati
Brazilian web corpus, unioned with the human-judged passages (so relevants stay
rankable). The model reranks those ~100 candidates per query — which include
lexical hard negatives drawn from a realistic first stage — under Quati's graded
relevance scale 0-3. This isolates reranking ability from first-stage retrieval.

Candidates are precomputed (BM25 via bm25s over the 1M corpus, Portuguese
stemming) and pinned at mteb-pt/quati-reranking
(corpus / queries / qrels / top_ranked configs).

Domain pairing for the reranking story:
- QuatiReranking    — web / general PT-BR (this task)
- JurisTCUReranking — legal / TCU jurisprudence
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

_REPO = "mteb-pt/quati-reranking"
_REVISION = "68d40ca9a44e8ea0704fb628f31ace070c16bdbc"


class QuatiReranking(AbsTaskRetrieval):
    """Genuine Quati reranking — rerank BM25 top-100 candidates per query."""

    metadata = TaskMetadata(
        name="QuatiReranking",
        description=(
            "Genuine web-domain reranking: for each of 50 PT-BR queries, rerank "
            "the ~100 first-stage candidates (BM25 top-100 over the full "
            "1M-passage Quati Brazilian web corpus, unioned with human-judged "
            "passages) under graded relevance 0-3. Candidates include lexical "
            "hard negatives, so the task is distinct from first-stage retrieval. "
            "Complements JurisTCUReranking (legal) with a web-domain probe."
        ),
        reference="https://aclanthology.org/2024.stil-1.19/",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="map_at_1000",
        date=("2022-01-01", "2024-06-30"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        adapted_from=["Quati"],
        bibtex_citation=r"""@inproceedings{bueno-etal-2024-quati,
    title = "{Q}uati: A {B}razilian {P}ortuguese Information Retrieval Dataset from Native Speakers",
    author = "Bueno, Mirelle and Maia Sanches, Eduardo and Lotufo, Roberto and Nogueira, Rodrigo",
    booktitle = "Proceedings of the 16th Symposium in Information and Human Language Technology (STIL 2024)",
    year = "2024",
    url = "https://aclanthology.org/2024.stil-1.19/",
}""",
    )

    def load_data(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Load corpus/queries/qrels/top_ranked from the pinned reranking dataset."""
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
        top_ranked: dict[str, list[str]] = {}
        for r in load_dataset(_REPO, "top_ranked", split="test", revision=_REVISION):
            top_ranked[str(r["query-id"])] = [str(d) for d in r["corpus-ids"]]

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.top_ranked = {"test": top_ranked}
        self.data_loaded = True
