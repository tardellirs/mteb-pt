"""JurisTCUReranking — genuine legal-domain reranking with BM25 hard negatives.

Reference: https://huggingface.co/datasets/LeandroRibeiro/JurisTCU

Genuine reranking (not the qrels-union shortcut): for each of the 150 queries
we take the BM25 top-100 first-stage candidates over the FULL 16k-document
JurisTCU corpus, unioned with the human-judged documents (so relevants stay
rankable). The model reranks those ~100 candidates per query — which include
lexical hard negatives — under graded relevance 0-3. This isolates reranking
ability from first-stage retrieval: unlike a qrels-restricted corpus, the
candidate pool contains plausible distractors a model must order correctly.

Candidates are precomputed (BM25 via bm25s, Portuguese stemming) and pinned at
mteb-pt/juristcu-reranking (corpus / queries / qrels / top_ranked).

Domain pairing for the reranking story:
- QuatiReranking    — web / general PT-BR
- JurisTCUReranking — legal / TCU jurisprudence (this task)

License: CC-BY-4.0 (Springer LREv 2026, doi:10.1007/s10579-025-09881-w).
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

_REPO = "mteb-pt/juristcu-reranking"
_REVISION = "83d1eec1aac2ba4e639d72c32a34b4efe70aef82"


class JurisTCUReranking(AbsTaskRetrieval):
    """Genuine JurisTCU reranking — rerank BM25 top-100 candidates per query."""

    metadata = TaskMetadata(
        name="JurisTCUReranking",
        description=(
            "Genuine legal-domain reranking: for each of 150 queries, rerank the "
            "~100 first-stage candidates (BM25 top-100 over the full 16k-document "
            "TCU jurisprudence corpus, unioned with human-judged docs) under "
            "graded relevance 0-3. Candidates include lexical hard negatives, so "
            "the task is distinct from first-stage retrieval. Complements "
            "QuatiReranking (web) with a legal-domain reranking probe."
        ),
        reference="https://huggingface.co/datasets/LeandroRibeiro/JurisTCU",
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
        date=("2020-01-01", "2025-12-31"),
        domains=["Legal", "Government", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        adapted_from=["JurisTCU"],
        bibtex_citation=r"""@article{juristcu2026,
    author    = {Fernandes, Leandro Car{\'i}sio and
                 Ribeiro, Leandro dos Santos and
                 de Castro, Marcos Vin{\'i}cius Borela and
                 da Silva Pacheco, Leonardo Augusto and
                 de Oliveira Sandes, Edans Fl{\'a}vius},
    title     = {{JurisTCU: a Brazilian Portuguese information retrieval dataset with query relevance judgments}},
    journal   = {Language Resources and Evaluation},
    year      = {2026},
    volume    = {60},
    number    = {1},
    doi       = {10.1007/s10579-025-09881-w},
    url       = {https://doi.org/10.1007/s10579-025-09881-w},
    issn      = {1574-0218},
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
