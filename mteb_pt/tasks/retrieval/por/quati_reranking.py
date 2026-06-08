"""QuatiReranking — Brazilian PT retrieval restricted to the judged-passage pool.

Reference: https://huggingface.co/datasets/unicamp-dl/quati

In mteb 2.12+ the legacy AbsTaskReranking format was deprecated in favor
of AbsTaskRetrieval with a smaller, qrels-restricted corpus. We follow
that pattern here: instead of retrieving over the full 1M-passage Quati
corpus, we restrict the corpus to the union of all qrel documents
(~1,933 judged passages across 50 queries). The model still ranks the
entire restricted corpus per query, but each query only has 30-40
"candidates" with explicit human relevance labels — this is the
canonical reranking setup.

Complements `Quati` (full retrieval) by isolating the model's reranking
ability from its first-stage candidate generation ability. Same source
data, different task framing.
"""

from __future__ import annotations

from typing import Any

from huggingface_hub import hf_hub_download
from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

_QUATI_REPO = "unicamp-dl/quati"
_QUATI_REVISION = "e5279055bba3e7ba1bece5c0eddd0ba232df49c3"
_PASSAGES_FILE = "quati_1M.tsv"
_TOPICS_FILE = "topics/quati_test_topics.tsv"
_QRELS_FILE = "qrels/quati_1M_qrels.txt"


class QuatiReranking(AbsTaskRetrieval):
    """Quati reranking — rank only the ~1,933 judged passages per query."""

    metadata = TaskMetadata(
        name="QuatiReranking",
        description=(
            "Quati reranking task: rank the ~1,933 human-judged passages "
            "(union of qrels across 50 PT-BR queries from the Quati Brazilian "
            "web retrieval corpus). Tests pure reranking ability — first-stage "
            "candidate generation is replaced by the official qrels pool. "
            "Graded relevance scale 0-3 from Quati's human annotators."
        ),
        reference="https://aclanthology.org/2024.stil-1.19/",
        dataset={
            "path": _QUATI_REPO,
            "revision": _QUATI_REVISION,
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2024-06-30"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""@inproceedings{bueno-etal-2024-quati,
    title = "{Q}uati: A {B}razilian {P}ortuguese Information Retrieval Dataset from Native Speakers",
    author = "Bueno, Mirelle and Maia Sanches, Eduardo and Lotufo, Roberto and Nogueira, Rodrigo",
    booktitle = "Proceedings of the 16th Symposium in Information and Human Language Technology (STIL 2024)",
    year = "2024",
    url = "https://aclanthology.org/2024.stil-1.19/",
}""",
    )

    def load_data(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Build corpus = union of all qrel docids; queries + qrels from Quati TSVs."""
        if getattr(self, "data_loaded", False):
            return

        passages_path = hf_hub_download(
            _QUATI_REPO, _PASSAGES_FILE, repo_type="dataset", revision=_QUATI_REVISION
        )
        topics_path = hf_hub_download(
            _QUATI_REPO, _TOPICS_FILE, repo_type="dataset", revision=_QUATI_REVISION
        )
        qrels_path = hf_hub_download(
            _QUATI_REPO, _QRELS_FILE, repo_type="dataset", revision=_QUATI_REVISION
        )

        # Load qrels first — these define the restricted corpus.
        relevant_docs: dict[str, dict[str, int]] = {}
        judged_docids: set[str] = set()
        with open(qrels_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                    relevant_docs.setdefault(qid, {})[docid] = rel
                    judged_docids.add(docid)

        # Load queries.
        queries: dict[str, str] = {}
        with open(topics_path, encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    queries[parts[0]] = parts[1]

        # Corpus = only judged passages (filter the 1M file).
        corpus: dict[str, dict[str, str]] = {}
        with open(passages_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2 and parts[0] in judged_docids:
                    corpus[parts[0]] = {"text": parts[1], "title": ""}
                    if len(corpus) == len(judged_docids):
                        break

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.data_loaded = True
