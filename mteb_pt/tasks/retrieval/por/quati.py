"""Quati — native Brazilian Portuguese web retrieval (Bueno et al., STIL 2024).

Reference: https://aclanthology.org/2024.stil-1.19/
Dataset:   https://huggingface.co/datasets/unicamp-dl/quati

Quati ships 50 native PT-BR test topics + ~1933 human relevance judgements
over a 1M Brazilian web passage pool from ClueWeb22-PT. It's the first
retrieval benchmark designed natively for PT-BR (no translation).

Because `unicamp-dl/quati` is a script-based dataset incompatible with
`datasets>=3`, we override `load_data` and read the raw TSV files via
`hf_hub_download`.
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


class Quati(AbsTaskRetrieval):
    """Quati 1M — Brazilian Portuguese native web retrieval, 50 test topics."""

    metadata = TaskMetadata(
        name="Quati",
        description=(
            "Quati: native Brazilian Portuguese web retrieval benchmark. 50 test "
            "topics with 1,933 human relevance judgements over a 1M Brazilian "
            "web passage pool (ClueWeb22-PT). First PT-BR retrieval benchmark "
            "designed without translation; queries are naturally-occurring "
            "Brazilian web questions."
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
    author = "Bueno, Mirelle  and
      Maia Sanches, Eduardo  and
      Lotufo, Roberto  and
      Nogueira, Rodrigo",
    booktitle = "Proceedings of the 16th Symposium in Information and Human Language Technology (STIL 2024)",
    year = "2024",
    url = "https://aclanthology.org/2024.stil-1.19/",
}""",
    )

    def load_data(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Populate self.corpus / self.queries / self.relevant_docs from Quati TSVs.

        We override AbsTaskRetrieval.load_data with our own loader because
        unicamp-dl/quati is script-based (incompatible with datasets>=3).
        """
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

        corpus: dict[str, dict[str, str]] = {}
        with open(passages_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    corpus[parts[0]] = {"text": parts[1], "title": ""}

        queries: dict[str, str] = {}
        with open(topics_path, encoding="utf-8") as f:
            f.readline()  # skip header
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    queries[parts[0]] = parts[1]

        relevant_docs: dict[str, dict[str, int]] = {}
        with open(qrels_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                    relevant_docs.setdefault(qid, {})[docid] = rel

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.data_loaded = True
