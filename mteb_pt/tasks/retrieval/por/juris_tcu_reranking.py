"""JurisTCUReranking — legal-domain reranking restricted to the judged-doc pool.

Reference: https://huggingface.co/datasets/LeandroRibeiro/JurisTCU

Companion to QuatiReranking: instead of retrieving over the full 16k-doc
JurisTCU corpus, we restrict the corpus to the union of all qrel
documents (~2,250 graded judgments across 150 queries, capped at 15
candidates/query). The model still ranks the entire restricted corpus
per query, but each query has ~15 explicit human-labeled candidates —
the canonical mteb 2.12+ reranking pattern (AbsTaskRetrieval with a
small, qrels-restricted corpus).

Domain pairing for the cross-domain reranking story:
- QuatiReranking   — web / general PT-BR
- JurisTCUReranking — legal / TCU jurisprudence (this task)

License: parent dataset declares no SPDX identifier. Recorded as "not
specified"; authors emailed for clarification (same status as JurisTCU
retrieval task).
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download
from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

_JURISTCU_REPO = "LeandroRibeiro/JurisTCU"
_JURISTCU_REVISION = "ac7bea9e580626a586ffea69b245d26f5a73d44e"

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Strip HTML markup (<b>, <s>, ...) from JurisTCU ENUNCIADO fields."""
    if not isinstance(text, str):
        return ""
    return _HTML_TAG_RE.sub("", text).strip()


class JurisTCUReranking(AbsTaskRetrieval):
    """JurisTCU reranking — rank only the ~2,250 judged docs across 150 queries."""

    metadata = TaskMetadata(
        name="JurisTCUReranking",
        description=(
            "JurisTCU reranking task: rank the union of human-judged TCU "
            "jurisprudence documents (~2,250 docs, 15 candidates per query, "
            "150 queries) using graded relevance labels 0-3. Pure reranking "
            "setup — first-stage candidate generation is replaced by the "
            "official qrels pool. Complements QuatiReranking (web) with a "
            "legal-domain reranking probe."
        ),
        reference="https://huggingface.co/datasets/LeandroRibeiro/JurisTCU",
        dataset={
            "path": _JURISTCU_REPO,
            "revision": _JURISTCU_REVISION,
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2025-12-31"),
        domains=["Legal", "Government", "Written"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""@article{juristcu2026,
    author    = {Fernandes, Leandro Car{\'i}sio and
                 Ribeiro, Leandro dos Santos and
                 de Castro, Marcos Vin{\'i}cius Borela and
                 da Silva Pacheco, Leonardo Augusto and
                 de Oliveira Sandes, Edans Fl{\'a}vius},
    title     = {{JurisTCU: a Brazilian Portuguese information retrieval dataset with query relevance judgments}},
    year      = {2026},
    url       = {https://huggingface.co/datasets/LeandroRibeiro/JurisTCU},
}""",
    )

    def load_data(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Corpus restricted to docs referenced in any qrel (union across all queries)."""
        if getattr(self, "data_loaded", False):
            return

        doc_path = hf_hub_download(
            _JURISTCU_REPO, "doc.csv", repo_type="dataset", revision=_JURISTCU_REVISION
        )
        query_path = hf_hub_download(
            _JURISTCU_REPO, "query.csv", repo_type="dataset", revision=_JURISTCU_REVISION
        )
        qrel_path = hf_hub_download(
            _JURISTCU_REPO, "qrel.csv", repo_type="dataset", revision=_JURISTCU_REVISION
        )

        qrels_df = pd.read_csv(qrel_path)
        relevant_docs: dict[str, dict[str, int]] = {}
        judged_docids: set[str] = set()
        for _, row in qrels_df.iterrows():
            qid = str(row["QUERY_ID"])
            docid = str(row["DOC_ID"])
            score = int(row["SCORE"])
            relevant_docs.setdefault(qid, {})[docid] = score
            judged_docids.add(docid)

        queries_df = pd.read_csv(query_path)
        queries: dict[str, str] = {
            str(row["ID"]): str(row["TEXT"]).strip() for _, row in queries_df.iterrows()
        }

        docs_df = pd.read_csv(doc_path)
        corpus: dict[str, dict[str, str]] = {}
        for _, row in docs_df.iterrows():
            docid = str(row["KEY"])
            if docid not in judged_docids:
                continue
            text = _strip_html(row.get("ENUNCIADO", ""))
            if not text:
                continue
            corpus[docid] = {"text": text, "title": ""}

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.data_loaded = True
