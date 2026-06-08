"""BR-TaxQA-R — Brazilian tax-law information retrieval.

Reference: https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R
Original paper: Fernandes et al. 2025

BR-TaxQA-R packages:
- 478 legal documents (Brazilian tax law: Leis, Decretos, INs, etc.)
- 715 tax-law questions, each citing one or more legal documents
- Relevance judgments derived from `formatted_references` (cited docs)
  and `formatted_embedded_references` (additional docs cited within the
  answer text).

We treat the citation links as graded relevance:
- score=2 for documents in `formatted_references` (primary citations)
- score=1 for documents in `formatted_embedded_references` (secondary)

The retrieval task: given a tax question, retrieve the cited legal
documents. Tests legal-domain retrieval in PT-BR with a different
sub-domain from JurisTCU (Tribunal de Contas) — here we have tax-code
statutes rather than jurisprudence decisions.

License: CC-BY-4.0 (from upstream README + LICENSE file).
"""

from __future__ import annotations

import json
from typing import Any

from huggingface_hub import hf_hub_download
from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

_REPO = "unicamp-dl/BR-TaxQA-R"
_REVISION = "9f0f4263928c"
_QUESTIONS_FILE = "questions_QA_2024_v1.1.json"
_DOCS_FILE = "referred_legal_documents_QA_2024_v1.1.json"


class BRTaxQAR(AbsTaskRetrieval):
    """BR-TaxQA-R — Brazilian Portuguese tax-law retrieval, 715 queries / 478 docs."""

    metadata = TaskMetadata(
        name="BRTaxQAR",
        description=(
            "BR-TaxQA-R: Brazilian Portuguese tax-law information retrieval. 715 "
            "questions about Brazilian tax law paired with 478 legal documents "
            "(Leis, Decretos, Instruções Normativas, etc.). Relevance judgments "
            "are derived from the explicit legal citations in each answer: "
            "primary references (score 2) and embedded references mentioned "
            "within the answer text (score 1). Complements JurisTCU's "
            "jurisprudence retrieval with statutory tax-code retrieval."
        ),
        reference="https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R",
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
        date=("2024-01-01", "2024-12-31"),
        domains=["Legal", "Government", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""@article{br-taxqa-r-2025,
    title = {{BR-TaxQA-R}: A {B}razilian {P}ortuguese tax-law information retrieval dataset},
    author = {Fernandes, Leandro Car{\'i}sio and others},
    year = {2025},
    url = {https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R},
}""",
    )

    def load_data(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Populate self.corpus / self.queries / self.relevant_docs from BR-TaxQA-R."""
        if getattr(self, "data_loaded", False):
            return

        docs_path = hf_hub_download(_REPO, _DOCS_FILE, repo_type="dataset", revision=_REVISION)
        questions_path = hf_hub_download(
            _REPO, _QUESTIONS_FILE, repo_type="dataset", revision=_REVISION
        )

        # Corpus: 478 legal documents, keyed by filename (without .txt)
        with open(docs_path, encoding="utf-8") as fh:
            docs_raw: list[dict[str, Any]] = json.load(fh)
        corpus: dict[str, dict[str, str]] = {}
        for d in docs_raw:
            fname = d.get("filename", "").strip()
            text = d.get("filedata", "").strip()
            if not fname or not text:
                continue
            docid = fname[:-4] if fname.endswith(".txt") else fname
            corpus[docid] = {"text": text, "title": docid}

        # Queries + qrels from question file
        with open(questions_path, encoding="utf-8") as fh:
            questions_raw: list[dict[str, Any]] = json.load(fh)
        queries: dict[str, str] = {}
        relevant_docs: dict[str, dict[str, int]] = {}
        for q in questions_raw:
            qid = str(q.get("question_number", "")).strip()
            text = q.get("question_text", "").strip()
            if not qid or not text:
                continue
            queries[qid] = text
            qrels: dict[str, int] = {}
            # Primary references (score 2)
            for ref in q.get("formatted_references", []) or []:
                f = (ref.get("file") or "").strip()
                if not f:
                    continue
                docid = f[:-4] if f.endswith(".txt") else f
                if docid in corpus:
                    qrels[docid] = 2
            # Embedded references (score 1, don't downgrade existing 2s)
            for ref in q.get("formatted_embedded_references", []) or []:
                f = (ref.get("file") or "").strip()
                if not f:
                    continue
                docid = f[:-4] if f.endswith(".txt") else f
                if docid in corpus and docid not in qrels:
                    qrels[docid] = 1
            if qrels:  # only keep queries with at least one valid relevance link
                relevant_docs[qid] = qrels

        # Only keep queries that have at least one qrel resolvable in corpus
        queries = {q: t for q, t in queries.items() if q in relevant_docs}

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.data_loaded = True
