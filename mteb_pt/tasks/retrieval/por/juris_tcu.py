"""JurisTCU — Brazilian Portuguese legal IR (TCU jurisprudence).

Reference: https://huggingface.co/datasets/LeandroRibeiro/JurisTCU
Citation:  @article{juristcu2026}

JurisTCU ships:
- 16,045 legal documents (TCU "Selected Jurisprudence" curation)
- 150 standardized queries across 3 sources (search log / LLM-derived / LLM-synthetic)
- 2,250 graded relevance judgments (scale 0-3, 15 per query)

Document text is taken from the ENUNCIADO field (formal jurisprudence
statement). HTML markup present in source (e.g., `<b>`, `<s>`) is stripped.

We use the full 150 queries as the test split (no train/dev split in the
upstream release).

License: the upstream dataset README does not specify a license; we
record license="not-specified" and have contacted the authors for
clarification.
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
    """Strip HTML tags from document text. JurisTCU ENUNCIADO contains <b>, <s>, etc."""
    if not isinstance(text, str):
        return ""
    return _HTML_TAG_RE.sub("", text).strip()


class JurisTCU(AbsTaskRetrieval):
    """JurisTCU — Brazilian Portuguese legal retrieval, 150 queries / 16k docs."""

    metadata = TaskMetadata(
        name="JurisTCU",
        description=(
            "JurisTCU: Brazilian Portuguese legal IR over Tribunal de Contas da União "
            "jurisprudence. 150 queries (mix of real search log + LLM-derived) with "
            "2,250 graded relevance judgments (0-3 scale) over 16,045 curated "
            "jurisprudence documents from the TCU 'Selected Jurisprudence' collection. "
            "Tests legal-domain retrieval in PT-BR, complementing Quati's general-web "
            "domain."
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
        """Populate self.corpus / self.queries / self.relevant_docs from JurisTCU CSVs."""
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

        docs_df = pd.read_csv(doc_path)
        corpus: dict[str, dict[str, str]] = {}
        for _, row in docs_df.iterrows():
            text = _strip_html(row.get("ENUNCIADO", ""))
            if not text:
                continue
            corpus[str(row["KEY"])] = {"text": text, "title": ""}

        queries_df = pd.read_csv(query_path)
        queries: dict[str, str] = {
            str(row["ID"]): str(row["TEXT"]).strip() for _, row in queries_df.iterrows()
        }

        qrels_df = pd.read_csv(qrel_path)
        relevant_docs: dict[str, dict[str, int]] = {}
        for _, row in qrels_df.iterrows():
            qid = str(row["QUERY_ID"])
            docid = str(row["DOC_ID"])
            score = int(row["SCORE"])
            relevant_docs.setdefault(qid, {})[docid] = score

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.data_loaded = True
