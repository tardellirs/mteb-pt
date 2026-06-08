"""FaQuAD-IR — Brazilian Portuguese academic retrieval, reformulated from FaQuAD SQuAD.

Reference: https://huggingface.co/datasets/eraldoluis/faquad
Original paper: Sayama et al., BRACIS 2019

FaQuAD is originally a SQuAD-style extractive QA dataset (Sayama et al.,
BRACIS 2019) covering Brazilian higher-education questions and answer
spans within source paragraphs. The corpus is built from 18 official
documents of a federal university CS program + 21 Wikipedia articles on
the Brazilian higher-education system.

We *reformulate* it as a passage retrieval task: given a question,
retrieve the paragraph that contains the answer. corpus = 249 unique
paragraphs (each indexed by (article_title, paragraph_index)); queries
= 900 questions; qrels: each question's source paragraph has relevance
score 1.

Complements the LEGAL retrieval suite (Quati web + JurisTCU
jurisprudence + BR-TaxQA-R tax law) with ACADEMIC/educational domain
retrieval, completing the multi-domain story.

We fetch the SQuAD JSON files directly from the upstream GitHub raw
mirror (https://github.com/liafacom/faquad), since the HF mirror is
script-based and incompatible with datasets>=3.

License: CC-BY-4.0 (declared in HF mirror README).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

_GITHUB_RAW = "https://raw.githubusercontent.com/liafacom/faquad/master/data"
_SPLITS = ("train", "dev")


class FaQuADIR(AbsTaskRetrieval):
    """FaQuAD-IR — Brazilian Portuguese academic retrieval, 900 queries / 249 paragraphs."""

    metadata = TaskMetadata(
        name="FaQuADIR",
        description=(
            "FaQuAD reformulated as PT-BR academic retrieval: given a question "
            "about Brazilian higher education, retrieve the source paragraph "
            "containing the answer. 900 questions over 249 unique paragraphs, "
            "drawn from 18 official documents of a federal-university CS "
            "program plus 21 Wikipedia articles about Brazil's higher-education "
            "system. Complements legal/web retrieval domains with academic/"
            "educational content."
        ),
        reference="https://github.com/liafacom/faquad",
        dataset={
            "path": "eraldoluis/faquad",
            "revision": "205ba826a2282a4a5aa9bd3651e55ee4f2da1546",
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
        """Fetch SQuAD-format JSONs from upstream GitHub, reshape into IR triples."""
        if getattr(self, "data_loaded", False):
            return

        corpus: dict[str, dict[str, str]] = {}
        queries: dict[str, str] = {}
        relevant_docs: dict[str, dict[str, int]] = {}

        for split in _SPLITS:
            url = f"{_GITHUB_RAW}/{split}.json"
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = json.loads(resp.read())
            for article in payload.get("data", []):
                title = (article.get("title") or "").strip()
                for pi, para in enumerate(article.get("paragraphs", [])):
                    context = (para.get("context") or "").strip()
                    if not context:
                        continue
                    docid = f"{title}__p{pi:03d}"
                    if docid not in corpus:
                        corpus[docid] = {"text": context, "title": title}
                    for qa in para.get("qas", []):
                        qid = str(qa.get("id", "")).strip()
                        q = (qa.get("question") or "").strip()
                        if not qid or not q:
                            continue
                        queries[qid] = q
                        relevant_docs.setdefault(qid, {})[docid] = 1

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.data_loaded = True
