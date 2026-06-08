"""MedPTRetrieval — Brazilian PT biomedical question-answer retrieval.

Reference: https://huggingface.co/datasets/AKCIT/MedPT
Original paper: AKCIT MedPT team (2024)

MedPT ships 384,095 PT-BR question/answer pairs spanning 3,288 medical
conditions and 1,278 specialties. We subsample to a benchmark-friendly
size: 500 (question, answer) pairs balanced across the 7 question_type
categories (Tratamento, Diagnóstico, Epidemiologia, etc.).

For retrieval: corpus = the 500 sampled answers, queries = the 500
sampled questions, qrels = 1:1 (each question's gold answer is the
single relevant document). This gives a 500-query × 500-doc retrieval
task — comparable in size to JurisTCU (150 q × 16k docs) and FaQuAD-IR
(900 q × 244 docs), but in a new domain (biomedical) covered by no
other PT-BR benchmark we know of.

License: CC-BY-4.0 (declared in HF dataset card).
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download
from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

_REPO = "AKCIT/MedPT"
_REVISION = "6a55a33e289539ad676da89dd20208729acc7509"
_DATA_FILE = "data/train-00000-of-00001.parquet"
_SAMPLE_N = 500
_SEED = 42


class MedPTRetrieval(AbsTaskRetrieval):
    """MedPT — Brazilian Portuguese biomedical QA retrieval, 500 queries × 500 docs."""

    metadata = TaskMetadata(
        name="MedPTRetrieval",
        description=(
            "MedPT: Brazilian Portuguese biomedical question-answer retrieval. "
            "Subsample of 500 question/answer pairs from the AKCIT/MedPT corpus "
            "(384k QA pairs across 3,288 medical conditions and 1,278 "
            "specialties), stratified across 7 question_type categories. Each "
            "question has one canonical answer as its single relevant document. "
            "Tests biomedical-domain retrieval transfer for PT-BR embedding models."
        ),
        reference="https://huggingface.co/datasets/AKCIT/MedPT",
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
        domains=["Medical", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""@misc{akcit-medpt-2024,
    title = {{M}ed{PT}: Brazilian Portuguese Medical Question Answering Dataset},
    author = {{AKCIT}},
    year = {2024},
    howpublished = {\url{https://huggingface.co/datasets/AKCIT/MedPT}},
}""",
    )

    def load_data(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Stratified subsample 500 pairs, build retrieval triples."""
        if getattr(self, "data_loaded", False):
            return

        local = hf_hub_download(_REPO, _DATA_FILE, repo_type="dataset", revision=_REVISION)
        df = pd.read_parquet(local, columns=["id", "question", "answer", "question_type"])

        # Stratified sample across the 7 question_type categories.
        n_per_type = _SAMPLE_N // df["question_type"].nunique()
        sampled = df.groupby("question_type", group_keys=False).apply(
            lambda g: g.sample(n=min(n_per_type, len(g)), random_state=_SEED)
        )
        # Top up to exactly _SAMPLE_N rows from a separate sample if rounding lost a few.
        if len(sampled) < _SAMPLE_N:
            extras = df.drop(sampled.index).sample(n=_SAMPLE_N - len(sampled), random_state=_SEED)
            sampled = pd.concat([sampled, extras])
        sampled = sampled.head(_SAMPLE_N).reset_index(drop=True)

        # Build corpus + queries + qrels.
        # Use stable hash-based ids to avoid collision with upstream id format.
        corpus: dict[str, dict[str, str]] = {}
        queries: dict[str, str] = {}
        relevant_docs: dict[str, dict[str, int]] = {}
        for _, row in sampled.iterrows():
            q_text = str(row["question"]).strip()
            a_text = str(row["answer"]).strip()
            if not q_text or not a_text:
                continue
            row_hash = hashlib.sha1(f"{q_text}|||{a_text}".encode()).hexdigest()[:12]
            qid = f"q_{row_hash}"
            docid = f"a_{row_hash}"
            queries[qid] = q_text
            corpus[docid] = {"text": a_text, "title": ""}
            relevant_docs[qid] = {docid: 1}

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.data_loaded = True
