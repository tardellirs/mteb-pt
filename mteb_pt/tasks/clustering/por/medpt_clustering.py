"""MedPTClustering — cluster Brazilian PT medical questions by specialty.

Reference: https://huggingface.co/datasets/AKCIT/MedPT

Replaces CSTNewsClustering's degenerate behavior (every model scores
v_measure=1.0 due to small/redundant clusters) with a discriminative
medical-domain clustering task: sample ~600 medical questions stratified
across 12 broad medical specialties and ask the embedding to cluster
them by specialty.

Why this is more discriminative than CSTNews:
- 12 distinct medical specialties with varied vocabulary
- ~50 questions per cluster (more than 2-3)
- Domain has rich technical vocabulary that should reward better PT
  understanding (similar to legal/tax discrimination patterns)

Why 12 specialties (not 1,278 in raw MedPT):
- mteb clustering scales kmeans by gold k; 1,278 is impractical
- We pick the 12 largest specialties to ensure enough docs per cluster

License: CC-BY-4.0 (from MedPT dataset card).
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClustering

_REPO = "AKCIT/MedPT"
_REVISION = "6a55a33e289539ad676da89dd20208729acc7509"
_DATA_FILE = "data/train-00000-of-00001.parquet"
_TOP_K_SPECIALTIES = 12
_PER_SPECIALTY = 50
_SEED = 42


class MedPTClustering(AbsTaskClustering):
    """MedPT medical-specialty clustering. 12 specialties × ~50 questions."""

    metadata = TaskMetadata(
        name="MedPTClustering",
        description=(
            "Cluster Brazilian Portuguese medical questions by specialty. Uses "
            "the 12 largest specialties from AKCIT/MedPT (e.g., Cardiologista, "
            "Ortopedista, Ginecologista, etc.), 50 questions sampled per "
            "specialty (~600 total). Replaces CSTNewsClustering which degenerated "
            "to v_measure=1.0 for every model. Tests embedding discrimination "
            "across PT-BR medical vocabulary."
        ),
        reference="https://huggingface.co/datasets/AKCIT/MedPT",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="v_measure",
        date=("2024-01-01", "2024-12-31"),
        domains=["Medical", "Written"],
        task_subtypes=["Topic classification"],
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

    input_column_name = "sentences"
    label_column_name = "label"

    def load_data(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Pick top 12 specialties, sample 50 questions each."""
        if getattr(self, "data_loaded", False):
            return

        local = hf_hub_download(_REPO, _DATA_FILE, repo_type="dataset", revision=_REVISION)
        df = pd.read_parquet(local, columns=["question", "medical_specialty"])
        df = df.dropna(subset=["question", "medical_specialty"]).reset_index(drop=True)

        # Pick the 12 specialties with most questions
        top_specs = df["medical_specialty"].value_counts().head(_TOP_K_SPECIALTIES).index.tolist()
        df = df[df["medical_specialty"].isin(top_specs)]

        # Stratified sample: 50 questions per specialty (explicit loop for stability).
        rows: list[dict[str, str]] = []
        for spec, group in df.groupby("medical_specialty"):
            sampled = group.sample(n=min(_PER_SPECIALTY, len(group)), random_state=_SEED)
            for q in sampled["question"]:
                q = str(q).strip()
                if q:
                    _ = hashlib.sha1(q.encode()).hexdigest()[:8]
                    rows.append({"sentences": q, "label": str(spec)})

        ds = Dataset.from_list(rows)
        self.dataset = DatasetDict({"test": ds})
        self.data_loaded = True
