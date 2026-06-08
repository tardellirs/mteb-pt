"""TweetSentBR — Brazilian Portuguese Twitter 3-way sentiment classification.

Reference: https://huggingface.co/datasets/eduagarcia/tweetsentbr_fewshot
Original paper: Brum & Nunes (2018), LREC

The full TweetSentBR ships 15,000+ annotated tweets across 3 classes
(Positive / Negative / Neutral). The HF mirror we use is the *few-shot*
subset (eduagarcia/tweetsentbr_fewshot): the full 2,010-tweet test set
plus a tiny 75-tweet train sample (25/class).

For our classification setup we use the 2,010 test set, applying an
80/20 stratified split to create train/test for embedding probing.

Complements HateBR (Instagram offensive) and OffComBR (web offensive)
with a 3-class sentiment-analysis dimension. Tests whether embeddings
distinguish positive/negative/neutral on PT-BR social media.

License: not declared in HF mirror README. Recorded as "not specified"
— Brum & Nunes 2018 paper distributes the corpus for academic use;
need to confirm SPDX with authors.
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

_REPO = "eduagarcia/tweetsentbr_fewshot"
_REVISION = "ef67a39e16eaef20bda1e799d5822187c7913398"


class TweetSentBR(AbsTaskClassification):
    """TweetSentBR — Brazilian Portuguese tweet sentiment (3-class)."""

    metadata = TaskMetadata(
        name="TweetSentBR",
        description=(
            "TweetSentBR: 3-class sentiment classification of Brazilian "
            "Portuguese tweets (Positive / Negative / Neutral). 2,010 tweets "
            "from the full TweetSentBR test set; we apply an 80/20 stratified "
            "split for train/test."
        ),
        reference="http://www.lrec-conf.org/proceedings/lrec2018/summaries/389.html",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@inproceedings{brum-nunes-2018-tweetsentbr,
    title = {Building a Sentiment Corpus of Tweets in {B}razilian {P}ortuguese},
    author = {Brum, Henrico Bertini and Nunes, Maria das Gra{\c{c}}as Volpe},
    booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
    year = {2018},
}""",
    )

    input_column_name = "sentence"
    label_column_name = "label"
    samples_per_label = 32

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Use 2,010-tweet test set as full pool, then 80/20 stratified split."""
        assert self.dataset is not None, "load_data() must run before dataset_transform"
        if set(self.dataset.keys()) == {"train", "test"} and len(self.dataset["test"]) == 2010:
            # Use the 2,010 test set as the pool; ignore the 75-tweet fewshot train.
            pool = self.dataset["test"].class_encode_column(self.label_column_name)
            split: Any = pool.train_test_split(
                test_size=0.2,
                seed=self.seed,
                stratify_by_column=self.label_column_name,
            )
            self.dataset = split
