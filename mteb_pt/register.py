"""Register MTEB-PT tasks with the upstream ``mteb`` global registry.

Importing this module side-effect-fully adds our 16 headline task classes into
``mteb.get_tasks._TASKS_REGISTRY`` so calls like ``mteb.get_task("HateBR")``
resolve correctly. This registration is required for any ``mteb`` internal that
resolves tasks by name string — including instruction-tuned models that look up
task-specific prompts from the registry.

Usage::

    import mteb_pt.register  # noqa: F401  -- side-effect import
    import mteb

    task = mteb.get_task("HateBR")
"""

from __future__ import annotations

from mteb.get_tasks import _TASKS_REGISTRY

# Classification
from mteb_pt.tasks.classification.por.hate_br import HateBR
from mteb_pt.tasks.classification.por.offcombr import OffComBR
from mteb_pt.tasks.classification.por.toxsyn_pt import ToxSynPT
from mteb_pt.tasks.classification.por.tweet_sent_br import TweetSentBR

# Pair classification (NLI)
from mteb_pt.tasks.pair_classification.por.assin_rte import AssinRTE
from mteb_pt.tasks.pair_classification.por.infer_br import InferBR

# Semantic textual similarity
from mteb_pt.tasks.sts.por.assin_sts import AssinSTS

# Clustering
from mteb_pt.tasks.clustering.por.medpt_clustering import MedPTClustering
from mteb_pt.tasks.clustering.por.wikipedia_pt_categories import (
    WikipediaPTCategoriesClusteringP2P,
)

# Retrieval
from mteb_pt.tasks.retrieval.por.br_taxqa_r import BRTaxQAR
from mteb_pt.tasks.retrieval.por.faquad_ir import FaQuADIR
from mteb_pt.tasks.retrieval.por.juris_tcu import JurisTCU
from mteb_pt.tasks.retrieval.por.medpt import MedPTRetrieval
from mteb_pt.tasks.retrieval.por.quati import Quati

# Reranking
from mteb_pt.tasks.retrieval.por.juris_tcu_reranking import JurisTCUReranking
from mteb_pt.tasks.retrieval.por.quati_reranking import QuatiReranking

_TASKS_TO_REGISTER = [
    HateBR, OffComBR, ToxSynPT, TweetSentBR,
    AssinRTE, InferBR,
    AssinSTS,
    MedPTClustering, WikipediaPTCategoriesClusteringP2P,
    Quati, JurisTCU, BRTaxQAR, FaQuADIR, MedPTRetrieval,
    QuatiReranking, JurisTCUReranking,
]

for _cls in _TASKS_TO_REGISTER:
    _name = _cls.metadata.name
    if _name not in _TASKS_REGISTRY:
        _TASKS_REGISTRY[_name] = _cls

__all__ = [_cls.__name__ for _cls in _TASKS_TO_REGISTER]
