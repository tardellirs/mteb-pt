"""Register MTEB-PT tasks with the upstream mteb global registry.

Importing this module side-effect-fully adds our task classes into
`mteb.get_tasks._TASKS_REGISTRY` so calls like `mteb.get_task("HateBR")`
resolve correctly. Required for:

  * instruct models (`intfloat/multilingual-e5-large-instruct`) that look up
    task-specific prompts via the global registry — without this they fail
    with `KeyError("HateBR not found. Did you mean Assin2RTE?")`
  * any other mteb internal that resolves tasks by name string

We keep this isolated from `mteb_pt.__init__` so the package can still be
imported in venvs without mteb (e.g. for the E5 OpenAI bootstrap script which
only uses `hf_io`).

After upstream PR merges, this whole module becomes a no-op — the tasks will
be in `_TASKS_REGISTRY` natively. Wrap the registration in a try/except so
we don't error if upstream already added them.
"""

from __future__ import annotations

from mteb.get_tasks import _TASKS_REGISTRY

from mteb_pt.tasks.classification.por.factckbr import FactckBrClassification
from mteb_pt.tasks.classification.por.hate_br import HateBR
from mteb_pt.tasks.classification.por.portulex_rrip import PortuLexRRIP
from mteb_pt.tasks.classification.por.toxsyn_pt import ToxSynPT
from mteb_pt.tasks.clustering.por.camara_proposicoes import CamaraProposicoesClustering
from mteb_pt.tasks.clustering.por.juris_tcu_clustering import JurisTCUClusteringP2P
from mteb_pt.tasks.clustering.por.medpt_clustering import MedPTClustering
from mteb_pt.tasks.clustering.por.scielo_clustering import SciELOClusteringP2P
from mteb_pt.tasks.clustering.por.stackoverflow_pt import StackoverflowPtClustering
from mteb_pt.tasks.clustering.por.wikipedia_pt_categories import (
    WikipediaPTCategoriesClusteringP2P,
)
from mteb_pt.tasks.multilabel_classification.por.brighter_emotion import (
    BrighterEmotionMultilabelClassification,
)
from mteb_pt.tasks.multilabel_classification.por.olid_br import OlidBrMultilabelClassification
from mteb_pt.tasks.pair_classification.por.assin_rte import AssinRTE
from mteb_pt.tasks.pair_classification.por.infer_br import InferBR
from mteb_pt.tasks.regression.por.brighter_intensity import BrighterEmotionIntensityRegression
from mteb_pt.tasks.regression.por.enem_essay import EnemEssayRegression
from mteb_pt.tasks.regression.por.narrative_essays import NarrativeEssaysBRRegression
from mteb_pt.tasks.reranking.por.juris_tcu_reranking import JurisTCUReranking
from mteb_pt.tasks.reranking.por.quati_reranking import QuatiReranking
from mteb_pt.tasks.retrieval.por.br_taxqa_r import BRTaxQAR
from mteb_pt.tasks.retrieval.por.faq_bacen import FaqBacenRetrieval
from mteb_pt.tasks.retrieval.por.faquad_ir import FaQuADIR
from mteb_pt.tasks.retrieval.por.juris_tcu import JurisTCU
from mteb_pt.tasks.retrieval.por.medpt import MedPTRetrieval
from mteb_pt.tasks.retrieval.por.quati import Quati
from mteb_pt.tasks.sts.por.assin_sts import AssinSTS

_TASKS_TO_REGISTER = [
    HateBR,
    FactckBrClassification,
    ToxSynPT,
    PortuLexRRIP,
    AssinRTE,
    InferBR,
    AssinSTS,
    EnemEssayRegression,
    NarrativeEssaysBRRegression,
    BrighterEmotionIntensityRegression,
    BrighterEmotionMultilabelClassification,
    OlidBrMultilabelClassification,
    Quati,
    QuatiReranking,
    JurisTCU,
    JurisTCUReranking,
    BRTaxQAR,
    FaQuADIR,
    FaqBacenRetrieval,
    MedPTRetrieval,
    WikipediaPTCategoriesClusteringP2P,
    MedPTClustering,
    JurisTCUClusteringP2P,
    SciELOClusteringP2P,
    CamaraProposicoesClustering,
    StackoverflowPtClustering,
]

for _cls in _TASKS_TO_REGISTER:
    name = _cls.metadata.name
    if name not in _TASKS_REGISTRY:
        _TASKS_REGISTRY[name] = _cls  # type: ignore[type-abstract]  # all are concrete subclasses
