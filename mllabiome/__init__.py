import sys

from .threading import configure_thread_limits

configure_thread_limits()

from ._version import __version__
from .configs_sweep import (
    MPDR,
    MPMA,
    Ensemble,
    Evaluation,
    Explainability,
    Explore,
    Inference,
    LocalExplanationMode,
    LocalExplanations,
    QualificationGate,
    Robustness,
    Sweep,
    SweepTask,
    build_sweep_configs,
    build_sweep_from_module,
    evaluate,
)
from .data import Data, Dataset, Metadata, load_dataset
from .causilo import CausiloClassifier, CausiloDependencyError
from .exaone import EXAONETabularClassifier, EXAONETabularDependencyError
from .ensemble_sweep import sweep_ensemble
from .explainability import (
    ExplainabilityConfigurationError,
    ExplainabilityDependencyError,
)
from .explainability_methods import ALE, LIME, SHAP, ALEInteractions, Permutation
from .final_explainability import explain
from .federated import Federated
from .foundation_models import (
    MBEMBED_REFERENCE,
    MGM2_REFERENCE,
    MGM_REFERENCE,
    WAYPOINT_REFERENCE,
    QWEN3_EMBEDDING_REFERENCE,
    QwenOllamaEmbedding,
    FoundationModelDependencyError,
    FoundationModelReference,
    MBEmbed,
    MGM2Embedding,
    MGMEmbedding,
    WaypointEmbedding,
    foundation_model_references,
)
from .integrations import IntegratedCoordinate, Integration, IntegrationModel
from .kumo import KumoTabularClassifier, KumoTabularDependencyError
from .kumo_regressor import KumoTabularRegressor
from .limix2 import LimiX2Classifier, LimiX2DependencyError
from .limix import LimiXClassifier
from .learners import FLAMLClassifier, build_learner, validate_model_specs
from .ollama_decision import OllamaDecisionClassifier
from .ridge_softmax import RidgeSoftmaxClassifier
from .rank_calibration import GroupedRankCalibratedClassifier
from .metrics import compute_metrics, compute_regression_metrics
from .modalities import (
    Modality,
    ModalityDataset,
    ModalityMatrix,
    Samples,
    load_modalities,
)
from .pipeline import run_all, run_explore, run_inference, run_robustness
from .report import write_report
from .resolutions import materialize_mpdr
from .siamcat import SIAMCATClassifier
from .tabdpt import TabDPTClassifier, TabDPTDependencyError
from .tabicl import TabICLClassifier, TabICLDependencyError
from .tabfm import TabFMClassifier, TabFMDependencyError
from .tabm import TabMClassifier, TabMDependencyError
from .realmlp import RealMLPClassifier, RealMLPDependencyError
from .tabpfn import TabPFNClassifier, TabPFNDependencyError
from .transformations import (
    TRANSFORMATION_LABELS,
    TRANSFORMATION_SPACE,
    CountTransformation,
    CountTransformationAdapter,
    PrevalenceFilter,
    ScopedTransformation,
    Transform,
    Transformation,
    TransformationCoordinate,
    TransformationLabel,
    build_count_transformations,
    transformation_label,
    transformation_space_table,
)
from .utils import (
    CLASSIFICATION_METRIC_COLUMNS,
    METRIC_COLUMNS,
    REGRESSION_METRIC_COLUMNS,
    TAXONOMIC_LEVELS,
)

__all__ = [
    "ALE",
    "CLASSIFICATION_METRIC_COLUMNS",
    "CausiloClassifier",
    "CausiloDependencyError",
    "EXAONETabularClassifier",
    "EXAONETabularDependencyError",
    "KumoTabularClassifier",
    "KumoTabularRegressor",
    "KumoTabularDependencyError",
    "LIME",
    "LimiXClassifier",
    "LimiX2Classifier",
    "LimiX2DependencyError",
    "METRIC_COLUMNS",
    "MPDR",
    "MPMA",
    "REGRESSION_METRIC_COLUMNS",
    "SHAP",
    "TAXONOMIC_LEVELS",
    "TRANSFORMATION_LABELS",
    "TRANSFORMATION_SPACE",
    "ALEInteractions",
    "Data",
    "Metadata",
    "Dataset",
    "Ensemble",
    "Evaluation",
    "Federated",
    "Explore",
    "Explainability",
    "ExplainabilityConfigurationError",
    "ExplainabilityDependencyError",
    "Inference",
    "FLAMLClassifier",
    "FoundationModelDependencyError",
    "FoundationModelReference",
    "MBEmbed",
    "MBEMBED_REFERENCE",
    "MGM2Embedding",
    "MGM2_REFERENCE",
    "MGMEmbedding",
    "MGM_REFERENCE",
    "IntegratedCoordinate",
    "Integration",
    "IntegrationModel",
    "LocalExplanationMode",
    "LocalExplanations",
    "Modality",
    "ModalityDataset",
    "ModalityMatrix",
    "Permutation",
    "PrevalenceFilter",
    "ScopedTransformation",
    "QualificationGate",
    "Robustness",
    "OllamaDecisionClassifier",
    "RidgeSoftmaxClassifier",
    "GroupedRankCalibratedClassifier",
    "SIAMCATClassifier",
    "TabFMClassifier",
    "TabFMDependencyError",
    "TabMClassifier",
    "TabMDependencyError",
    "TabICLClassifier",
    "TabICLDependencyError",
    "RealMLPClassifier",
    "RealMLPDependencyError",
    "TabDPTClassifier",
    "TabDPTDependencyError",
    "TabPFNClassifier",
    "TabPFNDependencyError",
    "Samples",
    "Sweep",
    "SweepTask",
    "Transform",
    "Transformation",
    "TransformationCoordinate",
    "TransformationLabel",
    "WAYPOINT_REFERENCE",
    "QWEN3_EMBEDDING_REFERENCE",
    "QwenOllamaEmbedding",
    "WaypointEmbedding",
    "build_count_transformations",
    "build_learner",
    "build_sweep_configs",
    "build_sweep_from_module",
    "compute_metrics",
    "compute_regression_metrics",
    "configure_thread_limits",
    "evaluate",
    "explain",
    "foundation_model_references",
    "load_dataset",
    "load_modalities",
    "materialize_mpdr",
    "mll",
    "run_all",
    "run_explore",
    "run_inference",
    "run_robustness",
    "sweep_ensemble",
    "transformation_label",
    "transformation_space_table",
    "validate_model_specs",
    "write_report",
]


mll = sys.modules[__name__]
