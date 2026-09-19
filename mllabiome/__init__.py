import sys

from .threading import configure_thread_limits

configure_thread_limits()

__version__ = "0.1.0rc47"
from .configs_sweep import (
    MPDR,
    MPMA,
    Ensemble,
    Evaluation,
    Explainability,
    QualificationGate,
    Sweep,
    build_sweep_configs,
    build_sweep_from_module,
    evaluate,
)
from .data import Data, Dataset, load_dataset
from .integrations import Integration, IntegratedCoordinate, IntegrationModel
from .modalities import (
    Samples,
    Modality,
    ModalityDataset,
    ModalityMatrix,
    load_modalities,
)
from .ensemble_sweep import sweep_ensemble
from .explainability_methods import ALE, ALEInteractions, LIME, Permutation, SHAP
from .explainability import (
    ExplainabilityConfigurationError,
    ExplainabilityDependencyError,
)
from .final_explainability import explain
from .learners import FLAMLClassifier, build_learner
from .metrics import compute_metrics, compute_regression_metrics
from .pipeline import run_all
from .report import write_report
from .resolutions import materialize_mpdr
from .siamcat import SIAMCATClassifier
from .transformations import (
    TRANSFORMATION_LABELS,
    TRANSFORMATION_SPACE,
    CountTransformation,
    CountTransformationAdapter,
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
    "Data",
    "Samples",
    "Modality",
    "ModalityDataset",
    "ModalityMatrix",
    "Integration",
    "IntegratedCoordinate",
    "IntegrationModel",
    "load_modalities",
    "Dataset",
    "Ensemble",
    "Evaluation",
    "Explainability",
    "ExplainabilityConfigurationError",
    "ExplainabilityDependencyError",
    "SHAP",
    "Permutation",
    "ALE",
    "LIME",
    "ALEInteractions",
    "MPDR",
    "MPMA",
    "mll",
    "METRIC_COLUMNS",
    "QualificationGate",
    "SIAMCATClassifier",
    "Sweep",
    "TAXONOMIC_LEVELS",
    "Transform",
    "Transformation",
    "TransformationCoordinate",
    "TransformationLabel",
    "TRANSFORMATION_LABELS",
    "TRANSFORMATION_SPACE",
    "build_count_transformations",
    "FLAMLClassifier",
    "configure_thread_limits",
    "build_learner",
    "build_sweep_configs",
    "build_sweep_from_module",
    "compute_metrics",
    "compute_regression_metrics",
    "CLASSIFICATION_METRIC_COLUMNS",
    "REGRESSION_METRIC_COLUMNS",
    "evaluate",
    "explain",
    "load_dataset",
    "materialize_mpdr",
    "run_all",
    "write_report",
    "sweep_ensemble",
    "transformation_label",
    "transformation_space_table",
]


mll = sys.modules[__name__]
