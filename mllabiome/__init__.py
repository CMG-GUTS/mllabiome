import sys

from .threading import configure_thread_limits

configure_thread_limits()

from ._version import __version__
from .configs_sweep import (
    MPDR,
    MPMA,
    Ensemble,
    Evaluation,
    Explore,
    Explainability,
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
from .ensemble_sweep import sweep_ensemble
from .explainability import (
    ExplainabilityConfigurationError,
    ExplainabilityDependencyError,
)
from .explainability_methods import ALE, LIME, SHAP, ALEInteractions, Permutation
from .final_explainability import explain
from .integrations import IntegratedCoordinate, Integration, IntegrationModel
from .learners import FLAMLClassifier, build_learner, validate_model_specs
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
from .transformations import (
    TRANSFORMATION_LABELS,
    TRANSFORMATION_SPACE,
    CountTransformation,
    CountTransformationAdapter,
    PrevalenceFilter,
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
    "LIME",
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
    "Explore",
    "Explainability",
    "ExplainabilityConfigurationError",
    "ExplainabilityDependencyError",
    "Inference",
    "FLAMLClassifier",
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
    "QualificationGate",
    "Robustness",
    "SIAMCATClassifier",
    "Samples",
    "Sweep",
    "SweepTask",
    "Transform",
    "Transformation",
    "TransformationCoordinate",
    "TransformationLabel",
    "build_count_transformations",
    "build_learner",
    "build_sweep_configs",
    "build_sweep_from_module",
    "compute_metrics",
    "compute_regression_metrics",
    "configure_thread_limits",
    "evaluate",
    "explain",
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
