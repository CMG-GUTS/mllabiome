import sys

from .threading import configure_thread_limits

configure_thread_limits()

__version__ = "0.1.0"
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
from .ensemble_sweep import sweep_ensemble
from .explainability import (
    ExplainabilityConfigurationError,
    ExplainabilityDependencyError,
)
from .final_explainability import explain
from .learners import FLAMLClassifier, build_learner
from .metrics import compute_metrics
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
    TransformationLabel,
    build_count_transformations,
    transformation_label,
    transformation_space_table,
)
from .utils import METRIC_COLUMNS, TAXONOMIC_LEVELS

__all__ = [
    "Data",
    "Dataset",
    "Ensemble",
    "Evaluation",
    "Explainability",
    "ExplainabilityConfigurationError",
    "ExplainabilityDependencyError",
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
