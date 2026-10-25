from .fmriprep import FMRIPrepRun, discover_fmriprep_runs
from .graph_centrality import graph_centrality
from .labus_2019 import labus_2019_centrality
from .timeseries import BOLDPreprocessing, extract_schaefer400
from .handcrafted import (
    HandcraftedSettings,
    HandcraftedResult,
    export_handcrafted_fmriprep,
    pearson_fc,
    partial_fc,
    spectral_amplitudes,
)
from .structural import TissueMaps, discover_tissue_maps, tissue_volumes

__all__ = [
    "BOLDPreprocessing",
    "FMRIPrepRun",
    "discover_fmriprep_runs",
    "extract_schaefer400",
    "TissueMaps",
    "discover_tissue_maps",
    "tissue_volumes",
    "HandcraftedSettings",
    "HandcraftedResult",
    "export_handcrafted_fmriprep",
    "pearson_fc",
    "partial_fc",
    "spectral_amplitudes",
    "graph_centrality",
    "labus_2019_centrality",
    "BrainAttribution",
    "read_brain_attributions",
    "available_brain_regions",
    "render_brain_attributions",
    "BrainHarmonixFEncoder",
    "BrainHarmonixFResult",
    "BrainHarmonixSEncoder",
    "BrainHarmonixSFEncoder",
    "embed_fmriprep",
    "BrainSource",
    "with_brain_modality",
    "BrainFeatureTrace",
    "Schaefer400Provenance",
    "annotate_brain_features",
    "regional_embedding_sensitivity",
    "regional_prediction_sensitivity",
    "regional_interaction_sensitivity",
    "summarize_regional_sensitivity",
    "render_schaefer_scores",
    "BrainTraceResult",
    "trace_fmriprep_embedding_feature",
    "BrainTraceabilityConfig",
]


def __getattr__(name: str):
    if name in {
        "BrainAttribution",
        "read_brain_attributions",
        "available_brain_regions",
    }:
        from . import attributions

        return getattr(attributions, name)
    if name == "render_brain_attributions":
        from . import visualization

        return getattr(visualization, name)
    if name in {"BrainHarmonixFEncoder", "BrainHarmonixFResult", "embed_fmriprep"}:
        from . import brainharmonix

        return getattr(brainharmonix, name)
    if name in {"BrainHarmonixSEncoder", "BrainHarmonixSFEncoder"}:
        from . import brainharmonix_multimodal

        return getattr(brainharmonix_multimodal, name)
    if name in {"BrainSource", "with_brain_modality"}:
        from . import source

        return getattr(source, name)
    if name in {
        "BrainFeatureTrace",
        "Schaefer400Provenance",
        "annotate_brain_features",
        "regional_embedding_sensitivity",
        "regional_prediction_sensitivity",
        "regional_interaction_sensitivity",
        "summarize_regional_sensitivity",
    }:
        from . import traceability

        return getattr(traceability, name)
    if name in {
        "BrainTraceResult",
        "trace_fmriprep_embedding_feature",
        "BrainTraceabilityConfig",
    }:
        from . import explanation

        return getattr(explanation, name)
    if name == "render_schaefer_scores":
        from . import schaefer_figures

        return getattr(schaefer_figures, name)
    raise AttributeError(name)


from .representations import (
    Brain,
    BrainRepresentation,
    BrainRepresentationResult,
    prepare_brain_representations,
)

from .representation_explainability import (
    annotate_brain_representation_table,
    regional_feature_scores,
    render_brain_feature_importance,
)
