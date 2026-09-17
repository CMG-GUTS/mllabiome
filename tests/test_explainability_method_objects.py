from types import SimpleNamespace

import pytest

from mllabiome import ALE, ALEInteractions, Explainability, LIME, Permutation, SHAP
from mllabiome.explainability import (
    _auto_ale_bins,
    _explainability_config_signature,
    _normalise_explainability_methods,
    _resolve_explainability_classes,
)


def test_default_method_objects_are_explicit_and_reproducible():
    cfg = Explainability()
    assert _normalise_explainability_methods(cfg.methods) == (
        "shap",
        "permutation",
        "ale",
    )
    shap = next(x for x in cfg.methods if isinstance(x, SHAP))
    permutation = next(x for x in cfg.methods if isinstance(x, Permutation))
    ale = next(x for x in cfg.methods if isinstance(x, ALE))
    assert shap.algorithm == "permutation"
    assert shap.masker == "independent"
    assert cfg.profile == "standard"
    assert shap.background_size == 50
    assert shap.max_explain == 50
    assert shap.permutation_rounds == 5
    assert permutation.n_repeats == 10
    assert permutation.scoring == "log_loss"
    assert permutation.max_samples == 1.0
    assert ale.bins == "auto"
    lime = LIME()
    assert lime.num_samples == 2000
    assert lime.max_explain == 50
    assert ALEInteractions().top_k == 20


def test_legacy_method_strings_coerce_to_default_objects():
    cfg = Explainability(methods=("shap", "permutation", "ale", "lime", "interactions"))
    assert isinstance(cfg.methods[0], SHAP)
    assert isinstance(cfg.methods[1], Permutation)
    assert isinstance(cfg.methods[2], ALE)
    assert isinstance(cfg.methods[3], LIME)
    assert isinstance(cfg.methods[4], ALEInteractions)


def test_binary_auto_explains_positive_class_only():
    dataset = SimpleNamespace(class_labels=("control", "case"), positive_class=1)
    assert _resolve_explainability_classes(dataset, "auto") == (1,)


def test_multiclass_auto_explains_every_class():
    dataset = SimpleNamespace(class_labels=("A", "B", "C", "D"), positive_class=None)
    assert _resolve_explainability_classes(dataset, "auto") == (0, 1, 2, 3)


def test_explicit_class_labels_are_resolved_case_insensitively():
    dataset = SimpleNamespace(
        class_labels=("Underweight", "Normal", "Obese"), positive_class=None
    )
    assert _resolve_explainability_classes(dataset, ("obese", "Normal")) == (2, 1)


def test_explainability_signature_changes_with_method_parameters_and_classes():
    a = Explainability(methods=(SHAP(background_size=100),), classes="auto")
    b = Explainability(methods=(SHAP(background_size=50),), classes="auto")
    c = Explainability(methods=(SHAP(background_size=100),), classes="all")
    assert _explainability_config_signature(a) != _explainability_config_signature(b)
    assert _explainability_config_signature(a) != _explainability_config_signature(c)


def test_auto_ale_bins_are_bounded_and_sample_size_adaptive():
    spec = ALE(bins="auto", min_bins=5, max_bins=20)
    assert _auto_ale_bins(4, spec) == 5
    assert _auto_ale_bins(100, spec) == 10
    assert _auto_ale_bins(10000, spec) == 20
    assert _auto_ale_bins(100, ALE(bins=7)) == 7


def test_invalid_method_parameters_fail_early():
    with pytest.raises(ValueError):
        SHAP(background_size=0)
    with pytest.raises(ValueError):
        Permutation(n_repeats=0)
    with pytest.raises(ValueError):
        ALE(bins=1)
    with pytest.raises(ValueError):
        LIME(num_samples=20)
