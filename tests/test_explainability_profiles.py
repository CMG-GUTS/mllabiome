from mllabiome import ALE, ALEInteractions, Explainability, LIME, Permutation, SHAP
from mllabiome.explainability import _explainability_config_signature


def _by_type(cfg, cls):
    return next(x for x in cfg.methods if isinstance(x, cls))


def test_standard_profile_is_default_and_uses_efficient_reproducible_settings():
    cfg = Explainability(
        methods=(SHAP(), Permutation(), ALE(), LIME(), ALEInteractions())
    )
    assert cfg.profile == "standard"
    shap = _by_type(cfg, SHAP)
    permutation = _by_type(cfg, Permutation)
    ale = _by_type(cfg, ALE)
    lime = _by_type(cfg, LIME)
    interactions = _by_type(cfg, ALEInteractions)
    assert (shap.background_size, shap.max_explain, shap.permutation_rounds) == (
        50,
        50,
        5,
    )
    assert permutation.n_repeats == 10
    assert ale.max_bins == 15
    assert (lime.num_samples, lime.max_explain) == (2000, 50)
    assert (interactions.max_bins, interactions.top_k) == (8, 20)


def test_screening_reproduces_legacy_pre_profile_budgets():
    cfg = Explainability(
        profile="screening",
        methods=(SHAP(), Permutation(), ALE(), LIME(), ALEInteractions()),
    )
    shap = _by_type(cfg, SHAP)
    permutation = _by_type(cfg, Permutation)
    ale = _by_type(cfg, ALE)
    lime = _by_type(cfg, LIME)
    interactions = _by_type(cfg, ALEInteractions)
    assert cfg.profile == "screening"
    assert (shap.background_size, shap.max_explain, shap.permutation_rounds) == (
        50,
        200,
        1,
    )
    assert (permutation.n_repeats, permutation.max_samples) == (1, 1.0)
    assert (ale.bins, ale.max_bins) == (8, 8)
    assert (lime.num_samples, lime.max_explain) == (500, 200)
    assert (interactions.bins, interactions.max_bins, interactions.top_k) == (8, 8, 50)


def test_screening_is_canonical_and_legacy_quick_aliases_resolve_identically():
    a = Explainability(profile="screening", methods=(SHAP(), LIME()))
    b = Explainability(profile="quick_screening", methods=(SHAP(), LIME()))
    c = Explainability(profile="quick", methods=(SHAP(), LIME()))
    assert a.profile == "screening"
    assert b.profile == "screening"
    assert c.profile == "screening"
    assert a.methods == b.methods == c.methods


def test_comprehensive_profile_expands_unmodified_method_defaults():
    cfg = Explainability(
        profile="comprehensive",
        methods=(SHAP(), Permutation(), ALE(), LIME(), ALEInteractions()),
    )
    shap = _by_type(cfg, SHAP)
    permutation = _by_type(cfg, Permutation)
    ale = _by_type(cfg, ALE)
    lime = _by_type(cfg, LIME)
    interactions = _by_type(cfg, ALEInteractions)
    assert cfg.profile == "comprehensive"
    assert (shap.background_size, shap.max_explain, shap.permutation_rounds) == (
        100,
        200,
        10,
    )
    assert permutation.n_repeats == 30
    assert ale.max_bins == 20
    assert (lime.num_samples, lime.max_explain) == (5000, 200)
    assert (interactions.max_bins, interactions.top_k) == (12, 50)


def test_profile_applies_to_legacy_method_names():
    cfg = Explainability(profile="screening", methods=("shap", "permutation", "ale"))
    assert _by_type(cfg, SHAP).max_explain == 200
    assert _by_type(cfg, SHAP).permutation_rounds == 1
    assert _by_type(cfg, Permutation).n_repeats == 1
    assert _by_type(cfg, ALE).bins == 8


def test_explicit_method_customisation_is_preserved_under_profiles():
    cfg = Explainability(
        profile="screening",
        methods=(SHAP(background_size=25, max_explain=40, permutation_rounds=2),),
    )
    shap = _by_type(cfg, SHAP)
    assert (shap.background_size, shap.max_explain, shap.permutation_rounds) == (
        25,
        40,
        2,
    )


def test_profile_changes_cache_signature():
    screening = Explainability(profile="screening", methods=(SHAP(),))
    standard = Explainability(profile="standard", methods=(SHAP(),))
    comprehensive = Explainability(profile="comprehensive", methods=(SHAP(),))
    assert _explainability_config_signature(
        screening
    ) != _explainability_config_signature(standard)
    assert _explainability_config_signature(
        standard
    ) != _explainability_config_signature(comprehensive)


def test_invalid_profile_fails_early():
    try:
        Explainability(profile="turbo")
    except ValueError as exc:
        assert "screening" in str(exc)
        assert "standard" in str(exc)
        assert "comprehensive" in str(exc)
    else:
        raise AssertionError("invalid profile should fail")
