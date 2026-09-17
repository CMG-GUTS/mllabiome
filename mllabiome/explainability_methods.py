from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SHAP:
    algorithm: str = "permutation"
    masker: str = "independent"
    background_size: int = 50
    max_explain: int = 50
    permutation_rounds: int = 5
    name: str = field(default="shap", init=False)

    def __post_init__(self) -> None:
        if self.algorithm not in {"auto", "tree", "permutation", "partition", "exact"}:
            raise ValueError(
                "SHAP.algorithm must be 'auto', 'tree', 'permutation', 'partition', or 'exact'."
            )
        if self.masker not in {"independent", "partition"}:
            raise ValueError("SHAP.masker must be 'independent' or 'partition'.")
        if self.algorithm == "partition" and self.masker != "partition":
            raise ValueError("SHAP algorithm='partition' requires masker='partition'.")
        if int(self.background_size) < 1:
            raise ValueError("SHAP.background_size must be at least 1.")
        if int(self.max_explain) < 1:
            raise ValueError("SHAP.max_explain must be at least 1.")
        if int(self.permutation_rounds) < 1:
            raise ValueError("SHAP.permutation_rounds must be at least 1.")


@dataclass(frozen=True)
class Permutation:
    n_repeats: int = 10
    scoring: str = "log_loss"
    max_samples: float | int = 1.0
    name: str = field(default="permutation", init=False)

    def __post_init__(self) -> None:
        if int(self.n_repeats) < 1:
            raise ValueError("Permutation.n_repeats must be at least 1.")
        if str(self.scoring).strip().lower() not in {"log_loss", "brier"}:
            raise ValueError("Permutation.scoring must be 'log_loss' or 'brier'.")
        if isinstance(self.max_samples, float) and not (
            0.0 < float(self.max_samples) <= 1.0
        ):
            raise ValueError("Permutation.max_samples as a float must be in (0, 1].")
        if isinstance(self.max_samples, int) and int(self.max_samples) < 1:
            raise ValueError(
                "Permutation.max_samples as an integer must be at least 1."
            )


@dataclass(frozen=True)
class ALE:
    bins: int | str = "auto"
    min_bins: int = 5
    max_bins: int = 15
    name: str = field(default="ale", init=False)

    def __post_init__(self) -> None:
        if isinstance(self.bins, str) and self.bins != "auto":
            raise ValueError("ALE.bins must be an integer or 'auto'.")
        if isinstance(self.bins, int) and int(self.bins) < 2:
            raise ValueError("ALE.bins must be at least 2.")
        if int(self.min_bins) < 2 or int(self.max_bins) < int(self.min_bins):
            raise ValueError("ALE bin limits are invalid.")


@dataclass(frozen=True)
class LIME:
    num_samples: int = 2000
    max_explain: int = 50
    feature_selection: str = "none"
    discretize_continuous: bool = False
    sampling_method: str = "gaussian"
    kernel_width: float | None = None
    name: str = field(default="lime", init=False)

    def __post_init__(self) -> None:
        if int(self.num_samples) < 100:
            raise ValueError("LIME.num_samples must be at least 100.")
        if int(self.max_explain) < 1:
            raise ValueError("LIME.max_explain must be at least 1.")
        if self.feature_selection not in {
            "none",
            "auto",
            "forward_selection",
            "lasso_path",
            "highest_weights",
        }:
            raise ValueError("Unsupported LIME.feature_selection.")
        if self.sampling_method not in {"gaussian", "lhs"}:
            raise ValueError("LIME.sampling_method must be 'gaussian' or 'lhs'.")
        if self.kernel_width is not None and float(self.kernel_width) <= 0.0:
            raise ValueError("LIME.kernel_width must be positive.")


@dataclass(frozen=True)
class ALEInteractions:
    bins: int | str = "auto"
    min_bins: int = 5
    max_bins: int = 8
    top_k: int = 20
    name: str = field(default="interactions", init=False)

    def __post_init__(self) -> None:
        if isinstance(self.bins, str) and self.bins != "auto":
            raise ValueError("ALEInteractions.bins must be an integer or 'auto'.")
        if isinstance(self.bins, int) and int(self.bins) < 2:
            raise ValueError("ALEInteractions.bins must be at least 2.")
        if int(self.min_bins) < 2 or int(self.max_bins) < int(self.min_bins):
            raise ValueError("ALEInteractions bin limits are invalid.")
        if int(self.top_k) < 2:
            raise ValueError("ALEInteractions.top_k must be at least 2.")


_METHOD_TYPES = (SHAP, Permutation, ALE, LIME, ALEInteractions)
_PROFILE_NAMES = {"screening", "standard", "comprehensive"}


def method_name(method: Any) -> str:
    if isinstance(method, str):
        aliases = {
            "ale_interactions": "interactions",
            "interaction": "interactions",
            "lime_tabular": "lime",
        }
        key = method.strip().lower().replace(" ", "_")
        return aliases.get(key, key)
    if isinstance(method, _METHOD_TYPES):
        return str(method.name)
    value = getattr(method, "name", "")
    return str(value).strip().lower().replace(" ", "_")


def default_method(method: str) -> Any:
    name = method_name(method)
    mapping = {
        "shap": SHAP,
        "permutation": Permutation,
        "ale": ALE,
        "lime": LIME,
        "interactions": ALEInteractions,
    }
    if name not in mapping:
        raise ValueError(name)
    return mapping[name]()


def screening_method(method: str) -> Any:
    name = method_name(method)
    mapping = {
        "shap": SHAP(
            algorithm="auto", background_size=50, max_explain=200, permutation_rounds=1
        ),
        "permutation": Permutation(n_repeats=1, max_samples=1.0),
        "ale": ALE(bins=8, min_bins=5, max_bins=8),
        "lime": LIME(num_samples=500, max_explain=200),
        "interactions": ALEInteractions(bins=8, min_bins=5, max_bins=8, top_k=50),
    }
    if name not in mapping:
        raise ValueError(name)
    return mapping[name]


def comprehensive_method(method: str) -> Any:
    name = method_name(method)
    mapping = {
        "shap": SHAP(background_size=100, max_explain=200, permutation_rounds=10),
        "permutation": Permutation(n_repeats=30),
        "ale": ALE(max_bins=20),
        "lime": LIME(num_samples=5000, max_explain=200),
        "interactions": ALEInteractions(max_bins=12, top_k=50),
    }
    if name not in mapping:
        raise ValueError(name)
    return mapping[name]


def normalise_profile(profile: str) -> str:
    value = str(profile).strip().lower().replace("-", "_")
    aliases = {
        "default": "standard",
        "quick_screening": "screening",
        "quick": "screening",
        "full": "comprehensive",
        "extended": "comprehensive",
    }
    value = aliases.get(value, value)
    if value not in _PROFILE_NAMES:
        raise ValueError(
            "Explainability.profile must be 'screening', 'standard', or 'comprehensive'."
        )
    return value


def coerce_method(method: Any) -> Any:
    if isinstance(method, _METHOD_TYPES):
        return method
    if isinstance(method, str):
        return default_method(method)
    raise TypeError(type(method).__name__)


def apply_profile(method: Any, profile: str) -> Any:
    obj = coerce_method(method)
    resolved = normalise_profile(profile)
    if resolved == "standard":
        return obj
    baseline = default_method(method_name(obj))
    if obj != baseline:
        return obj
    if resolved == "screening":
        return screening_method(method_name(obj))
    return comprehensive_method(method_name(obj))


def method_to_dict(method: Any) -> dict[str, Any]:
    obj = coerce_method(method)
    return {key: value for key, value in vars(obj).items()}
