from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import types
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .transformations import TransformationCoordinate


_WAYPOINT_RUNTIME_CACHE: dict[tuple[str, str], tuple[Any, Any, Any, Any]] = {}
_MGM_RUNTIME_CACHE: dict[tuple[str, str, str, str], tuple[Any, Any]] = {}
_MBEMBED_RUNTIME_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any, str]] = {}
_CACHE_FORMAT = "mllabiome-foundation-row-cache-v1"


@dataclass(frozen=True)
class FoundationModelReference:
    key: str
    title: str
    authors: str
    year: int
    venue: str
    doi: str
    software_url: str
    package: str


WAYPOINT_REFERENCE = FoundationModelReference(
    key="waypoint",
    title="Learning the Language of the Microbiome with Transformers",
    authors="Neythen J Treloar, Saif Ur-Rehman, Jenny Yang",
    year=2026,
    venue="bioRxiv",
    doi="10.64898/2026.05.02.722381",
    software_url="https://github.com/Outpost-Bio/waypoint",
    package="waypoint-bio",
)

MGM_REFERENCE = FoundationModelReference(
    key="mgm",
    title="MGM as a Large-Scale Pretrained Foundation Model for Microbiome Analyses in Diverse Contexts",
    authors="Haohong Zhang, Yuli Zhang, Zixin Kang, Jiayun Xiong, Ronghua Yang, Kang Ning",
    year=2026,
    venue="Advanced Science",
    doi="10.1002/advs.202513333",
    software_url="https://github.com/HUST-NingKang-Lab/MGM",
    package="microformer-mgm",
)

MBEMBED_REFERENCE = FoundationModelReference(
    key="mbembed",
    title="Self-Supervised Representation Learning for Microbiome Improves Downstream Prediction in Data-Limited Settings and Cross-Cohort Generalizability",
    authors="Liron Zahavi, Zachary Levine, Eran Segal",
    year=2025,
    venue="ICML Workshop on Generative AI for Biology",
    doi="",
    software_url="https://github.com/LironZa/MBEmbed",
    package="MBEmbed",
)

MGM2_REFERENCE = FoundationModelReference(
    key="mgm2",
    title="MGM2 as a Unified Foundation Model for Microbiome World Exploration",
    authors="Haohong Zhang, Yuli Zhang, Yuxuan Qi, Tianao Liu, Ronghua Yang, Kang Ning",
    year=2026,
    venue="bioRxiv",
    doi="10.64898/2026.07.20.739063",
    software_url="https://github.com/HUST-NingKang-Lab/MGM2",
    package="LudensZhang/MGM2",
)


class FoundationModelDependencyError(ImportError):
    pass


def foundation_model_references() -> tuple[FoundationModelReference, ...]:
    return (WAYPOINT_REFERENCE, MGM_REFERENCE, MBEMBED_REFERENCE, MGM2_REFERENCE)


def _mgm_package_root() -> Path:
    spec = importlib.util.find_spec("mgm")
    if spec is None:
        raise FoundationModelDependencyError(
            "MGMEmbedding requires the mllabiome foundation extra. Run `uv sync --extra foundation`."
        )
    locations = spec.submodule_search_locations
    if locations:
        root = Path(next(iter(locations))).resolve()
    elif spec.origin:
        root = Path(spec.origin).resolve().parent
    else:
        raise FoundationModelDependencyError(
            "Could not locate the installed MGM package."
        )
    return root


def _ensure_mgm_pkg_resources_compat() -> None:
    try:
        import pkg_resources

        return
    except ModuleNotFoundError:
        pass
    module = types.ModuleType("pkg_resources")

    def resource_filename(package: str, resource: str) -> str:
        if package != "mgm":
            spec = importlib.util.find_spec(package)
            if spec is None:
                raise ModuleNotFoundError(package)
            locations = spec.submodule_search_locations
            root = (
                Path(next(iter(locations))).resolve()
                if locations
                else Path(spec.origin).resolve().parent
            )
        else:
            loaded = sys.modules.get("mgm")
            if loaded is not None and getattr(loaded, "__file__", None):
                root = Path(loaded.__file__).resolve().parent
            else:
                root = _mgm_package_root()
        return str((root / resource).resolve())

    def resource_exists(package: str, resource: str) -> bool:
        return Path(resource_filename(package, resource)).exists()

    module.resource_filename = resource_filename
    module.resource_exists = resource_exists
    sys.modules["pkg_resources"] = module


class _MGMCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "src" or module.startswith("src."):
            module = f"mgm.{module}"
        return super().find_class(module, name)


class _MGMTokenizerView:
    def __init__(self, source: Any):
        state = getattr(source, "__dict__", {})
        vocab = state.get("vocab")
        toks = state.get("toks")
        if not isinstance(vocab, dict) or not vocab:
            if not isinstance(toks, (list, tuple)) or not toks:
                raise RuntimeError(
                    "The bundled MGM tokenizer does not expose a usable vocabulary."
                )
            vocab = {str(token): i for i, token in enumerate(toks)}
        self.vocab = {str(token): int(index) for token, index in vocab.items()}
        if len(set(self.vocab.values())) != len(self.vocab):
            raise RuntimeError(
                "The bundled MGM tokenizer contains duplicate token ids."
            )
        for token in ("<pad>", "<mask>", "<bos>", "<eos>"):
            if token not in self.vocab:
                raise RuntimeError(
                    f"The bundled MGM tokenizer is missing required token {token!r}."
                )
        self.pad_token_id = self.vocab["<pad>"]
        self.mask_token_id = self.vocab["<mask>"]
        self.bos_token_id = self.vocab["<bos>"]
        self.eos_token_id = self.vocab["<eos>"]

    def encode(self, tokens: Any, *args: Any, **kwargs: Any) -> list[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        if not isinstance(tokens, (list, tuple)):
            raise TypeError(
                "MGM tokenizer input must be a token string or a sequence of token strings."
            )
        try:
            return [self.vocab[str(token)] for token in tokens]
        except KeyError as exc:
            raise ValueError(
                f"MGM tokenizer vocabulary does not contain token {exc.args[0]!r}."
            ) from exc


def _load_mgm_pickle(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        return _MGMCompatUnpickler(handle).load()


def _load_mgm_tokenizer(path: str | Path) -> _MGMTokenizerView:
    return _MGMTokenizerView(_load_mgm_pickle(path))


def _safe_fragment(value: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return fragment[-32:] if fragment else "model"


def _identity(prefix: str, payload: Sequence[Any]) -> str:
    raw = "\x1f".join(str(x) for x in payload)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _validate_matrix(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            "Foundation-model embeddings require a two-dimensional matrix."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Foundation-model embeddings require finite abundance values.")
    if np.any(arr < 0):
        raise ValueError(
            "Foundation-model embeddings require non-negative abundance values."
        )
    return arr


def _resolve_device(torch: Any, requested: str | None) -> str:
    if requested:
        return str(requested)
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _hidden_size_from_config(config: Any) -> int:
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise RuntimeError("Could not determine foundation-model hidden dimension.")


def _hidden_size(model: Any) -> int:
    return _hidden_size_from_config(getattr(model, "config", None))


def _feature_names(names: Sequence[str] | None, width: int) -> list[str]:
    if names is None:
        raise ValueError(
            "Foundation-model transformations require taxonomic feature names."
        )
    out = [str(x) for x in names]
    if len(out) != int(width):
        raise ValueError(
            f"Taxonomic feature-name count differs from matrix width: expected {width}, got {len(out)}."
        )
    return out


def _waypoint_taxon_name(value: Any) -> str:
    parts = re.split(r"(?:[|;]|___(?=[dkpcofgst]__))", str(value))
    return ";".join(part.strip() for part in parts if part.strip())


def _normalise_resolutions(values: Sequence[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ("genus",)
    if isinstance(values, str):
        values = (values,)
    out = tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )
    if not out:
        raise ValueError("At least one foundation-model resolution is required.")
    return out


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _default_cache_root() -> Path:
    configured = os.environ.get("MLLABIOME_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve() / "foundation_embeddings"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "mllabiome" / "foundation_embeddings"


def _cache_root(value: str | os.PathLike[str] | None) -> Path:
    return (
        _default_cache_root() if value is None else Path(value).expanduser().resolve()
    )


def _json_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _row_digest(row: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(row, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _load_row(path: Path, width: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(
            -1
        )
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink()
        return None
    if value.shape != (int(width),) or not np.all(np.isfinite(value)):
        with contextlib.suppress(OSError):
            path.unlink()
        return None
    return value


def _write_row(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npy", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, np.asarray(value, dtype=np.float64), allow_pickle=False)
    os.replace(temporary, path)


def _cached_rows(
    *,
    family: str,
    identity: str,
    runtime_signature: dict[str, Any],
    feature_names: Sequence[str],
    X: np.ndarray,
    width: int,
    cache_enabled: bool,
    cache_dir: str | os.PathLike[str] | None,
    compute: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if not cache_enabled or arr.shape[0] == 0:
        return np.asarray(compute(arr), dtype=np.float64)
    schema = {
        "format": _CACHE_FORMAT,
        "family": family,
        "identity": identity,
        "runtime": runtime_signature,
        "feature_names": [str(x) for x in feature_names],
        "width": int(width),
    }
    schema_key = _json_digest(schema)
    root = _cache_root(cache_dir) / family / schema_key
    row_keys = [_row_digest(arr[index]) for index in range(arr.shape[0])]
    unique: dict[str, np.ndarray] = {}
    for index, key in enumerate(row_keys):
        unique.setdefault(key, arr[index])
    cached: dict[str, np.ndarray] = {}
    for key in unique:
        value = _load_row(root / "rows" / f"{key}.npy", width)
        if value is not None:
            cached[key] = value
    missing = [key for key in unique if key not in cached]
    if missing:
        try:
            from filelock import FileLock
        except Exception as exc:
            raise FoundationModelDependencyError(
                "Foundation-model embedding caching requires filelock. Install the mllabiome foundation extra."
            ) from exc
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root / ".lock")):
            pending: list[str] = []
            for key in missing:
                value = _load_row(root / "rows" / f"{key}.npy", width)
                if value is None:
                    pending.append(key)
                else:
                    cached[key] = value
            if pending:
                matrix = np.vstack([unique[key] for key in pending])
                generated = np.asarray(compute(matrix), dtype=np.float64)
                expected = (len(pending), int(width))
                if generated.shape != expected:
                    raise RuntimeError(
                        f"Foundation-model embedding shape {generated.shape} differs from expected {expected}."
                    )
                if not np.all(np.isfinite(generated)):
                    raise RuntimeError(
                        "Foundation-model embedding contains non-finite values."
                    )
                for index, key in enumerate(pending):
                    value = np.asarray(generated[index], dtype=np.float64)
                    _write_row(root / "rows" / f"{key}.npy", value)
                    cached[key] = value
            metadata = root / "metadata.json"
            if not metadata.exists():
                metadata.write_text(
                    json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8"
                )
    return np.vstack([cached[key] for key in row_keys]).astype(np.float64, copy=False)


def _default_resource_root() -> Path:
    configured = os.environ.get("MLLABIOME_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve() / "foundation_resources"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "mllabiome" / "foundation_resources"


def _resource_root(value: str | os.PathLike[str] | None) -> Path:
    return (
        _default_resource_root()
        if value is None
        else Path(value).expanduser().resolve()
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_resource(url: str, target: Path) -> Path:
    if target.is_file() and target.stat().st_size > 0:
        return target
    try:
        from filelock import FileLock
    except Exception as exc:
        raise FoundationModelDependencyError(
            "Foundation-model resource downloads require filelock. Install the mllabiome foundation extra."
        ) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock"):
        if target.is_file() and target.stat().st_size > 0:
            return target
        request = urllib.request.Request(
            url, headers={"User-Agent": "mllabiome-foundation-models"}
        )
        temporary = target.with_name(target.name + ".part")
        with (
            urllib.request.urlopen(request) as response,
            open(temporary, "wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        os.replace(temporary, target)
    return target


def _load_module(path: str | Path, prefix: str) -> Any:
    resolved = Path(path).expanduser().resolve()
    name = f"{prefix}_{hashlib.sha256(str(resolved).encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load author inference module from {resolved}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _species_terminal(value: str) -> str:
    text = str(value).strip()
    matches = re.findall(r"(?:^|[;|])\s*s__([^;|]+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    parts = [part.strip() for part in re.split(r"[;|]", text) if part.strip()]
    terminal = parts[-1] if parts else text
    terminal = re.sub(
        r"^(?:species(?:__|[:=])?|s(?:__|[:=])?)", "", terminal, flags=re.IGNORECASE
    )
    return terminal.strip()


def _species_alias(value: str) -> str:
    text = _species_terminal(value)
    text = re.sub(r"[_\s]+", " ", text).strip().casefold()
    return text


def _coordinate_metadata(prefix: str, width: int) -> list[TransformationCoordinate]:
    return [
        TransformationCoordinate(
            name=f"{prefix}_dim_{i}",
            coordinate_type="foundation_embedding",
            anchor_feature=None,
            components=(),
            coefficients=(),
            exact_feature_identity=False,
        )
        for i in range(int(width))
    ]


def _project_embedding_input(
    X: np.ndarray, width: int | None, label: str
) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2 or width is None or arr.shape[1] != int(width):
        raise ValueError(
            f"{label} model-input width differs from fitted embedding width."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} model-input projection requires finite values.")
    return arr


def _github_source_tree(owner: str, repo: str, revision: str, root: Path) -> Path:
    target = root / f"{repo}-{_safe_fragment(revision)}"
    if target.is_dir() and (target / "README.md").is_file():
        return target
    try:
        from filelock import FileLock
    except Exception as exc:
        raise FoundationModelDependencyError(
            "Foundation-model source downloads require filelock. Install the mllabiome foundation extra."
        ) from exc
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock"):
        if target.is_dir() and (target / "README.md").is_file():
            return target
        archive = root / f"{repo}-{_safe_fragment(revision)}.zip"
        _download_resource(
            f"https://codeload.github.com/{owner}/{repo}/zip/{revision}",
            archive,
        )
        temporary = Path(tempfile.mkdtemp(prefix=f"{repo}-", dir=root))
        try:
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(temporary)
            directories = [path for path in temporary.iterdir() if path.is_dir()]
            if len(directories) != 1:
                raise RuntimeError(f"Unexpected {repo} source archive layout.")
            if target.exists():
                shutil.rmtree(target)
            os.replace(directories[0], target)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    return target


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*.py") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


class WaypointEmbedding:
    reference = WAYPOINT_REFERENCE
    foundation_model = True
    name = "waypoint_frozen"
    composition_scope = "joint"
    feature_filter = None

    def __init__(
        self,
        model: str = "outpost-bio/Waypoint-6m",
        *,
        pooling: str = "last_token",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
        taxonomy_format: str = "full",
        resolutions: Sequence[str] | str | None = ("genus",),
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if pooling not in {"mean", "first_token", "last_token", "cls_token"}:
            raise ValueError(
                "Waypoint pooling must be one of 'mean', 'first_token', 'last_token', or 'cls_token'."
            )
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1.")
        if int(max_length) < 2:
            raise ValueError("max_length must be at least 2.")
        if taxonomy_format not in {"full", "genus"}:
            raise ValueError("Waypoint taxonomy_format must be 'full' or 'genus'.")
        self.model = str(model)
        self.pooling = str(pooling)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = None if device is None else str(device)
        self.taxonomy_format = str(taxonomy_format)
        self.resolutions = _normalise_resolutions(resolutions)
        self.cache = bool(cache)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.identity = _identity(
            f"waypoint_frozen_{_safe_fragment(Path(self.model).name)}_{_safe_fragment(self.pooling)}",
            (
                self.model,
                self.pooling,
                self.max_length,
                self.taxonomy_format,
                self.resolutions,
            ),
        )
        self.feature_names_: list[str] | None = None
        self.n_features_in_: int | None = None
        self.n_features_out_: int | None = None
        self._runtime: tuple[Any, Any, Any, Any, Any, str] | None = None

    def fresh(self) -> WaypointEmbedding:
        return type(self)(
            self.model,
            pooling=self.pooling,
            batch_size=self.batch_size,
            max_length=self.max_length,
            device=self.device,
            taxonomy_format=self.taxonomy_format,
            resolutions=self.resolutions,
            cache=self.cache,
            cache_dir=self.cache_dir,
        )

    def bind_schema(
        self,
        feature_names: Sequence[str],
        *,
        feature_blocks: Any = None,
    ) -> WaypointEmbedding:
        self.feature_names_ = [str(x) for x in feature_names]
        return self

    def runtime_provenance(self) -> dict[str, Any]:
        return {
            "waypoint_bio": _distribution_version("waypoint-bio"),
            "torch": _distribution_version("torch"),
            "transformers": _distribution_version("transformers"),
            "filelock": _distribution_version("filelock"),
        }

    def _runtime_signature(self) -> dict[str, Any]:
        return {
            **self.runtime_provenance(),
            "model": self.model,
            "pooling": self.pooling,
            "max_length": self.max_length,
            "taxonomy_format": self.taxonomy_format,
            "taxonomy_adapter": "mllabiome-lineage-normalizer-v2",
        }

    def _load_runtime(self) -> tuple[Any, Any, Any, Any, Any, str]:
        if self._runtime is not None:
            return self._runtime
        try:
            import torch
            from torch.utils.data import DataLoader
            from transformers import AutoModel
            from waypoint_bio import matrix_to_waypoint_df
            from waypoint_bio.dataset import try_load_token_std_means
            from waypoint_bio.embed import tokenize_for_embedding
            from waypoint_bio.models import _pool
            from waypoint_bio.tokenizer import load_tokenizer
        except Exception as exc:
            raise FoundationModelDependencyError(
                "WaypointEmbedding requires the mllabiome foundation extra. Run `uv sync --extra foundation`, request access to the selected Outpost Bio checkpoint, and authenticate with `hf auth login`."
            ) from exc
        resolved_device = _resolve_device(torch, self.device)
        key = (self.model, resolved_device)
        cached = _WAYPOINT_RUNTIME_CACHE.get(key)
        if cached is None:
            tokenizer = load_tokenizer(self.model)
            model = AutoModel.from_pretrained(self.model, trust_remote_code=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.eval()
            model.to(resolved_device)
            token_std_means = try_load_token_std_means(self.model)
            cached = (model, tokenizer, token_std_means, _pool)
            _WAYPOINT_RUNTIME_CACHE[key] = cached
        self._runtime = (
            torch,
            DataLoader,
            matrix_to_waypoint_df,
            tokenize_for_embedding,
            cached,
            resolved_device,
        )
        return self._runtime

    def fit(self, X: np.ndarray) -> WaypointEmbedding:
        arr = _validate_matrix(X)
        self.feature_names_ = _feature_names(self.feature_names_, arr.shape[1])
        try:
            from transformers import AutoConfig
        except Exception as exc:
            raise FoundationModelDependencyError(
                "WaypointEmbedding requires the mllabiome foundation extra. Run `uv sync --extra foundation`, request access to the selected Outpost Bio checkpoint, and authenticate with `hf auth login`."
            ) from exc
        config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.n_features_in_ = int(arr.shape[1])
        self.n_features_out_ = _hidden_size_from_config(config)
        return self

    def apply_fitted_pair(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        train = _validate_matrix(X_tr)
        test = _validate_matrix(X_te)
        joined = np.concatenate((train, test), axis=0)
        embedded = self.apply(joined)
        cut = int(train.shape[0])
        return embedded[:cut], embedded[cut:]

    def _embed_uncached(self, arr: np.ndarray) -> np.ndarray:
        names = [
            _waypoint_taxon_name(name)
            for name in _feature_names(self.feature_names_, arr.shape[1])
        ]
        (
            torch,
            DataLoader,
            matrix_to_waypoint_df,
            tokenize_for_embedding,
            runtime,
            resolved_device,
        ) = self._load_runtime()
        model, tokenizer, token_std_means, pool = runtime
        matrix = pd.DataFrame(
            arr,
            index=[f"sample_{i}" for i in range(arr.shape[0])],
            columns=names,
        )
        data = matrix_to_waypoint_df(
            matrix,
            taxonomy_format=self.taxonomy_format,
            normalize=True,
            drop_zeros=True,
        )
        unknown_id = int(tokenizer.unk_token_id)
        known_tokens = sum(
            int(tokenizer._convert_token_to_id(str(taxon))) != unknown_id
            for taxa in data["Taxa"]
            for taxon in taxa
        )
        if known_tokens == 0:
            raise ValueError(
                "Waypoint tokenizer recognized no taxa. Check taxonomy rank labels and lineage separators."
            )
        samples = tokenize_for_embedding(
            data,
            tokenizer,
            self.max_length,
            token_std_means,
        )
        loader = DataLoader(samples, batch_size=self.batch_size, shuffle=False)
        embeddings: list[Any] = []
        with torch.inference_mode():
            for batch in loader:
                input_ids = batch["input_ids"].to(resolved_device)
                attention_mask = batch["attention_mask"].to(resolved_device)
                hidden = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
                pooled = pool(hidden, attention_mask, self.pooling)
                embeddings.append(pooled.detach().cpu())
        if not embeddings:
            return np.empty((0, int(self.n_features_out_ or 0)), dtype=np.float64)
        return torch.cat(embeddings, dim=0).numpy().astype(np.float64, copy=False)

    def apply(self, X: np.ndarray) -> np.ndarray:
        arr = _validate_matrix(X)
        if self.n_features_in_ is None or self.n_features_out_ is None:
            raise RuntimeError("WaypointEmbedding has not been fitted.")
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count differs from fitted Waypoint schema: expected {self.n_features_in_}, got {arr.shape[1]}."
            )
        names = _feature_names(self.feature_names_, arr.shape[1])
        return _cached_rows(
            family="waypoint",
            identity=self.identity,
            runtime_signature=self._runtime_signature(),
            feature_names=names,
            X=arr,
            width=self.n_features_out_,
            cache_enabled=self.cache,
            cache_dir=self.cache_dir,
            compute=self._embed_uncached,
        )

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> list[str]:
        if self.n_features_out_ is None:
            raise RuntimeError("WaypointEmbedding has not been fitted.")
        return [f"waypoint_dim_{i}" for i in range(self.n_features_out_)]

    def coordinate_metadata(
        self, input_features: Sequence[str] | None = None
    ) -> list[TransformationCoordinate]:
        return [
            TransformationCoordinate(
                name=name,
                coordinate_type="foundation_embedding",
                anchor_feature=None,
                components=(),
                coefficients=(),
                exact_feature_identity=False,
            )
            for name in self.get_feature_names_out(input_features)
        ]

    def perturbation_geometry(self) -> str:
        return "foundation_embedding"

    def project_model_input(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if (
            arr.ndim != 2
            or self.n_features_out_ is None
            or arr.shape[1] != self.n_features_out_
        ):
            raise ValueError(
                "Waypoint model-input width differs from fitted embedding width."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Waypoint model-input projection requires finite values.")
        return arr

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_runtime"] = None
        return state


class MGMEmbedding:
    reference = MGM_REFERENCE
    foundation_model = True
    name = "mgm_frozen"
    composition_scope = "joint"
    feature_filter = None

    def __init__(
        self,
        model: str | None = None,
        *,
        tokenizer: str | None = None,
        phylogeny: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
        taxonomy_format: str = "full",
        resolutions: Sequence[str] | str | None = ("genus",),
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1.")
        if int(max_length) < 2:
            raise ValueError("max_length must be at least 2.")
        if taxonomy_format not in {"full", "genus"}:
            raise ValueError("MGM taxonomy_format must be 'full' or 'genus'.")
        resolution_scope = _normalise_resolutions(resolutions)
        if resolution_scope != ("genus",):
            raise ValueError("MGMEmbedding supports only the genus resolution.")
        self.model = None if model is None else str(model)
        self.tokenizer = None if tokenizer is None else str(tokenizer)
        self.phylogeny = None if phylogeny is None else str(phylogeny)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = None if device is None else str(device)
        self.taxonomy_format = str(taxonomy_format)
        self.resolutions = resolution_scope
        self.cache = bool(cache)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.identity = _identity(
            "mgm_frozen_mean",
            (
                self.model,
                self.tokenizer,
                self.phylogeny,
                self.max_length,
                self.taxonomy_format,
                self.resolutions,
            ),
        )
        self.feature_names_: list[str] | None = None
        self.n_features_in_: int | None = None
        self.n_features_out_: int | None = None
        self._runtime: tuple[Any, Any, Any, Any, Any, str] | None = None

    def fresh(self) -> MGMEmbedding:
        return type(self)(
            self.model,
            tokenizer=self.tokenizer,
            phylogeny=self.phylogeny,
            batch_size=self.batch_size,
            max_length=self.max_length,
            device=self.device,
            taxonomy_format=self.taxonomy_format,
            resolutions=self.resolutions,
            cache=self.cache,
            cache_dir=self.cache_dir,
        )

    def bind_schema(
        self,
        feature_names: Sequence[str],
        *,
        feature_blocks: Any = None,
    ) -> MGMEmbedding:
        self.feature_names_ = [str(x) for x in feature_names]
        return self

    def runtime_provenance(self) -> dict[str, Any]:
        return {
            "microformer_mgm": _distribution_version("microformer-mgm"),
            "torch": _distribution_version("torch"),
            "transformers": _distribution_version("transformers"),
            "pytorch_lightning": _distribution_version("pytorch-lightning"),
            "numpy": _distribution_version("numpy"),
            "pandas": _distribution_version("pandas"),
            "scikit_learn": _distribution_version("scikit-learn"),
            "tqdm": _distribution_version("tqdm"),
            "filelock": _distribution_version("filelock"),
        }

    def _runtime_signature(self) -> dict[str, Any]:
        return {
            **self.runtime_provenance(),
            "model": self.model or "bundled:resources/general_model",
            "tokenizer": self.tokenizer or "bundled:resources/MicroTokenizer.pkl",
            "phylogeny": self.phylogeny or "bundled:resources/phylogeny.csv",
            "max_length": self.max_length,
            "taxonomy_format": self.taxonomy_format,
            "pooling": "mean_contextualized_genus_tokens",
        }

    def _load_runtime(self) -> tuple[Any, Any, Any, Any, Any, str]:
        if self._runtime is not None:
            return self._runtime
        _ensure_mgm_pkg_resources_compat()
        try:
            import torch
            from mgm.src.MicroCorpus import MicroCorpus
            from torch.utils.data import DataLoader
            from transformers import AutoModel
        except ModuleNotFoundError as exc:
            if exc.name in {"mgm", "torch", "transformers"}:
                raise FoundationModelDependencyError(
                    "MGMEmbedding requires the mllabiome foundation extra. Run `uv sync --extra foundation`."
                ) from exc
            raise FoundationModelDependencyError(
                f"The installed MGM runtime could not be imported because {exc.name!r} is unavailable: {exc}."
            ) from exc
        except Exception as exc:
            raise FoundationModelDependencyError(
                f"The installed MGM runtime failed during import: {type(exc).__name__}: {exc}."
            ) from exc
        model_path, tokenizer_path, phylogeny_path = self._paths()
        resolved_device = _resolve_device(torch, self.device)
        key = (
            str(model_path),
            str(tokenizer_path),
            str(phylogeny_path),
            resolved_device,
        )
        cached = _MGM_RUNTIME_CACHE.get(key)
        if cached is None:
            tokenizer = _load_mgm_tokenizer(tokenizer_path)
            model = AutoModel.from_pretrained(model_path)
            input_embeddings = model.get_input_embeddings()
            vocabulary_limit = int(getattr(input_embeddings, "num_embeddings", 0))
            maximum_token_id = max(tokenizer.vocab.values())
            if vocabulary_limit <= maximum_token_id:
                raise RuntimeError(
                    f"MGM tokenizer/model vocabulary mismatch: maximum tokenizer id {maximum_token_id}, model embedding rows {vocabulary_limit}."
                )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.eval()
            model.to(resolved_device)
            cached = (model, tokenizer)
            _MGM_RUNTIME_CACHE[key] = cached
        model, tokenizer = cached
        self._runtime = (
            torch,
            DataLoader,
            MicroCorpus,
            tokenizer,
            (model, str(phylogeny_path)),
            resolved_device,
        )
        return self._runtime

    def _paths(self) -> tuple[str, str, str]:
        root = _mgm_package_root()
        model = (
            Path(self.model).expanduser().resolve()
            if self.model
            else root / "resources" / "general_model"
        )
        tokenizer = (
            Path(self.tokenizer).expanduser().resolve()
            if self.tokenizer
            else root / "resources" / "MicroTokenizer.pkl"
        )
        phylogeny = (
            Path(self.phylogeny).expanduser().resolve()
            if self.phylogeny
            else root / "resources" / "phylogeny.csv"
        )
        missing = [
            str(path) for path in (model, tokenizer, phylogeny) if not path.exists()
        ]
        if missing:
            raise FoundationModelDependencyError(
                "MGMEmbedding could not locate the author-provided MGM resources: "
                + ", ".join(missing)
            )
        return str(model), str(tokenizer), str(phylogeny)

    def _genus_names(self, width: int) -> list[str]:
        source = _feature_names(self.feature_names_, width)
        out: list[str] = []
        for value in source:
            match = re.search(r"(g__[A-Za-z0-9_]+)", value)
            if match is not None:
                out.append(match.group(1))
                continue
            parts = [part.strip() for part in re.split(r"[;|]", value) if part.strip()]
            terminal = parts[-1] if parts else value.strip()
            terminal = re.sub(
                r"^(?:genus(?:__|[:=])?|g(?:__|[:=])?)",
                "",
                terminal,
                flags=re.IGNORECASE,
            )
            terminal = re.sub(r"[^A-Za-z0-9_]+", "_", terminal).strip("_")
            if not terminal:
                raise ValueError(
                    f"MGM could not derive a genus token from feature name {value!r}."
                )
            out.append(f"g__{terminal}")
        return out

    def _matrix(self, arr: np.ndarray) -> pd.DataFrame:
        names = self._genus_names(arr.shape[1])
        return pd.DataFrame(
            arr,
            index=[f"sample_{i}" for i in range(arr.shape[0])],
            columns=names,
        )

    def fit(self, X: np.ndarray) -> MGMEmbedding:
        arr = _validate_matrix(X)
        self.feature_names_ = _feature_names(self.feature_names_, arr.shape[1])
        try:
            from transformers import AutoConfig
        except Exception as exc:
            raise FoundationModelDependencyError(
                "MGMEmbedding requires the mllabiome foundation extra. Run `uv sync --extra foundation`."
            ) from exc
        model_path, _, _ = self._paths()
        config = AutoConfig.from_pretrained(model_path)
        self.n_features_in_ = int(arr.shape[1])
        self.n_features_out_ = _hidden_size_from_config(config)
        return self

    def apply_fitted_pair(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        train = _validate_matrix(X_tr)
        test = _validate_matrix(X_te)
        joined = np.concatenate((train, test), axis=0)
        embedded = self.apply(joined)
        cut = int(train.shape[0])
        return embedded[:cut], embedded[cut:]

    def _embed_uncached(self, arr: np.ndarray) -> np.ndarray:
        torch, DataLoader, MicroCorpus, tokenizer, runtime, resolved_device = (
            self._load_runtime()
        )
        model, phylogeny_path = runtime
        matrix = self._matrix(arr)
        phylogeny_index = pd.read_csv(phylogeny_path, index_col=0).index.astype(str)
        supported = set(matrix.columns.astype(str)).intersection(set(phylogeny_index))
        if not supported:
            examples = ", ".join(list(matrix.columns.astype(str)[:8]))
            raise ValueError(
                f"MGM taxonomy mapping retained 0 supported genera from {matrix.shape[1]} input features. Mapped examples: {examples}."
            )
        supported_columns = [
            name for name in matrix.columns.astype(str) if name in supported
        ]
        supported_matrix = matrix.loc[:, supported_columns]
        unsupported_samples = np.flatnonzero(
            np.asarray(supported_matrix.sum(axis=1), dtype=np.float64) <= 0
        )
        if unsupported_samples.size:
            raise ValueError(
                f"MGM taxonomy mapping leaves {unsupported_samples.size}/{matrix.shape[0]} samples without supported genera."
            )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            corpus = MicroCorpus(
                tokenizer=tokenizer,
                abu=matrix,
                phylogeny_path=phylogeny_path,
                max_len=self.max_length,
                preprocess=True,
            )
        if len(corpus) != arr.shape[0]:
            raise ValueError(
                "MGM preprocessing dropped one or more samples because no supported genus remained after author-provided taxonomy mapping."
            )
        loader = DataLoader(corpus, batch_size=self.batch_size, shuffle=False)
        embeddings: list[Any] = []
        special_ids = {
            int(tokenizer.pad_token_id),
            int(tokenizer.bos_token_id),
            int(tokenizer.eos_token_id),
        }
        with torch.inference_mode():
            for batch in loader:
                input_ids = batch["input_ids"].to(resolved_device)
                attention_mask = batch["attention_mask"].to(resolved_device)
                hidden = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
                genus_mask = attention_mask.bool()
                for token_id in special_ids:
                    genus_mask = genus_mask & input_ids.ne(token_id)
                counts = genus_mask.sum(dim=1)
                if torch.any(counts == 0):
                    raise ValueError(
                        "MGM preprocessing produced a sample without supported genus tokens."
                    )
                weights = genus_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
                embeddings.append(pooled.detach().cpu())
        if not embeddings:
            return np.empty((0, int(self.n_features_out_ or 0)), dtype=np.float64)
        return torch.cat(embeddings, dim=0).numpy().astype(np.float64, copy=False)

    def apply(self, X: np.ndarray) -> np.ndarray:
        arr = _validate_matrix(X)
        if self.n_features_in_ is None or self.n_features_out_ is None:
            raise RuntimeError("MGMEmbedding has not been fitted.")
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count differs from fitted MGM schema: expected {self.n_features_in_}, got {arr.shape[1]}."
            )
        names = _feature_names(self.feature_names_, arr.shape[1])
        return _cached_rows(
            family="mgm",
            identity=self.identity,
            runtime_signature=self._runtime_signature(),
            feature_names=names,
            X=arr,
            width=self.n_features_out_,
            cache_enabled=self.cache,
            cache_dir=self.cache_dir,
            compute=self._embed_uncached,
        )

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> list[str]:
        if self.n_features_out_ is None:
            raise RuntimeError("MGMEmbedding has not been fitted.")
        return [f"mgm_dim_{i}" for i in range(self.n_features_out_)]

    def coordinate_metadata(
        self, input_features: Sequence[str] | None = None
    ) -> list[TransformationCoordinate]:
        return [
            TransformationCoordinate(
                name=name,
                coordinate_type="foundation_embedding",
                anchor_feature=None,
                components=(),
                coefficients=(),
                exact_feature_identity=False,
            )
            for name in self.get_feature_names_out(input_features)
        ]

    def perturbation_geometry(self) -> str:
        return "foundation_embedding"

    def project_model_input(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if (
            arr.ndim != 2
            or self.n_features_out_ is None
            or arr.shape[1] != self.n_features_out_
        ):
            raise ValueError(
                "MGM model-input width differs from fitted embedding width."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("MGM model-input projection requires finite values.")
        return arr

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_runtime"] = None
        return state


class MBEmbed:
    reference = MBEMBED_REFERENCE
    _WIDTH = 1024
    name = "mbembed_mae30_frozen"
    composition_scope = "joint"
    feature_filter = None
    foundation_model = True

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        species_reference: str | os.PathLike[str] | None = None,
        source_path: str | os.PathLike[str] | None = None,
        revision: str = "main",
        batch_size: int = 32,
        device: str | None = None,
        resolutions: Sequence[str] | str | None = ("species",),
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        resource_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1.")
        resolution_scope = _normalise_resolutions(resolutions)
        if resolution_scope != ("species",):
            raise ValueError("MBEmbed supports only the species resolution.")
        self.model_path = None if model_path is None else str(model_path)
        self.species_reference = (
            None if species_reference is None else str(species_reference)
        )
        self.source_path = None if source_path is None else str(source_path)
        self.revision = str(revision)
        self.batch_size = int(batch_size)
        self.device = None if device is None else str(device)
        self.resolutions = resolution_scope
        self.cache = bool(cache)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.resource_dir = None if resource_dir is None else str(resource_dir)
        self.identity = _identity(
            "mbembed_mae30_frozen",
            (
                self.model_path,
                self.species_reference,
                self.source_path,
                self.revision,
                self.resolutions,
            ),
        )
        self.feature_names_: list[str] | None = None
        self.n_features_in_: int | None = None
        self.n_features_out_: int | None = None
        self._runtime: tuple[Any, Any, Any, str] | None = None

    def fresh(self) -> MBEmbed:
        return type(self)(
            self.model_path,
            species_reference=self.species_reference,
            source_path=self.source_path,
            revision=self.revision,
            batch_size=self.batch_size,
            device=self.device,
            resolutions=self.resolutions,
            cache=self.cache,
            cache_dir=self.cache_dir,
            resource_dir=self.resource_dir,
        )

    def bind_schema(
        self,
        feature_names: Sequence[str],
        *,
        feature_blocks: Any = None,
    ) -> MBEmbed:
        self.feature_names_ = [str(x) for x in feature_names]
        return self

    def _resources(self) -> tuple[Path, Path, Path]:
        root = (
            _resource_root(self.resource_dir)
            / "mbembed"
            / _safe_fragment(self.revision)
        )
        source = (
            Path(self.source_path).expanduser().resolve()
            if self.source_path
            else _download_resource(
                f"https://raw.githubusercontent.com/LironZa/MBEmbed/{self.revision}/embed_new_samples.py",
                root / "embed_new_samples.py",
            )
        )
        model = (
            Path(self.model_path).expanduser().resolve()
            if self.model_path
            else _download_resource(
                f"https://raw.githubusercontent.com/LironZa/MBEmbed/{self.revision}/model/mae30_encoder_full.pkl",
                root / "mae30_encoder_full.pkl",
            )
        )
        reference = (
            Path(self.species_reference).expanduser().resolve()
            if self.species_reference
            else _download_resource(
                f"https://raw.githubusercontent.com/LironZa/MBEmbed/{self.revision}/model/model_species_reference.csv",
                root / "model_species_reference.csv",
            )
        )
        missing = [
            str(path) for path in (source, model, reference) if not path.is_file()
        ]
        if missing:
            raise FoundationModelDependencyError(
                "MBEmbed could not locate the author-provided inference resources: "
                + ", ".join(missing)
            )
        return source, model, reference

    def runtime_provenance(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "torch": _distribution_version("torch"),
            "revision": self.revision,
        }
        with contextlib.suppress(Exception):
            source, model, reference = self._resources()
            payload.update(
                {
                    "source_sha256": _sha256_file(source),
                    "model_sha256": _sha256_file(model),
                    "species_reference_sha256": _sha256_file(reference),
                }
            )
        return payload

    def _runtime_signature(self) -> dict[str, Any]:
        source, model, reference = self._resources()
        return {
            "revision": self.revision,
            "source_sha256": _sha256_file(source),
            "model_sha256": _sha256_file(model),
            "species_reference_sha256": _sha256_file(reference),
            "torch": _distribution_version("torch"),
        }

    def _load_runtime(self) -> tuple[Any, Any, Any, str]:
        try:
            import torch
        except Exception as exc:
            raise FoundationModelDependencyError("MBEmbed requires PyTorch.") from exc
        if self.model_path is None:
            raise FoundationModelDependencyError(
                "MBEmbed's official mae30_encoder_full.pkl is not portable in a clean Python environment because it references an unreleased top-level LabData module. The authors' public repository does not provide that module and upstream issue LironZa/MBEmbed#1 remains unresolved. mllabiome will not reconstruct or guess the missing model class. Use a portable author-provided checkpoint via model_path=... if one becomes available."
            )
        source, model_path, reference_path = self._resources()
        resolved_device = _resolve_device(torch, self.device)
        key = (str(source), str(model_path), resolved_device)
        cached = _MBEMBED_RUNTIME_CACHE.get(key)
        if cached is None:
            module = _load_module(source, "mllabiome_mbembed_author")
            try:
                model = module.load_model(str(model_path), resolved_device)
            except Exception as exc:
                text = str(exc).casefold()
                if "weights_only" not in text and "weights only" not in text:
                    raise
                model = torch.load(
                    str(model_path), map_location=resolved_device, weights_only=False
                )
                model.eval()
                model.to(resolved_device)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.eval()
            cached = (module, model, reference_path, resolved_device)
            _MBEMBED_RUNTIME_CACHE[key] = cached
        self._runtime = cached
        return cached

    def _mapped_matrix(self, arr: np.ndarray, reference_path: Path) -> pd.DataFrame:
        names = _feature_names(self.feature_names_, arr.shape[1])
        reference = pd.read_csv(reference_path)
        if "species_name" not in reference.columns:
            raise ValueError(
                "MBEmbed species reference is missing the species_name column."
            )
        expected = [str(value) for value in reference["species_name"].tolist()]
        exact = {value.casefold(): value for value in expected}
        alias_candidates: dict[str, set[str]] = {}
        for value in expected:
            alias_candidates.setdefault(_species_alias(value), set()).add(value)
        aliases = {
            key: next(iter(values))
            for key, values in alias_candidates.items()
            if key and len(values) == 1
        }
        mapping: dict[int, str] = {}
        for index, name in enumerate(names):
            terminal = _species_terminal(name)
            matched = exact.get(terminal.casefold())
            if matched is None:
                matched = aliases.get(_species_alias(name))
            if matched is not None:
                mapping[index] = matched
        if not mapping:
            examples = ", ".join(names[:8])
            raise ValueError(
                f"MBEmbed species mapping retained 0/{len(names)} input features. Input examples: {examples}."
            )
        totals = np.asarray(arr.sum(axis=1), dtype=np.float64)
        empty = np.flatnonzero(totals <= 0)
        if empty.size:
            raise ValueError(
                f"MBEmbed received {empty.size}/{arr.shape[0]} samples with zero total abundance."
            )
        relative = arr / totals[:, None]
        values: dict[str, np.ndarray] = {}
        for index, species in mapping.items():
            column = np.asarray(relative[:, index], dtype=np.float64)
            values[species] = (
                values.get(species, np.zeros(arr.shape[0], dtype=np.float64)) + column
            )
        frame = pd.DataFrame(values, index=[f"sample_{i}" for i in range(arr.shape[0])])
        retained = np.asarray(frame.sum(axis=1), dtype=np.float64)
        unsupported = np.flatnonzero(retained <= 0)
        if unsupported.size:
            raise ValueError(
                f"MBEmbed species mapping leaves {unsupported.size}/{arr.shape[0]} samples without supported species."
            )
        return frame

    def _embed_uncached(self, arr: np.ndarray) -> np.ndarray:
        module, model, reference_path, resolved_device = self._load_runtime()
        frame = self._mapped_matrix(arr, reference_path)
        chunks: list[np.ndarray] = []
        stream = io.StringIO()
        for start in range(0, len(frame), self.batch_size):
            batch = frame.iloc[start : start + self.batch_size]
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                tensor, processed = module.arrange_input(batch, str(reference_path))
                embedded = module.embed_samples(
                    tensor, processed, model, resolved_device
                )
            chunks.append(np.asarray(embedded, dtype=np.float64))
        if not chunks:
            return np.empty((0, int(self.n_features_out_ or 0)), dtype=np.float64)
        return np.concatenate(chunks, axis=0)

    def fit(self, X: np.ndarray) -> MBEmbed:
        arr = _validate_matrix(X)
        self.feature_names_ = _feature_names(self.feature_names_, arr.shape[1])
        self.n_features_in_ = int(arr.shape[1])
        self.n_features_out_ = self._WIDTH
        return self

    def apply_fitted_pair(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        train = _validate_matrix(X_tr)
        test = _validate_matrix(X_te)
        joined = np.concatenate((train, test), axis=0)
        embedded = self.apply(joined)
        cut = int(train.shape[0])
        return embedded[:cut], embedded[cut:]

    def apply(self, X: np.ndarray) -> np.ndarray:
        arr = _validate_matrix(X)
        if self.n_features_in_ is None or self.n_features_out_ is None:
            raise RuntimeError("MBEmbed has not been fitted.")
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count differs from fitted MBEmbed schema: expected {self.n_features_in_}, got {arr.shape[1]}."
            )
        names = _feature_names(self.feature_names_, arr.shape[1])
        return _cached_rows(
            family="mbembed",
            identity=self.identity,
            runtime_signature=self._runtime_signature(),
            feature_names=names,
            X=arr,
            width=self.n_features_out_,
            cache_enabled=self.cache,
            cache_dir=self.cache_dir,
            compute=self._embed_uncached,
        )

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> list[str]:
        if self.n_features_out_ is None:
            raise RuntimeError("MBEmbed has not been fitted.")
        return [f"mbembed_dim_{i}" for i in range(self.n_features_out_)]

    def coordinate_metadata(
        self, input_features: Sequence[str] | None = None
    ) -> list[TransformationCoordinate]:
        if self.n_features_out_ is None:
            raise RuntimeError("MBEmbed has not been fitted.")
        return _coordinate_metadata("mbembed", self.n_features_out_)

    def perturbation_geometry(self) -> str:
        return "foundation_embedding"

    def project_model_input(self, X: np.ndarray) -> np.ndarray:
        return _project_embedding_input(X, self.n_features_out_, "MBEmbed")

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_runtime"] = None
        return state


class MGM2Embedding:
    reference = MGM2_REFERENCE
    name = "mgm2_frozen"
    composition_scope = "joint"
    feature_filter = None
    foundation_model = True
    _WIDTHS = {"small": 224, "medium": 320, "large": 384, "xlarge": 512}

    def __init__(
        self,
        model: str = "small",
        *,
        embedding_strategy: str = "cls",
        batch_size: int = 64,
        top_k_otus: int = 768,
        device: str | None = None,
        resolutions: Sequence[str] | str | None = ("species",),
        repo_id: str = "LudensZhang/MGM2",
        revision: str = "54bae7e240033eab2a52019a910038e9ab0605a3",
        source_revision: str = "main",
        source_dir: str | os.PathLike[str] | None = None,
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        resource_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        model_key = str(model).strip().lower().replace("mgm2-", "")
        if model_key not in self._WIDTHS:
            raise ValueError(
                "MGM2 model must be one of 'small', 'medium', 'large', or 'xlarge'."
            )
        if embedding_strategy not in {"cls", "mean_pool", "qwen3_proj"}:
            raise ValueError(
                "MGM2 embedding_strategy must be 'cls', 'mean_pool', or 'qwen3_proj'."
            )
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1.")
        if int(top_k_otus) < 1:
            raise ValueError("top_k_otus must be at least 1.")
        resolution_scope = _normalise_resolutions(resolutions)
        self.model = model_key
        self.embedding_strategy = str(embedding_strategy)
        self.batch_size = int(batch_size)
        self.top_k_otus = int(top_k_otus)
        self.device = None if device is None else str(device)
        self.resolutions = resolution_scope
        self.repo_id = str(repo_id)
        self.revision = str(revision)
        self.source_revision = str(source_revision)
        self.source_dir = None if source_dir is None else str(source_dir)
        self.cache = bool(cache)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.resource_dir = None if resource_dir is None else str(resource_dir)
        self.identity = _identity(
            f"mgm2_{self.model}_{self.embedding_strategy}",
            (
                self.model,
                self.embedding_strategy,
                self.top_k_otus,
                self.repo_id,
                self.revision,
                self.source_revision,
                self.resolutions,
            ),
        )
        self.feature_names_: list[str] | None = None
        self.n_features_in_: int | None = None
        self.n_features_out_: int | None = None

    def fresh(self) -> MGM2Embedding:
        return type(self)(
            self.model,
            embedding_strategy=self.embedding_strategy,
            batch_size=self.batch_size,
            top_k_otus=self.top_k_otus,
            device=self.device,
            resolutions=self.resolutions,
            repo_id=self.repo_id,
            revision=self.revision,
            source_revision=self.source_revision,
            source_dir=self.source_dir,
            cache=self.cache,
            cache_dir=self.cache_dir,
            resource_dir=self.resource_dir,
        )

    def bind_schema(
        self,
        feature_names: Sequence[str],
        *,
        feature_blocks: Any = None,
    ) -> MGM2Embedding:
        self.feature_names_ = [str(x) for x in feature_names]
        return self

    def _source_root(self) -> Path:
        if self.source_dir:
            root = Path(self.source_dir).expanduser().resolve()
            if not (root / "scripts" / "extract_community_embeddings.py").is_file():
                raise FoundationModelDependencyError(
                    f"Invalid MGM2 source directory: {root}"
                )
            return root
        return _github_source_tree(
            "HUST-NingKang-Lab",
            "MGM2",
            self.source_revision,
            _resource_root(self.resource_dir) / "mgm2" / "source",
        )

    def _release_files(self) -> tuple[Path, Path, Path, Path]:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise FoundationModelDependencyError(
                "MGM2Embedding requires huggingface-hub. Install the mllabiome foundation extra."
            ) from exc
        root = Path(
            snapshot_download(
                repo_id=self.repo_id,
                revision=self.revision,
                allow_patterns=(
                    f"models/{self.model}/model.ckpt",
                    f"models/{self.model}/config.json",
                    "embeddings/reference_ntv3_embeddings.h5",
                    "embeddings/taxonomy_fallback.h5",
                ),
            )
        )
        model_dir = root / "models" / self.model
        checkpoint = model_dir / "model.ckpt"
        config = model_dir / "config.json"
        reference = root / "embeddings" / "reference_ntv3_embeddings.h5"
        fallback = root / "embeddings" / "taxonomy_fallback.h5"
        expected = {
            "checkpoint": checkpoint,
            "config": config,
            "reference_ntv3": reference,
            "taxonomy_fallback": fallback,
        }
        missing = [
            f"{name}={path}" for name, path in expected.items() if not path.is_file()
        ]
        if missing:
            raise FoundationModelDependencyError(
                "MGM2 release snapshot is incomplete: " + "; ".join(missing)
            )
        return model_dir, config, reference, fallback

    def runtime_provenance(self) -> dict[str, Any]:
        payload = {
            "torch": _distribution_version("torch"),
            "pytorch_lightning": _distribution_version("pytorch-lightning"),
            "huggingface_hub": _distribution_version("huggingface-hub"),
            "h5py": _distribution_version("h5py"),
            "repo_id": self.repo_id,
            "revision": self.revision,
            "source_revision": self.source_revision,
        }
        with contextlib.suppress(Exception):
            payload["source_sha256"] = _tree_sha256(self._source_root())
        return payload

    def _runtime_signature(self) -> dict[str, Any]:
        source_root = self._source_root()
        model_dir, config, reference, fallback = self._release_files()
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "source_revision": self.source_revision,
            "source_sha256": _tree_sha256(source_root),
            "model": self.model,
            "strategy": self.embedding_strategy,
            "top_k_otus": self.top_k_otus,
            "checkpoint_sha256": _sha256_file(model_dir / "model.ckpt"),
            "config_sha256": _sha256_file(config),
            "reference_ntv3_sha256": _sha256_file(reference),
            "fallback_sha256": _sha256_file(fallback),
            "torch": _distribution_version("torch"),
            "pytorch_lightning": _distribution_version("pytorch-lightning"),
        }

    def fit(self, X: np.ndarray) -> MGM2Embedding:
        arr = _validate_matrix(X)
        self.feature_names_ = _feature_names(self.feature_names_, arr.shape[1])
        self.n_features_in_ = int(arr.shape[1])
        _, config_path, _, _ = self._release_files()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        hidden_size = int(
            config.get("model", {}).get("hidden_size", self._WIDTHS[self.model])
        )
        if self.embedding_strategy == "qwen3_proj":
            alignment = config.get("qwen3_alignment", {})
            if not bool(alignment.get("enabled", False)):
                raise ValueError(
                    "MGM2 qwen3_proj requires a checkpoint with Qwen3 alignment enabled."
                )
            self.n_features_out_ = int(alignment.get("projection_dim", 4096))
        else:
            self.n_features_out_ = hidden_size
        return self

    def _device_name(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
        except Exception as exc:
            raise FoundationModelDependencyError(
                "MGM2Embedding requires torch. Install the mllabiome foundation extra."
            ) from exc
        return _resolve_device(torch, None)

    def _embed_uncached(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == 0:
            return np.empty((0, int(self.n_features_out_ or 0)), dtype=np.float64)
        if np.any(arr.sum(axis=1) <= 0):
            count = int(np.sum(arr.sum(axis=1) <= 0))
            raise ValueError(
                f"MGM2 received {count}/{arr.shape[0]} samples with zero total abundance."
            )
        names = _feature_names(self.feature_names_, arr.shape[1])
        source_root = self._source_root()
        model_dir, _, reference, fallback = self._release_files()
        try:
            import h5py
        except Exception as exc:
            raise FoundationModelDependencyError(
                "MGM2Embedding requires h5py. Install the mllabiome foundation extra."
            ) from exc
        with tempfile.TemporaryDirectory(prefix="mllabiome-mgm2-") as temporary:
            tmp = Path(temporary)
            table = tmp / "community.csv"
            dataset = tmp / "community.pkl"
            resolved = tmp / "resolved_embeddings.h5"
            output = tmp / "community_embeddings.npz"
            sample_ids = [f"sample_{i}" for i in range(arr.shape[0])]
            frame = pd.DataFrame(arr.T, index=names, columns=sample_ids)
            frame.index.name = "feature_id"
            frame.to_csv(table)
            build_cmd = [
                sys.executable,
                str(source_root / "src" / "data" / "build_dataset.py"),
                "--otu_table",
                str(table),
                "--format",
                "csv",
                "--ntv3_embedding",
                str(reference),
                "--fallback_embedding",
                str(fallback),
                "--resolved_embedding",
                str(resolved),
                "--top_k_otus",
                str(self.top_k_otus),
                "--output",
                str(dataset),
            ]
            extract_cmd = [
                sys.executable,
                str(source_root / "scripts" / "extract_community_embeddings.py"),
                "--model_dir",
                str(model_dir),
                "--data_file",
                str(dataset),
                "--output",
                str(output),
                "--embedding_strategy",
                self.embedding_strategy,
                "--batch_size",
                str(self.batch_size),
                "--device",
                self._device_name(),
                "--disable_progress",
            ]
            for command in (build_cmd, extract_cmd):
                completed = subprocess.run(
                    command,
                    cwd=source_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    tail = completed.stdout[-6000:]
                    raise RuntimeError(
                        f"MGM2 author inference command failed with exit code {completed.returncode}:\n{tail}"
                    )
            payload = np.load(output, allow_pickle=False)
            embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
            returned_ids = [str(value) for value in payload["sample_ids"]]
            if returned_ids != sample_ids:
                position = {
                    sample_id: index for index, sample_id in enumerate(returned_ids)
                }
                if set(position) != set(sample_ids):
                    raise RuntimeError(
                        "MGM2 output sample IDs differ from the requested samples."
                    )
                embeddings = embeddings[
                    [position[sample_id] for sample_id in sample_ids]
                ]
            return embeddings

    def apply_fitted_pair(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        train = _validate_matrix(X_tr)
        test = _validate_matrix(X_te)
        joined = np.concatenate((train, test), axis=0)
        embedded = self.apply(joined)
        cut = int(train.shape[0])
        return embedded[:cut], embedded[cut:]

    def apply(self, X: np.ndarray) -> np.ndarray:
        arr = _validate_matrix(X)
        if self.n_features_in_ is None or self.n_features_out_ is None:
            raise RuntimeError("MGM2Embedding has not been fitted.")
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count differs from fitted MGM2 schema: expected {self.n_features_in_}, got {arr.shape[1]}."
            )
        names = _feature_names(self.feature_names_, arr.shape[1])
        return _cached_rows(
            family="mgm2",
            identity=self.identity,
            runtime_signature=self._runtime_signature(),
            feature_names=names,
            X=arr,
            width=self.n_features_out_,
            cache_enabled=self.cache,
            cache_dir=self.cache_dir,
            compute=self._embed_uncached,
        )

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> list[str]:
        if self.n_features_out_ is None:
            raise RuntimeError("MGM2Embedding has not been fitted.")
        return [
            f"mgm2_{self.model}_{self.embedding_strategy}_dim_{i}"
            for i in range(self.n_features_out_)
        ]

    def coordinate_metadata(
        self, input_features: Sequence[str] | None = None
    ) -> list[TransformationCoordinate]:
        if self.n_features_out_ is None:
            raise RuntimeError("MGM2Embedding has not been fitted.")
        prefix = f"mgm2_{self.model}_{self.embedding_strategy}"
        return _coordinate_metadata(prefix, self.n_features_out_)

    def perturbation_geometry(self) -> str:
        return "foundation_embedding"

    def project_model_input(self, X: np.ndarray) -> np.ndarray:
        return _project_embedding_input(X, self.n_features_out_, "MGM2")
