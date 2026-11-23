from __future__ import annotations
import importlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class HistoricalConfigurationError(RuntimeError):
    pass


def recorded_configuration(root: Path, config_id: str) -> dict[str, Any]:
    from . import cooperative_store

    if cooperative_store.enabled(Path(root)):
        from ._config_store import (
            _artifact_path,
            _file_metadata,
            _validate_artifact_context,
        )

        path = _artifact_path(Path(root), config_id)
        if not path.is_file():
            raise HistoricalConfigurationError(
                f"No completed historical configuration artifact for {config_id!r} at {path}"
            )
        metadata = _file_metadata(path)
        _validate_artifact_context(Path(root), metadata)
        row = dict(metadata.get("configuration") or {})
        row["config_id"] = str(config_id)
        return row
    path = Path(root) / "configs.db"
    if not path.is_file():
        raise HistoricalConfigurationError(
            f"No experiment configuration database exists at {path}."
        )
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT * FROM configs WHERE config_id = ?", (str(config_id),)
        ).fetchone()
        if row is not None:
            return dict(row)
        row = connection.execute(
            "SELECT configuration_json FROM config_artifacts WHERE config_id = ?",
            (str(config_id),),
        ).fetchone()
        if row is not None:
            return json.loads(row[0])
    except sqlite3.Error as exc:
        raise HistoricalConfigurationError(
            f"Cannot read historical configuration {config_id!r}: {exc}"
        ) from exc
    finally:
        connection.close()
    raise HistoricalConfigurationError(
        f"Historical configuration {config_id!r} is not recorded in {path}."
    )


def _import_recorded(path: str) -> Any:
    if not isinstance(path, str) or not path or "<" in path:
        raise HistoricalConfigurationError(f"Invalid recorded object path {path!r}.")
    parts = path.split(".")
    last_error = None
    for n in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:n]))
        except ImportError as exc:
            last_error = exc
            continue
        obj = module
        try:
            for component in parts[n:]:
                obj = getattr(obj, component)
        except AttributeError as exc:
            raise HistoricalConfigurationError(
                f"Recorded object {path!r} is no longer available."
            ) from exc
        return obj
    raise HistoricalConfigurationError(
        f"Cannot import recorded object {path!r}: {last_error}."
    )


def _restore_value(value: Any) -> Any:
    if isinstance(value, str) and value in {"nan", "inf", "-inf"}:
        return float(value)
    if isinstance(value, list):
        return [_restore_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "class" in value and "params" in value:
        return _restore_estimator(value["class"], value["params"])
    if "callable" in value:
        return _import_recorded(value["callable"])
    if "class" in value and "state" in value:
        raise HistoricalConfigurationError(
            f"Recorded object {value['class']!r} needs a custom state deserializer."
        )
    return {key: _restore_value(item) for key, item in value.items()}


def _restore_estimator(path: str, saved_params: Mapping[str, Any]) -> Any:
    from .estimator_protocol import is_estimator_instance

    if not isinstance(saved_params, Mapping):
        raise HistoricalConfigurationError(
            "Recorded learner parameters are not a mapping."
        )
    cls = _import_recorded(path)
    if not isinstance(cls, type) or not issubclass(cls, object):
        raise HistoricalConfigurationError(f"Recorded learner {path!r} is not a class.")
    if path == "sklearn.pipeline.Pipeline":
        keys = {"steps", "memory", "transform_input", "verbose"}
    elif path == "sklearn.pipeline.FeatureUnion":
        keys = {
            "transformer_list",
            "n_jobs",
            "transformer_weights",
            "verbose",
            "verbose_feature_names_out",
        }
    else:
        keys = None
    params = {
        key: _restore_value(value)
        for key, value in saved_params.items()
        if "__" not in key and (keys is None or key in keys)
    }
    try:
        estimator = cls(**params)
    except Exception as exc:
        raise HistoricalConfigurationError(
            f"Cannot reconstruct recorded estimator {path!r} with persisted parameters: {exc}"
        ) from exc
    if not is_estimator_instance(estimator):
        raise HistoricalConfigurationError(
            f"Recorded learner {path!r} is not a scikit-learn-compatible estimator."
        )
    return estimator


def recorded_learner_factory(
    root: Path, config_id: str, expected_name: str, task: str = "classification"
):
    from .configs_sweep import _learner_fingerprint
    from .learners import _learner_factory

    row = recorded_configuration(root, config_id)
    if str(row.get("learner")) != str(expected_name):
        raise HistoricalConfigurationError(
            f"Configuration {config_id!r} records learner {row.get('learner')!r}, not {expected_name!r}."
        )
    path = row.get("learner_class")
    params = row.get("learner_params")
    fingerprint = row.get("learner_fingerprint")
    if not path or not params or not fingerprint:
        raise HistoricalConfigurationError(
            f"Configuration {config_id!r} does not contain a complete persisted learner specification."
        )
    try:
        saved_params = json.loads(params) if isinstance(params, str) else params
        estimator = _restore_estimator(path, saved_params)
    except (ValueError, TypeError) as exc:
        raise HistoricalConfigurationError(
            f"Invalid stored learner parameters for {config_id!r}: {exc}"
        ) from exc
    actual = _learner_fingerprint((expected_name, estimator))
    if actual != str(fingerprint):
        raise HistoricalConfigurationError(
            f"Learner {expected_name!r} for historical configuration {config_id!r} "
            "cannot be reconstructed exactly with the installed dependency versions. "
            f"Saved fingerprint {fingerprint!r}, reconstructed {actual!r}."
        )
    return _learner_factory((expected_name, estimator), task=task)[1]


def recorded_transformation_spec(root: Path, config_id: str, expected_name: str):
    from .configs_sweep import _transformation_fingerprint
    from .transformations import PrevalenceFilter, Transform

    row = recorded_configuration(root, config_id)
    if str(row.get("count_transformation")) != str(expected_name):
        raise HistoricalConfigurationError(
            f"Configuration {config_id!r} records transformation {row.get('count_transformation')!r}, not {expected_name!r}."
        )
    expected = row.get("transformation_fingerprint")
    if not expected:
        raise HistoricalConfigurationError(
            f"Configuration {config_id!r} has no transformation fingerprint."
        )
    if _transformation_fingerprint(expected_name, None) == expected:
        return None
    name = str(expected_name)
    filter_value = None
    if "|filter=" in name:
        name, suffix = name.rsplit("|filter=", 1)
        try:
            options = dict(item.split("=", 1) for item in suffix.split(","))
            filter_value = PrevalenceFilter(
                float(options["prevalence"]), float(options["detection"])
            )
        except (ValueError, KeyError) as exc:
            raise HistoricalConfigurationError(
                f"Historical feature filter {suffix!r} cannot be reconstructed."
            ) from exc
    try:
        spec = Transform(name, feature_filter=filter_value)
    except (ValueError, KeyError, TypeError) as exc:
        raise HistoricalConfigurationError(
            f"Historical transformation {expected_name!r} requires a specification "
            "that was not persisted in the database."
        ) from exc
    actual = _transformation_fingerprint(expected_name, spec)
    if actual != expected:
        raise HistoricalConfigurationError(
            f"Historical transformation {expected_name!r} for {config_id!r} "
            "has parameters or implementation identity that cannot be recovered from the "
            "stored name and fingerprint. It will not be silently substituted."
        )
    return spec
