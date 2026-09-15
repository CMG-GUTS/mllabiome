from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .siamcat_runtime import ensure_siamcat_runtime


def _feature_names_from_X(X: Any) -> list[str]:
    if hasattr(X, "columns"):
        raw = [str(c) for c in list(X.columns)]
    else:
        arr = np.asarray(X)
        n_features = 1 if arr.ndim == 1 else int(arr.shape[1])
        raw = [f"feature_{i:06d}" for i in range(n_features)]
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in raw:
        clean = name if name else "feature"
        count = seen.get(clean, 0)
        seen[clean] = count + 1
        out.append(clean if count == 0 else f"{clean}__dup{count}")
    return out


def _as_2d_float(X: Any) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        arr = X.to_numpy(dtype=float)
    else:
        arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(
            f"SIAMCATClassifier expects a 2D feature matrix; got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("SIAMCATClassifier received NaN or infinite feature values.")
    return arr


class SIAMCATClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        method: str = "lasso",
        filter_method: str = "abundance",
        filter_cutoff: float = 0.001,
        normalization: str = "log.std",
        num_folds: int = 2,
        num_resample: int = 1,
        measure: str = "classif.acc",
        grid_size: int = 11,
        min_nonzero: int = 5,
        perform_fs: bool = False,
        no_features: int = 100,
        fs_method: str = "AUC",
        fs_direction: str = "absolute",
        random_state: int = 42,
        verbose: int = 0,
        runtime: str = "auto",
        rscript: str | None = None,
    ):
        self.method = method
        self.filter_method = filter_method
        self.filter_cutoff = filter_cutoff
        self.normalization = normalization
        self.num_folds = num_folds
        self.num_resample = num_resample
        self.measure = measure
        self.grid_size = grid_size
        self.min_nonzero = min_nonzero
        self.perform_fs = perform_fs
        self.no_features = no_features
        self.fs_method = fs_method
        self.fs_direction = fs_direction
        self.random_state = random_state
        self.verbose = verbose
        self.runtime = runtime
        self.rscript = rscript

    def _runtime_env(self) -> tuple[Path, dict[str, str]]:
        runtime = ensure_siamcat_runtime(runtime=self.runtime, rscript=self.rscript)
        env = os.environ.copy()
        env["R_LIBS_USER"] = str(runtime.library)
        return runtime.rscript, env

    @staticmethod
    def _r_runner_resource():
        return files("mllabiome").joinpath("_r", "siamcat.R")

    def _write_features(self, path: Path, X: Any, *, fitted: bool) -> np.ndarray:
        arr = _as_2d_float(X)
        if fitted:
            if arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"SIAMCATClassifier was fitted with {self.n_features_in_} features but received {arr.shape[1]}."
                )
            names = list(self.feature_names_in_)
        else:
            names = _feature_names_from_X(X)
        sample_ids = [f"sample_{i:08d}" for i in range(arr.shape[0])]
        frame = pd.DataFrame(arr, index=sample_ids, columns=names)
        frame.index.name = "sample_id"
        frame.to_csv(path, sep="\t")
        return arr

    @staticmethod
    def _write_binary_labels(path: Path, y_binary: np.ndarray) -> None:
        sample_ids = [f"sample_{i:08d}" for i in range(len(y_binary))]
        frame = pd.DataFrame(
            {"label": np.asarray(y_binary, dtype=int)}, index=sample_ids
        )
        frame.index.name = "sample_id"
        frame.to_csv(path, sep="\t")

    def _run_r(self, args: list[str]) -> None:
        rscript, env = self._runtime_env()
        with as_file(self._r_runner_resource()) as runner:
            cmd = [str(rscript), "--vanilla", str(runner), *args]
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
            except subprocess.CalledProcessError as exc:
                details = (exc.stderr or exc.stdout or "").strip()
                raise RuntimeError(f"SIAMCAT backend failed.\n{details}") from exc
            if self.verbose >= 2 and result.stdout.strip():
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")

    def _fit_binary_model(
        self, X_path: Path, y_binary: np.ndarray, model_path: Path
    ) -> None:
        labels_path = model_path.with_suffix(".labels.tsv")
        self._write_binary_labels(labels_path, y_binary)
        min_class = int(np.bincount(y_binary, minlength=2).min())
        folds = min(int(self.num_folds), min_class)
        if folds < 2:
            raise ValueError(
                "SIAMCAT requires at least two training samples in each binary class."
            )
        self._run_r(
            [
                "fit",
                str(X_path),
                str(labels_path),
                str(model_path),
                str(self.method),
                str(self.filter_method),
                repr(float(self.filter_cutoff)),
                str(self.normalization),
                str(folds),
                str(max(1, int(self.num_resample))),
                str(self.measure),
                str(int(self.grid_size)),
                str(int(self.min_nonzero)),
                "1" if self.perform_fs else "0",
                str(int(self.no_features)),
                str(self.fs_method),
                str(self.fs_direction),
                str(int(self.random_state)),
                str(int(self.verbose)),
            ]
        )
        labels_path.unlink(missing_ok=True)

    def fit(self, X: Any, y: Any):
        arr = _as_2d_float(X)
        y_arr = np.asarray(y)
        if y_arr.ndim != 1 or len(y_arr) != arr.shape[0]:
            raise ValueError("X and y have incompatible shapes for SIAMCATClassifier.")
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError("SIAMCATClassifier requires at least two classes.")
        self.n_features_in_ = int(arr.shape[1])
        self.feature_names_in_ = np.asarray(_feature_names_from_X(X), dtype=object)
        self._workdir_ = Path(tempfile.mkdtemp(prefix="mllabiome-siamcat-"))
        X_path = self._workdir_ / "train.tsv"
        self._write_features(X_path, X, fitted=True)
        self._model_paths_: list[Path] = []
        target_classes = (
            [self.classes_[-1]] if len(self.classes_) == 2 else list(self.classes_)
        )
        for index, klass in enumerate(target_classes):
            y_binary = (y_arr == klass).astype(int)
            model_path = self._workdir_ / f"model_{index:03d}.rds"
            self._fit_binary_model(X_path, y_binary, model_path)
            self._model_paths_.append(model_path)
        self._binary_target_classes_ = np.asarray(target_classes, dtype=object)
        return self

    def _predict_binary_score(
        self, model_path: Path, X_path: Path, index: int
    ) -> np.ndarray:
        output_path = self._workdir_ / f"prediction_{index:03d}.tsv"
        self._run_r(
            [
                "predict",
                str(model_path),
                str(X_path),
                str(output_path),
                str(int(self.verbose)),
            ]
        )
        out = pd.read_csv(output_path, sep="\t")
        if "score" not in out.columns:
            raise RuntimeError(
                "SIAMCAT prediction output did not contain a 'score' column."
            )
        score = pd.to_numeric(out["score"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(score)):
            raise RuntimeError("SIAMCAT returned non-finite prediction scores.")
        return score

    def decision_function(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_model_paths_"):
            raise RuntimeError("SIAMCATClassifier has not been fitted.")
        arr = _as_2d_float(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"SIAMCATClassifier was fitted with {self.n_features_in_} features but received {arr.shape[1]}."
            )
        X_path = self._workdir_ / "holdout.tsv"
        self._write_features(X_path, X, fitted=True)
        scores = [
            self._predict_binary_score(path, X_path, i)
            for i, path in enumerate(self._model_paths_)
        ]
        if len(self.classes_) == 2:
            return np.asarray(scores[0], dtype=float)
        return np.column_stack(scores).astype(float)

    def predict(self, X: Any) -> np.ndarray:
        score = np.asarray(self.decision_function(X), dtype=float)
        if len(self.classes_) == 2:
            if np.all((score >= 0.0) & (score <= 1.0)):
                positive = score >= 0.5
            else:
                positive = score >= 0.0
            return np.where(positive, self.classes_[-1], self.classes_[0])
        return np.asarray(self.classes_)[np.argmax(score, axis=1)]

    def cleanup(self) -> None:
        workdir = getattr(self, "_workdir_", None)
        if workdir is not None:
            shutil.rmtree(workdir)
            self._workdir_ = None


__all__ = ["SIAMCATClassifier"]
