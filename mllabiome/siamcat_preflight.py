from __future__ import annotations

import argparse
import sys

import numpy as np

from .siamcat import SIAMCATClassifier
from .siamcat_runtime import ensure_siamcat_runtime, diagnose_siamcat_compiled_packages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the native SIAMCAT R backend")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Install/validate compatible R and SIAMCAT",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fit a tiny real R model and predict an independent holdout",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Isolate failures while loading installed native R packages",
    )
    parser.add_argument(
        "--runtime", choices=("auto", "system", "managed"), default="auto"
    )
    parser.add_argument(
        "--rscript", default=None, help="Path to a compatible Rscript 4.6.x"
    )
    args = parser.parse_args(argv)

    estimator = SIAMCATClassifier(runtime=args.runtime, rscript=args.rscript, verbose=1)
    script = estimator._r_runner_resource()
    if not script.is_file():
        print(
            "ERROR: mllabiome/_r/siamcat.R is not packaged with mllabiome",
            file=sys.stderr,
        )
        return 1
    print("OK: SIAMCAT R runner is present", flush=True)
    if args.diagnose:
        return (
            0
            if diagnose_siamcat_compiled_packages(
                runtime=args.runtime, rscript=args.rscript
            )
            else 1
        )
    if not (args.prepare or args.smoke):
        print(
            "Run with --prepare to validate R and SIAMCAT; --smoke to test fitting",
            flush=True,
        )
        return 0

    runtime = ensure_siamcat_runtime(runtime=args.runtime, rscript=args.rscript)
    print(
        f"OK: R {runtime.r_version}, SIAMCAT {runtime.siamcat_version}, Rscript={runtime.rscript}",
        flush=True,
    )
    if not args.smoke:
        return 0
    rng = np.random.default_rng(42)
    y = np.tile(np.array([0, 1], dtype=np.int64), 24)
    train = rng.gamma(shape=2.0, scale=1.0, size=(48, 24))
    train[:, 0] += 5 * y
    train[:, 1] += 3 * (1 - y)
    train /= train.sum(axis=1, keepdims=True)
    holdout = rng.gamma(shape=2.0, scale=1.0, size=(8, 24))
    holdout[:, 0] += 5 * np.tile([0, 1], 4)
    holdout /= holdout.sum(axis=1, keepdims=True)
    print("Starting native SIAMCAT fit on 48 synthetic training samples", flush=True)
    try:
        estimator.fit(train, y)
        proba = estimator.predict_proba(holdout)
        if (
            proba.shape != (8, 2)
            or not np.all(np.isfinite(proba))
            or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)
        ):
            raise RuntimeError(f"Unexpected SIAMCAT probability output: {proba}")
        print("OK: native SIAMCAT fit + holdout probabilities", flush=True)
        return 0
    finally:
        estimator.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
