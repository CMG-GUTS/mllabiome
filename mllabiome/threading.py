from __future__ import annotations

import os
from collections.abc import Mapping

_THREAD_ENV_DEFAULTS: dict[str, str] = {
    "POLARS_MAX_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def configure_thread_limits(defaults: Mapping[str, str] | None = None) -> None:
    for key, value in dict(defaults or _THREAD_ENV_DEFAULTS).items():
        os.environ.setdefault(str(key), str(value))
