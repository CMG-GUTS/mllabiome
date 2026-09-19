from __future__ import annotations

import numpy as np
import pandas as pd


def top_k_rank_support(values, top_k: int) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.notna()
    count = int(valid.sum())
    if count == 0:
        return out
    ranks = numeric.loc[valid].rank(ascending=False, method="average")
    k = max(1, min(int(top_k), count))
    out.loc[valid] = (1.0 - (ranks - 1.0) / float(k)).clip(lower=0.0, upper=1.0)
    return out
