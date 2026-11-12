from __future__ import annotations

import importlib.util
from pathlib import Path


def test_advanced_ensemble_standalone_import():
    path = Path(__file__).resolve().parents[1] / "mllabiome" / "advanced_ensemble.py"
    spec = importlib.util.spec_from_file_location("advanced_ensemble", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "bayesian_model_averaging" in module.ADVANCED_AGGREGATIONS
    assert callable(module.fit_probabilistic)
    assert callable(module.predict_probabilistic)
