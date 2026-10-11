from __future__ import annotations

import os

import pytest

from mllabiome import runtime as rt


def test_darwin_auto_is_conservative(monkeypatch):
    monkeypatch.setattr(rt.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(rt, "_physical_cpu_count", lambda: 18)
    monkeypatch.setattr(rt, "_logical_cpu_count", lambda: 18)
    monkeypatch.setattr(rt, "available_memory_bytes", lambda: 100 * 1024**3)
    monkeypatch.delenv("MLLABIOME_MAX_WORKERS", raising=False)
    plan = rt.resolve_execution_plan("auto", 60)
    assert plan.workers == 4
    assert plan.threads_per_worker == 1
    plan = rt.resolve_execution_plan(8, 60)
    assert plan.workers == 8
    assert plan.threads_per_worker == 1


def test_linux_plan_unchanged(monkeypatch):
    monkeypatch.setattr(rt.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rt, "_physical_cpu_count", lambda: 18)
    monkeypatch.setattr(rt, "_logical_cpu_count", lambda: 18)
    monkeypatch.setattr(rt, "available_memory_bytes", lambda: 100 * 1024**3)
    monkeypatch.delenv("MLLABIOME_MAX_WORKERS", raising=False)
    plan = rt.resolve_execution_plan("auto", 60)
    assert (plan.workers, plan.threads_per_worker) == (8, 2)


def test_environment_worker_cap(monkeypatch):
    monkeypatch.setattr(rt.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(rt, "_physical_cpu_count", lambda: 18)
    monkeypatch.setattr(rt, "_logical_cpu_count", lambda: 18)
    monkeypatch.setattr(rt, "available_memory_bytes", lambda: 100 * 1024**3)
    monkeypatch.setenv("MLLABIOME_MAX_WORKERS", "1")
    assert rt.resolve_execution_plan("auto", 60).workers == 1
    monkeypatch.setenv("MLLABIOME_MAX_WORKERS", "bad")
    with pytest.raises(ValueError, match="MLLABIOME_MAX_WORKERS"):
        rt.resolve_execution_plan("auto", 60)
    monkeypatch.setenv("MLLABIOME_MAX_WORKERS", "0")
    with pytest.raises(ValueError, match="MLLABIOME_MAX_WORKERS"):
        rt.resolve_execution_plan("auto", 60)


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_boosters_are_single_threaded_on_mac(monkeypatch, name):
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        base = make_pipeline(StandardScaler(), LGBMClassifier(n_jobs=12))
        monkeypatch.setattr(rt.platform, "system", lambda: "Darwin")
        model = rt.configure_estimator_threads(base, 6)
        assert model.named_steps["lgbmclassifier"].get_params()["n_jobs"] == 1
    else:
        from xgboost import XGBClassifier

        monkeypatch.setattr(rt.platform, "system", lambda: "Darwin")
        model = rt.configure_estimator_threads(XGBClassifier(n_jobs=12), 6)
        assert model.get_params()["n_jobs"] == 1


def test_loky_launch_inherits_native_thread_limits(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "32")
    seen = []

    class ParallelMock:
        def __init__(self, **kwargs):
            seen.append(
                (os.environ.get("OMP_NUM_THREADS"), kwargs["inner_max_num_threads"])
            )

        def __call__(self, wave):
            return (fn(*args, **kwargs) for fn, args, kwargs in wave)

    monkeypatch.setattr(rt, "Parallel", ParallelMock)
    plan = rt.ExecutionPlan(18, 18, None, 2, 1, "loky")
    values = list(
        rt.iter_parallel_tasks([(sum, ([1, 2],), {}), (sum, ([3, 4],), {})], plan)
    )
    assert values == [3, 7]
    assert seen == [("1", 1)]
    assert os.environ["OMP_NUM_THREADS"] == "32"
