from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

import mllabiome
import mllabiome.configs_sweep as cs
from mllabiome.configs_sweep import Evaluation, QualificationGate, Sweep
from mllabiome.data import Data, Dataset


def _dataset():
    return Dataset(
        X_by_level={"all": np.asarray([[1.0], [2.0]], dtype=np.float32)},
        feature_names_by_level={"all": ["g__A"]},
        y=np.asarray([0, 1], dtype=int),
        sample_ids=["s1", "s2"],
        subject_ids=["p1", "p2"],
        metadata=pd.DataFrame({"sample_id": ["s1", "s2"]}),
        class_labels=["control", "case"],
        positive_class=1,
    )


def _sweep(root):
    return Sweep(
        data=Data(abundance_path="unused"),
        experiment_dir=root,
        resolutions=(("genus", ("genus",)),),
        count_transformations=("identity",),
        learners=(("dummy", DummyClassifier(strategy="prior")),),
        evaluation=Evaluation(
            protocol="nested_cv",
            outer_folds=2,
            inner_folds=2,
            repeats=1,
            random_state=17,
        ),
        gate=QualificationGate(enabled=False),
    )


def test_runtime_version_is_single_sourced():
    from mllabiome._version import __version__ as source_version

    assert mllabiome.__version__ == source_version


def test_pyproject_uses_dynamic_single_source_version():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text
    assert re.search(
        r'version\s*=\s*\{\s*attr\s*=\s*"mllabiome\._version\.__version__"\s*\}', text
    )


def test_software_provenance_has_source_runtime_and_dependency_identity():
    provenance = cs._software_provenance()
    assert provenance["mllabiome_source_version"] == mllabiome.__version__
    assert provenance["version_consistent"] is True
    assert len(provenance["source_tree_sha256"]) == 64
    int(provenance["source_tree_sha256"], 16)
    assert provenance["source_tree_fingerprint_algorithm"] == "sha256-package-source-v1"
    assert provenance["python_version"]
    assert provenance["python_implementation"]
    assert isinstance(provenance["dependencies"], dict)


def test_manifest_records_version_and_software_provenance(tmp_path):
    cs._write_manifest(tmp_path, _sweep(tmp_path), _dataset())
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["package_version"] == mllabiome.__version__
    assert (
        manifest["software_provenance"]["mllabiome_source_version"]
        == mllabiome.__version__
    )
    assert manifest["software_provenance"]["version_consistent"] is True
    assert manifest["dataset_fingerprint_algorithm"] == "sha256-model-input-v2"
