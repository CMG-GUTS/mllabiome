import numpy as np

from mllabiome._evaluation_protocols import LODO_PROTOCOLS, is_lodo_protocol
from mllabiome.configs_sweep import _lodo_feature_pair
from mllabiome.report import _performance_methodology_html
from mllabiome.statistics_common import _LODO_PROTOCOLS
from mllabiome.sweep_types import Evaluation


def test_hierarchical_lodo_is_classified_as_lodo_everywhere():
    assert "hierarchical_lodo" in LODO_PROTOCOLS
    assert "hierarchical_lodo" in _LODO_PROTOCOLS
    assert is_lodo_protocol("hierarchical_lodo")
    assert is_lodo_protocol("hierarchical-lodo")
    assert not is_lodo_protocol("nested_cv")


def test_hierarchical_lodo_uses_training_only_feature_mask():
    X = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 4.0],
            [0.0, 5.0, 6.0],
        ]
    )
    train_idx = np.asarray([0, 1], dtype=int)
    test_idx = np.asarray([2, 3], dtype=int)
    outputs = {
        protocol: _lodo_feature_pair(X, train_idx, test_idx, protocol)
        for protocol in ("lodo", "leave_one_dataset_out", "hierarchical_lodo")
    }
    expected_mask = np.asarray([True, False, False])
    for X_train, X_test, mask in outputs.values():
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(X_train, X[train_idx][:, expected_mask])
        np.testing.assert_array_equal(X_test, X[test_idx][:, expected_mask])


def test_hierarchical_lodo_uses_lodo_reporting_and_normalization():
    evaluation = Evaluation(protocol="hierarchical-lodo")
    assert evaluation.protocol == "hierarchical_lodo"
    html = _performance_methodology_html(evaluation.protocol, n_bootstrap=20)
    assert "held-out datasets" in html
    assert "resample held-out datasets" in html
