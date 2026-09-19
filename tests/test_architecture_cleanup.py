import importlib.util


def test_obsolete_renamed_modules_are_absent():
    assert importlib.util.find_spec("mllabiome.multiview_sweep") is None
    assert importlib.util.find_spec("mllabiome.views") is None
    assert importlib.util.find_spec("mllabiome.report_server") is None
