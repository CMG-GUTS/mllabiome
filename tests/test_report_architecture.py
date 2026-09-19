from pathlib import Path

import numpy as np

from mllabiome import report


def test_report_numeric_display_uses_three_decimals_without_changing_integers():
    assert report._html_number(0.2222222222222222) == "0.222"
    assert report._html_number(-0.05204) == "-0.052"
    assert report._html_number(32) == "32"
    assert report._html_number(np.nan) == ""
    assert report._format_p_value(0.0002) == "<0.001"
    assert report._format_p_value(0.01234) == "0.012"


def test_favicon_is_embedded_in_html_data_uri():
    href = report._favicon_href()
    assert href.startswith("data:image/svg+xml;base64,")
    assert "http://" not in href
    assert "https://" not in href


def test_svg_figure_is_inlined_and_portable(tmp_path):
    source = tmp_path / "feature_support.svg"
    source.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><text>portable</text></svg>',
        encoding="utf-8",
    )

    rendered = report._fig(source, tmp_path, "Global explanations")

    assert "<svg" in rendered
    assert "portable" in rendered
    assert "src=" not in rendered
    assert str(source) not in rendered


def test_legacy_png_fallback_is_embedded_as_data_uri(tmp_path):
    stem = tmp_path / "legacy"
    stem.with_suffix(".png").write_bytes(b"png-bytes")

    rendered = report._fig(stem, tmp_path, "Legacy figure")

    assert 'src="data:image/png;base64,' in rendered
    assert str(stem.with_suffix(".png")) not in rendered
