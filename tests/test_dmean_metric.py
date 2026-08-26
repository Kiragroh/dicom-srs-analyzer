from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comparison import METRIC_COLS  # noqa: E402
import metrics  # noqa: E402


def test_dmean_is_volume_weighted_and_exported_for_comparison() -> None:
    histogram = {10.0: 1.0, 20.0: 3.0}

    dose_mean_from_histogram = getattr(metrics, "dose_mean_from_histogram", None)
    assert dose_mean_from_histogram is not None, "Dmean histogram calculation is missing"
    assert dose_mean_from_histogram(histogram) == pytest.approx(17.5)

    frame = metrics.results_to_dataframe(
        [metrics.MetricResult(patient_folder="case", structure_name="PTV01", Dmean=17.5)]
    )
    assert frame.loc[0, "Dmean_Gy"] == pytest.approx(17.5)
    assert ("Dmean_Gy", "Dmean (Gy)") in METRIC_COLS


def test_idl_approximation_is_exported_for_comparison() -> None:
    idl_approx_percent = getattr(metrics, "idl_approx_percent", None)
    assert idl_approx_percent is not None, "IDL approximation is missing"
    assert idl_approx_percent(d98=18.0, d2=24.0) == pytest.approx(75.0)

    frame = metrics.results_to_dataframe(
        [metrics.MetricResult(patient_folder="case", structure_name="PTV01", IDLApprox=75.0)]
    )
    assert frame.loc[0, "IDLApprox_pct"] == pytest.approx(75.0)
    assert ("IDLApprox_pct", "IDL approx. (%)") in METRIC_COLS
