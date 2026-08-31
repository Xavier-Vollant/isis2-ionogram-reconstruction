"""Tests for structural and numeric CDF comparison reporting."""

import json

import numpy as np

from isis_research.nasa.cdf_compare import compare_cdf_content
from isis_research.nasa.model_cdf import export_model_cdf, header_from_cdf


def test_compare_reports_structure_and_content_differences(tmp_path):
    sample = "data/samples/i2_av_ott_1975076233748_v01.cdf"
    header = header_from_cdf(sample)
    header["epoch"] = header["epoch"][:4]
    model = tmp_path / "prediction.npz"
    np.savez(
        model,
        prediction=np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4),
        frequency_mhz=np.linspace(0.25, 9.0, 4),
        virtual_height_km=np.linspace(0.0, 300.0, 3),
        meta_json=json.dumps({"orientation": "height,frequency"}),
    )
    generated = tmp_path / "model.cdf"
    export_model_cdf(model, header, generated)

    comparison = compare_cdf_content(sample, generated)
    assert "ampl" in comparison["shared_variables"]
    assert "valid_mask" in comparison["only_in_model"]
    assert "FINALISIS_SCHEMA" in comparison["global_attributes_only_in_model"]
    ampl = next(field for field in comparison["fields"] if field["name"] == "ampl")
    assert ampl["same_shape"] is False
