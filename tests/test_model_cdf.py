"""Tests for model-derived CDF export and metadata contracts."""

import json

import cdflib
import numpy as np
import pytest

from isis_research.nasa.model_cdf import (
    export_model_cdf,
    header_from_csa,
    read_model_output,
)


def _header(scan_count):
    return {
        "satellite_number": 4,
        "station_id": 50,
        "year": 75,
        "doy": 76,
        "hr": 23,
        "min": 37,
        "sec": 48.135,
        "satellite_height_km": 1408.5424,
        "geographic_latitude_deg": 43.98613,
        "geographic_longitude_deg": -104.75946,
        "FH": 0.8534,
        "DIP": 70.0,
        "CHI": 75.0,
        "L": 3.4116,
        "GMLAT": 52.9828,
        "GMLONG": -44.2980,
        "INV_LAT": 57.2209,
        "epoch": (6.23315507e13 + np.arange(scan_count) * 100.0).tolist(),
        "frequency_marker_times_ms": np.linspace(3000.0, 22000.0, 22).tolist(),
        "frequency_markers_mhz": np.linspace(0.1, 9.0, 22).tolist(),
    }


def test_model_cdf_round_trip(tmp_path):
    prediction = np.arange(12, dtype=np.float32).reshape(3, 4) / 11.0
    mask = np.ones_like(prediction, dtype=bool)
    mask[0, 0] = False
    model = tmp_path / "prediction.npz"
    np.savez(
        model,
        prediction=prediction,
        frequency_mhz=np.linspace(0.25, 9.0, 4),
        virtual_height_km=np.linspace(0.0, 300.0, 3),
        valid_mask=mask,
        meta_json=json.dumps({"orientation": "height,frequency", "model": "test"}),
    )
    output = tmp_path / "prediction.cdf"
    values, _ = export_model_cdf(model, _header(4), output)
    assert values["ampl"].shape == (4, 3)
    assert values["ampl"][0, 0] == 0
    assert values["ampl"][-1, -1] == 255
    assert values["swept_start"] == 1

    cdf = cdflib.CDF(str(output))
    assert cdf.varget("ampl").dtype == np.uint8
    assert cdf.varget("valid_mask")[0, 0] == 0
    assert cdf.varget("freq").shape == (4,)
    assert cdf.varget("v_height").shape == (3,)
    assert int(cdf.varget("vh_num")) == 4
    assert int(cdf.varget("f_num")) == 3
    assert "FINALISIS_PROVENANCE" in cdf.globalattsget()
    assert float(cdf.varget("geo_coord")[2]) == pytest.approx(1408.5424)


def test_model_cdf_requires_physical_header(tmp_path):
    model = tmp_path / "prediction.npz"
    np.savez(
        model,
        prediction=np.ones((3, 4), dtype=np.float32),
        frequency_mhz=np.linspace(0.25, 9.0, 4),
        virtual_height_km=np.linspace(0.0, 300.0, 3),
    )
    with pytest.raises(ValueError, match="pass header is missing"):
        export_model_cdf(model, {}, tmp_path / "prediction.cdf")


def test_exported_cdf_matches_the_model_grid(tmp_path):
    prediction = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    model = tmp_path / "prediction.npz"
    np.savez(
        model,
        prediction=prediction,
        frequency_mhz=np.linspace(0.25, 9.0, 4),
        virtual_height_km=np.linspace(0.0, 300.0, 3),
    )
    header = _header(4)
    output = tmp_path / "prediction.cdf"
    values, _ = export_model_cdf(model, header, output)
    cdf = cdflib.CDF(str(output))
    np.testing.assert_array_equal(cdf.varget("ampl"), values["ampl"])
    np.testing.assert_allclose(cdf.varget("freq"), values["freq"])
    np.testing.assert_allclose(cdf.varget("v_height"), values["v_height"])


def test_export_contains_the_checked_in_nasa_variable_set(tmp_path):
    model = tmp_path / "prediction.npz"
    np.savez(
        model,
        prediction=np.ones((3, 4), dtype=np.float32),
        frequency_mhz=np.linspace(0.25, 9.0, 4),
        virtual_height_km=np.linspace(0.0, 300.0, 3),
    )
    output = tmp_path / "prediction.cdf"
    export_model_cdf(model, _header(4), output)
    sample = cdflib.CDF("data/samples/i2_av_ott_1975076233748_v01.cdf")
    exported = cdflib.CDF(str(output))
    assert set(sample.cdf_info().zVariables).issubset(
        set(exported.cdf_info().zVariables)
    )


def test_csa_header_marks_unavailable_metadata_without_nasa_values(tmp_path):
    frequency = np.linspace(1.0, 9.0, 4)
    height = np.linspace(0.0, 300.0, 3)
    header = header_from_csa(
        "i2_av_res_1975076233748_v01",
        "",
        frequency,
        height,
    )
    model = tmp_path / "prediction.npz"
    np.savez(
        model,
        prediction=np.ones((3, 4), dtype=np.float32),
        frequency_mhz=frequency,
        virtual_height_km=height,
    )
    output = tmp_path / "prediction.cdf"
    export_model_cdf(model, header, output)

    cdf = cdflib.CDF(str(output))
    assert int(cdf.varget("satellite")) == -1
    assert np.all(cdf.varget("geo_coord") == -1.0)
    assert np.all(cdf.varget("Time_mark") == -1e31)
    np.testing.assert_allclose(cdf.varget("freq_mark"), np.linspace(1.0, 9.0, 22))
    provenance = json.loads(cdf.globalattsget()["FINALISIS_PROVENANCE"][0])
    assert provenance["pass_metadata"] == "csa_scan_only"
    assert provenance["csa_station"] == "RES"
    assert provenance["freq"] == "csa_artifact_axis"
    assert provenance["Epoch"] == "csa_pair_name"
    assert "satellite_number" in provenance["unknown_pass_fields"]


def test_model_output_rejects_nonfinite_prediction(tmp_path):
    model = tmp_path / "bad_prediction.npz"
    np.savez(
        model,
        prediction=np.array([[0.0, np.nan], [0.5, 1.0]]),
        frequency_mhz=np.array([1.0, 2.0]),
        virtual_height_km=np.array([0.0, 100.0]),
    )
    with pytest.raises(ValueError, match="finite two-dimensional"):
        read_model_output(model)


def test_model_output_rejects_invalid_metadata_json(tmp_path):
    model = tmp_path / "bad_metadata.npz"
    np.savez(
        model,
        prediction=np.zeros((2, 2)),
        frequency_mhz=np.array([1.0, 2.0]),
        virtual_height_km=np.array([0.0, 100.0]),
        meta_json=np.array(7),
    )
    with pytest.raises(ValueError, match="meta_json"):
        read_model_output(model)


def test_csa_header_rejects_a_name_without_observation_time():
    with pytest.raises(ValueError, match="must end with"):
        header_from_csa("not-an-observation", "KSH", [1.0, 2.0], [0.0, 100.0])
