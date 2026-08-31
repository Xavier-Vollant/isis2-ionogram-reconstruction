"""End-to-end contract test: audit reads historical files; production rejects them."""

import json

import numpy as np
import pytest

from isis_research import ionogram
from isis_research.grids import load_film


def _grid(height_bins=32, frequency_bins=48, freq=(0.1, 9.5), height=(0.0, 3000.0)):
    frequency = np.linspace(freq[0], freq[1], frequency_bins)
    heights = np.linspace(height[0], height[1], height_bins)
    # A dark horizontal trace, so an orientation error is visible in the values.
    values = np.full((height_bins, frequency_bins), 0.9)
    values[height_bins // 3, :] = 0.1
    return (
        values,
        np.ones((height_bins, frequency_bins), dtype=bool),
        frequency,
        heights,
    )


def _meta(**overrides):
    return {
        "status": "usable",
        "route": "film_only",
        "confidence": 0.9,
        "source": {"csa_scan": "test.png"},
        "provenance": {"producer": "tests/test_ionogram_contract.py"},
        **overrides,
    }


def _write_legacy(path, layout, values, frequency, heights):
    """Write the pre-contract layouts exactly as their producers did."""
    if layout == "phase6":
        np.savez(
            path,
            warped_intensity=values,
            valid_mask=np.ones(values.shape, bool),
            frequency_mhz=frequency,
            virtual_height_km=heights,
        )
    elif layout == "standardized":  # standardize_scan.py, (frequency, height)
        np.savez(path, warped=values.T, freq_axis=frequency, v_height=heights)
    elif layout == "film_to_nasa":  # warp_film_to_nasa.py, (frequency, height)
        np.savez(
            path,
            warped_film=values.T,
            nasa_ampl=values.T,
            freq=frequency,
            v_height=heights,
        )
    return path


# --- the canonical round trip -------------------------------------------------


def test_write_then_read_returns_what_went_in(tmp_path):
    values, valid, frequency, heights = _grid()
    ionogram.write(
        tmp_path / "a.npz",
        values,
        valid,
        frequency,
        heights,
        **_meta(confidence=0.91, source={"csa_scan": "046_Image0184.png"}),
    )
    result = ionogram.read(tmp_path / "a.npz")
    assert ionogram.validate(result) == []
    assert np.allclose(result.intensity, values)
    assert result.meta["status"] == "usable"
    assert result.meta["route"] == "film_only"
    assert result.meta["source"]["csa_scan"] == "046_Image0184.png"
    assert result.meta["orientation"] == ionogram.ORIENTATION


def test_metadata_travels_inside_the_artifact(tmp_path):
    values, valid, frequency, heights = _grid()
    path = ionogram.write(
        tmp_path / "a.npz", values, valid, frequency, heights, **_meta(status="review")
    )
    with np.load(path, allow_pickle=False) as data:
        assert "meta_json" in data.files
        assert json.loads(str(data["meta_json"]))["status"] == "review"


def test_support_and_coverage_are_recorded(tmp_path):
    values, valid, frequency, heights = _grid()
    valid[:, :10] = False
    path = ionogram.write(
        tmp_path / "a.npz", values, valid, frequency, heights, **_meta()
    )
    meta = ionogram.read(path).meta
    assert meta["support"]["frequency_mhz"] == pytest.approx([0.1, 9.5])
    assert meta["coverage"] == pytest.approx(valid.mean())


# --- historical layouts remain auditable, never production-ready -------------


@pytest.mark.parametrize("layout", ["phase6", "standardized", "film_to_nasa"])
def test_legacy_layouts_are_auditable_but_rejected_in_production(tmp_path, layout):
    values, _, frequency, heights = _grid()
    path = _write_legacy(tmp_path / f"{layout}.npz", layout, values, frequency, heights)
    result = ionogram.read(path)
    assert result.intensity.shape == (len(heights), len(frequency))
    assert np.allclose(result.intensity, values), (
        "legacy orientation was not normalized"
    )
    assert result.meta["source_layout"] in {"warped_intensity", "warped", "warped_film"}
    assert ionogram.validate(result)
    with pytest.raises(ValueError, match="invalid ionogram artifact"):
        ionogram.read_validated(path)


@pytest.mark.parametrize("layout", ["standardized", "film_to_nasa"])
def test_legacy_missing_mask_is_flagged_not_silently_trusted(tmp_path, layout):
    values, _, frequency, heights = _grid()
    path = _write_legacy(tmp_path / f"{layout}.npz", layout, values, frequency, heights)
    result = ionogram.read(path)
    assert result.meta["valid_mask_source"] == "synthesized_absent_in_source_artifact"


def test_phase6_sidecar_metadata_is_available_for_audit_but_not_production(tmp_path):
    values, _, frequency, heights = _grid()
    path = _write_legacy(tmp_path / "s.npz", "phase6", values, frequency, heights)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "isis.csa_warp_result.v1",
                "status": "review",
                "valid_coverage": 0.83,
            }
        )
    )
    artifact = ionogram.read(path)
    meta = artifact.meta
    assert meta["status"] == "review"
    assert meta["valid_coverage"] == 0.83
    assert meta["metadata_source"] == "sidecar"
    assert ionogram.validate(artifact)
    with pytest.raises(ValueError, match="metadata must be embedded"):
        ionogram.read_validated(path)


def test_unknown_layout_is_rejected_loudly(tmp_path):
    np.savez(tmp_path / "x.npz", something_else=np.zeros((4, 4)))
    with pytest.raises(KeyError):
        ionogram.read(tmp_path / "x.npz")


# --- the breaks the contract exists to catch ----------------------------------


def test_square_transpose_is_caught_even_though_shapes_agree(tmp_path):
    """The silent one: a square grid stored the wrong way round.

    Axis lengths still match, so no shape check can see it. Only the declared
    orientation can, which is why it is a required field.
    """
    values, valid, frequency, heights = _grid(height_bins=40, frequency_bins=40)
    ionogram.write(tmp_path / "a.npz", values, valid, frequency, heights, **_meta())
    with np.load(tmp_path / "a.npz", allow_pickle=False) as data:
        payload = dict(data)
    payload["warped_intensity"] = payload["warped_intensity"].T
    payload["meta_json"] = json.dumps({"orientation": "frequency,height"})
    np.savez(tmp_path / "bad.npz", **payload)

    result = ionogram.read(tmp_path / "bad.npz")
    assert result.meta["orientation"] == "frequency,height"
    assert any("orientation" in problem for problem in ionogram.validate(result))
    with pytest.raises(ValueError, match="orientation"):
        ionogram.read_validated(tmp_path / "bad.npz")


def test_missing_embedded_metadata_is_rejected_by_production_reader(tmp_path):
    values, valid, frequency, heights = _grid()
    path = tmp_path / "missing_meta.npz"
    np.savez(
        path,
        warped_intensity=values,
        valid_mask=valid,
        frequency_mhz=frequency,
        virtual_height_km=heights,
    )
    result = ionogram.read(path)
    problems = ionogram.validate(result)
    assert any("schema" in problem for problem in problems)
    assert any("orientation" in problem for problem in problems)
    assert any("metadata must be embedded" in problem for problem in problems)
    with pytest.raises(ValueError, match="invalid ionogram artifact"):
        ionogram.read_validated(path)


@pytest.mark.parametrize(
    "field", ("status", "route", "confidence", "source", "provenance")
)
def test_required_metadata_is_auditable_but_rejected_in_production(tmp_path, field):
    values, valid, frequency, heights = _grid()
    metadata = _meta()
    del metadata[field]
    path = tmp_path / f"missing_{field}.npz"
    np.savez(
        path,
        warped_intensity=values,
        valid_mask=valid,
        frequency_mhz=frequency,
        virtual_height_km=heights,
        meta_json=json.dumps(
            {
                "schema": ionogram.SCHEMA,
                "orientation": ionogram.ORIENTATION,
                **metadata,
            }
        ),
    )

    artifact = ionogram.read(path)
    assert field not in artifact.meta
    with pytest.raises(ValueError, match=field):
        ionogram.read_validated(path)


def test_missing_mask_is_rejected_by_production_reader(tmp_path):
    values, _, frequency, heights = _grid()
    path = tmp_path / "missing_mask.npz"
    np.savez(
        path,
        warped_intensity=values,
        frequency_mhz=frequency,
        virtual_height_km=heights,
        meta_json=json.dumps(
            {"schema": ionogram.SCHEMA, "orientation": ionogram.ORIENTATION}
        ),
    )
    result = ionogram.read(path)
    assert result.meta["valid_mask_source"] == "synthesized_absent_in_source_artifact"
    assert any(
        "valid_mask must be stored" in problem for problem in ionogram.validate(result)
    )
    with pytest.raises(ValueError, match="valid_mask must be stored"):
        ionogram.read_validated(path)


def test_signal_occupancy_loader_rejects_a_legacy_artifact(tmp_path):
    values, _, frequency, heights = _grid()
    path = _write_legacy(
        tmp_path / "legacy.npz", "standardized", values, frequency, heights
    )
    with pytest.raises(ValueError, match="invalid ionogram artifact"):
        load_film(path)


def test_axis_length_mismatch_is_rejected():
    values, valid, frequency, heights = _grid(height_bins=32, frequency_bins=48)
    bad = ionogram.Ionogram(
        values, valid, frequency[:-5], heights, {"orientation": ionogram.ORIENTATION}
    )
    assert any("frequency axis" in problem for problem in ionogram.validate(bad))


def test_non_monotonic_axis_is_rejected():
    values, valid, frequency, heights = _grid()
    frequency = frequency.copy()
    frequency[3], frequency[4] = frequency[4], frequency[3]
    bad = ionogram.Ionogram(
        values, valid, frequency, heights, {"orientation": ionogram.ORIENTATION}
    )
    assert any("strictly increasing" in problem for problem in ionogram.validate(bad))


def test_lying_support_bounds_are_rejected():
    values, valid, frequency, heights = _grid()
    bad = ionogram.Ionogram(
        values,
        valid,
        frequency,
        heights,
        {
            "orientation": ionogram.ORIENTATION,
            "support": {"frequency_mhz": [0.1, 20.0]},
        },
    )
    assert any("support" in problem for problem in ionogram.validate(bad))


def test_write_refuses_a_non_conforming_artifact(tmp_path):
    values, valid, frequency, heights = _grid()
    with pytest.raises(ValueError, match="refusing to write"):
        ionogram.write(
            tmp_path / "a.npz", values, valid, frequency[:-5], heights, **_meta()
        )


def test_write_refuses_missing_required_metadata(tmp_path):
    values, valid, frequency, heights = _grid()
    with pytest.raises(ValueError, match="confidence"):
        ionogram.write(
            tmp_path / "a.npz",
            values,
            valid,
            frequency,
            heights,
            **_meta(confidence=None),
        )


def test_intensity_outside_unit_range_is_rejected():
    values, valid, frequency, heights = _grid()
    values = values.copy()
    values[0, 0] = 4.2
    bad = ionogram.Ionogram(
        values, valid, frequency, heights, {"orientation": ionogram.ORIENTATION}
    )
    assert any("outside [0, 1]" in problem for problem in ionogram.validate(bad))


# --- the support cases Sol asks for by name -----------------------------------


def test_partial_frequency_scan_keeps_its_real_support(tmp_path):
    """A scan that only swept 0.5-4.2 MHz must not claim the full band."""
    values, valid, frequency, heights = _grid(freq=(0.5, 4.2))
    path = ionogram.write(
        tmp_path / "partial.npz", values, valid, frequency, heights, **_meta()
    )
    result = ionogram.read(path)
    assert ionogram.validate(result) == []
    assert result.support["frequency_mhz"] == pytest.approx([0.5, 4.2])
    assert result.meta["support"]["frequency_mhz"] == pytest.approx([0.5, 4.2])


def test_twenty_megahertz_scan_is_representable(tmp_path):
    """The 20 MHz population must not be silently forced onto a 9.5 MHz grid."""
    values, valid, frequency, heights = _grid(frequency_bins=200, freq=(0.1, 20.0))
    path = ionogram.write(
        tmp_path / "wide.npz", values, valid, frequency, heights, **_meta()
    )
    result = ionogram.read(path)
    assert ionogram.validate(result) == []
    assert result.support["frequency_mhz"][1] == pytest.approx(20.0)
    assert result.intensity.shape[1] == 200


def test_partial_and_full_scans_are_distinguishable(tmp_path):
    """Two artifacts, same grid shape, different physical support."""
    narrow = ionogram.write(tmp_path / "n.npz", *_grid(freq=(0.5, 4.2)), **_meta())
    wide = ionogram.write(tmp_path / "w.npz", *_grid(freq=(0.1, 9.5)), **_meta())
    assert ionogram.read(narrow).support != ionogram.read(wide).support
