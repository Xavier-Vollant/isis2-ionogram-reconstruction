from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.pipeline import run_scan


def test_model_name_selects_registered_checkpoint(monkeypatch, tmp_path):
    selected = tmp_path / "hybrid.pt"
    custom = tmp_path / "custom.pt"
    monkeypatch.setattr(run_scan, "candidate_checkpoint", lambda name: selected)

    assert run_scan.resolve_checkpoint(custom, "hybrid_unet") == selected
    assert run_scan.resolve_checkpoint(custom) == custom


def test_rejected_scan_with_diagnostics_keeps_inspection_output(monkeypatch, tmp_path):
    film = tmp_path / "scan.png"
    profile = tmp_path / "profile.json"
    output = tmp_path / "output"
    film.write_bytes(b"film")
    profile.write_text("{}")
    calls = []

    monkeypatch.setattr(
        run_scan,
        "standardize",
        lambda path, loaded_profile, destination: {
            "status": "not_usable",
            "reason": "insufficient marker support",
        },
    )
    monkeypatch.setattr(
        run_scan,
        "_write_diagnostics",
        lambda path, destination, loaded_profile: calls.append(destination),
    )

    with pytest.raises(ValueError, match="insufficient marker support"):
        run_scan.run_scan(film, output, profile_path=profile, diagnostics=True)

    assert calls == [output / "diagnostics"]


def test_run_scan_connects_the_three_product_stages(tmp_path, monkeypatch):
    film = tmp_path / "scan.png"
    output = tmp_path / "output"
    profile = tmp_path / "profile.json"
    checkpoint = tmp_path / "model.pt"
    profile.write_text("{}")
    checkpoint.write_bytes(b"checkpoint")

    calls = []

    def fake_standardize(path, loaded_profile, destination):
        calls.append("standardize")
        artifact = destination / "usable" / "scan.npz"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"artifact")
        return {"status": "usable", "artifact": "usable/scan.npz"}

    def fake_infer(artifact, model, destination):
        calls.append("infer")
        Path(destination).write_bytes(b"prediction")

    class Scan:
        intensity = np.zeros((2, 2))
        frequency_mhz = np.array([1.0, 2.0])
        virtual_height_km = np.array([0.0, 100.0])
        meta = {"status": "usable"}

    def fake_read(path):
        calls.append("read")
        return Scan()

    def fake_header(pair_name, station, frequency, height):
        calls.append("header")
        return {"pair_name": pair_name}

    def fake_export(prediction, header, destination):
        calls.append("export")
        Path(destination).write_bytes(b"cdf")
        return {"ampl": np.zeros((1, 1))}, {"source": "test"}

    monkeypatch.setattr(run_scan, "standardize", fake_standardize)
    monkeypatch.setattr(run_scan, "infer", fake_infer)
    monkeypatch.setattr(run_scan.ionogram, "read_validated", fake_read)
    monkeypatch.setattr(run_scan, "header_from_csa", fake_header)
    monkeypatch.setattr(run_scan, "export_model_cdf", fake_export)

    result = run_scan.run_scan(
        film,
        output,
        checkpoint=checkpoint,
        profile_path=profile,
        pair_name="scan_1975076233748",
    )

    assert calls == ["standardize", "infer", "read", "header", "export"]
    assert result["status"] == "usable"
    assert result["film"] == "scan.png"
    assert result["artifact"] == "usable/scan.npz"
    assert result["prediction"] == "scan_prediction.npz"
    assert result["cdf"] == "scan_model.cdf"
    assert (output / "scan_prediction.npz").is_file()
    assert (output / "scan_model.cdf").is_file()
    assert (output / "summary.json").is_file()


def test_diagnostics_writes_static_structure_and_warp_products(tmp_path):
    film = tmp_path / "scan.png"
    image = np.full((120, 200), 180, dtype=np.uint8)
    image[:, [30, 60, 90, 120, 150, 180]] = 20
    image[[20, 40, 60, 80, 100], :] = 30
    Image.fromarray(image, mode="L").save(film)

    group = {
        "sample_count": 1,
        "frequency": {
            "frequencies_mhz": [1, 2, 3, 4, 5, 6],
            "position_fraction": [0.15, 0.30, 0.45, 0.60, 0.75, 0.90],
        },
        "height": {
            "px_per_km": {"median": 0.1},
            "top_offset_px": {"median": 2.0},
            "ruling_spacing_px": {"median": 20.0},
            "km_per_ruling": {"median": 200.0},
        },
    }
    profile = {
        "source": {"min_profile_samples": 1, "min_fallback_samples": 1},
        "profiles": {},
        "format_fallbacks": {"narrow": group},
    }

    run_scan._write_diagnostics(film, tmp_path / "diagnostics", profile)

    expected = {
        "structure_overlay.png",
        "structure.json",
        "frequency.json",
        "height.json",
        "warp.json",
        "warped.png",
        "valid_mask.png",
    }
    assert {path.name for path in (tmp_path / "diagnostics").iterdir()} == expected
