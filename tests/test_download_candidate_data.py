"""Tests for candidate download validation and Phase 1 freshness checks."""

import csv
import sys
from pathlib import Path

import numpy as np

DATASET_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "dataset"
sys.path.insert(0, str(DATASET_SCRIPTS))

import download_candidate_data as downloader


def candidate(predicted="I2_AV_RES_1972003024600"):
    return {
        "rank": "1",
        "csa_film_subdir": "B1-35-20 ISIS B D-409",
        "csa_image_filename": "Image0001.png",
        "csa_image_url": "https://example.test/image.png",
        "nasa_id": "RES_75N_265E/1972/header#0000",
        "nasa_frame_sync_utc": "1972-003T02:46:15.300000",
        "nasa_cdf_name_predicted": predicted,
        "nasa_station": "RES",
        "csa_station": "RES",
        "csa_timestamp_utc": "1972-01-03T02:46:15",
        "dt_seconds": "0.5",
        "position_km": "5.0",
    }


def test_find_cdf_prefers_exact_and_supports_two_second_fallback(monkeypatch):
    names = ["I2_AV_RES_1972003024600_V01.cdf"]
    monkeypatch.setattr(downloader, "list_day", lambda *_: names)

    assert downloader.find_cdf(candidate()) == (names[0], "exact")
    assert downloader.find_cdf(candidate("I2_AV_RES_1972003024601")) == (
        names[0],
        "nearest_1s",
    )


def test_process_downloads_validates_and_writes_review_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "CSA_DIR", tmp_path / "csa")
    monkeypatch.setattr(downloader, "CDF_DIR", tmp_path / "cdf")
    monkeypatch.setattr(downloader, "NASA_PNG", tmp_path / "review" / "nasa_png")
    monkeypatch.setattr(
        downloader, "list_day", lambda *_: ["I2_AV_RES_1972003024600_V01.cdf"]
    )

    sample_png = (
        Path(__file__).resolve().parents[1]
        / "data/raw/csa_verified_bur_1973077231124.png"
    ).read_bytes()
    monkeypatch.setattr(
        downloader,
        "fetch",
        lambda url: sample_png if url.endswith("image.png") else b"cdf-bytes",
    )
    monkeypatch.setattr(
        downloader,
        "read_ionogram",
        lambda path: {
            "amplitude": np.ones((2, 3), dtype=np.uint8),
            "height": np.array([0.0, 100.0]),
            "frequency": np.array([0.1, 1.0, 2.0]),
            "has_freq_axis": True,
            "signal_fraction": 1.0,
            "shape": (2, 3),
        },
    )

    def fake_render(ionogram, path, title):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"review-png")

    monkeypatch.setattr(downloader, "render", fake_render)

    result = downloader.process(candidate())

    assert result["csa_ok"] is True
    assert result["nasa_ok"] is True
    assert result["cdf_match_kind"] == "exact"
    assert result["nasa_cdf_file"] == "I2_AV_RES_1972003024600_V01.cdf"
    assert Path(result["csa_path"]).is_file()
    assert Path(result["nasa_cdf_path"]).is_file()
    assert Path(result["nasa_png"]).is_file()

    output = tmp_path / "review.csv"
    downloader.write_ranked([result], output, list(candidate()))
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["csa_file"] == Path(result["csa_path"]).name
    assert row["nasa_png_file"] == Path(result["nasa_png"]).name
    assert row["nasa_cdf_file"] == "I2_AV_RES_1972003024600_V01.cdf"
    assert row["has_freq_axis"] == "True"


def test_fresh_phase1_batch_does_not_require_historical_outputs(tmp_path, monkeypatch):
    import prepare_phase1_pairs_target as prepare

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    csa_dir = tmp_path / "data/raw/matches/csa_png"
    cdf_dir = tmp_path / "data/raw/matches/nasa_cdf"
    csa_dir.mkdir(parents=True)
    cdf_dir.mkdir(parents=True)
    (csa_dir / "film.png").write_bytes(b"png")
    (cdf_dir / "i2_av_res_1972003024600_v01.cdf").write_bytes(b"cdf")

    candidate_path = tmp_path / "candidate.csv"
    fields = [
        "rank",
        "nasa_png_file",
        "csa_file",
        "csa_size",
        "cdf_match_kind",
        "ambiguous",
        "has_freq_axis",
        "csa_station",
        "csa_film_subdir",
        "signal_fraction",
        "nasa_swept_freq_range",
    ]
    with candidate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "rank": "1",
                "nasa_png_file": "001_i2_av_res_1972003024600_v01.png",
                "csa_file": "film.png",
                "csa_size": "100x100",
                "cdf_match_kind": "exact",
                "ambiguous": "False",
                "has_freq_axis": "True",
                "csa_station": "RES",
                "csa_film_subdir": "REEL-1",
                "signal_fraction": "0.5",
                "nasa_swept_freq_range": "0.1 - 10 MHz",
            }
        )

    output = tmp_path / "pairs"
    prepare.main(
        [
            "--fresh-only",
            "--candidates",
            str(candidate_path),
            "--target",
            "1",
            "--out",
            str(output),
            "--existing",
            str(tmp_path / "missing_manifest.csv"),
            "--records",
            str(tmp_path / "missing_records.csv"),
            "--landmarks",
            str(tmp_path / "missing_landmarks"),
        ]
    )

    assert (output / "manifest.csv").is_file()
    assert (output / "train").is_dir() or (output / "held_out").is_dir()
