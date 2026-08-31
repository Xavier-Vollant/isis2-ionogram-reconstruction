"""Tests for CSA and NASA catalog parsing and normalization."""

import csv
import sys
from pathlib import Path

import pytest

DATASET_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "dataset"
sys.path.insert(0, str(DATASET_SCRIPTS))

import align_phase1_pairs as aligner
import build_csa_master as csa
import build_nasa_master as nasa


def test_csa_path_and_timestamp_normalization():
    assert csa.split_image_path(
        "b11_R014207871/B1-35-20 ISIS B D-409/Image0001.png"
    ) == ("b11_R014207871", "B1-35-20 ISIS B D-409", "Image0001.png")
    assert csa.path_key("B1-35-20 ISIS B D-409", "Image0001.png") == (
        "b1-35-20 isis b d-409",
        "image0001",
    )
    assert csa.parse_timestamp("1972-01-03 02:44:04") == (
        "1972-01-03T02:44:04",
        "ok",
        "1972",
        "3",
        "9844",
    )
    assert csa.parse_timestamp("1972-10-00 02:44:04")[1] == "invalid_day"


def test_csa_coordinate_and_trust_normalization():
    assert csa.parse_coord("69.7N") == pytest.approx(69.7)
    assert csa.parse_coord("18.9E") == pytest.approx(18.9)
    assert csa.parse_coord("69.7S") == pytest.approx(-69.7)
    assert csa.trust("ok", "True") == "ok"
    assert csa.trust("ok", "False") == "suspect_below_horizon"
    assert csa.trust("invalid_day", "True") == "suspect_invalid_day"


def test_nasa_parser_expands_one_header_into_ionograms(tmp_path):
    header = tmp_path / "1972003T024600_res_isis2_tops_hdr.asc"
    header.write_text(
        """Satellite Number: 4
Station Name: RES
Station Code: 75
Pass Number: 00042
Start Time: 72/01/03  (72003)  02:44:00
IONOGRAM HEADERS FOLLOW 1
IONOGRAMS: A4RES00306B01_03509_72003_024600.BIN
F4RES00306B01_03509_72003_024600.BIN
YR: 72
DAY: 003
HR: 02
MIN: 46
SEC: 15.3
GGLAT: 69.86
GGLONG: -61.42
HGT: 1371.
""",
        encoding="utf-8",
    )

    rows, parsed_header = nasa.parse_file(header, "RES_75N_265E", "1972")

    assert parsed_header["station_code_pass"] == "RES"
    assert len(rows) == 1
    assert rows[0]["nasa_id"].endswith("#0000")
    assert rows[0]["frame_sync_utc"] == "1972-003T02:46:15.300000"
    assert rows[0]["cdf_name_predicted"] == "I2_AV_RES_1972003024600"
    assert rows[0]["parse_status"] == "ok"


def test_nasa_acquire_retries_empty_cached_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(nasa, "RAW_DIR", tmp_path / "headers")
    target = ("RES_75N_265E", "1972", "empty.asc")
    path = nasa.cache_path(*target)
    path.parent.mkdir(parents=True)
    path.touch()
    calls = []

    def fake_download(item):
        calls.append(item)
        nasa.cache_path(*item).write_bytes(b"recovered")
        return True

    monkeypatch.setattr(nasa, "download_one", fake_download)

    assert nasa.acquire([target], workers=1, passes=1, cooldown=0) == []
    assert calls == [target]
    assert path.read_bytes() == b"recovered"


def test_csa_build_joins_local_inventory_fixtures(tmp_path, monkeypatch):
    raw = tmp_path / "csa"
    output = tmp_path / "processed"
    raw.mkdir()
    monkeypatch.setattr(csa, "RAW_DIR", raw)
    monkeypatch.setattr(csa, "OUT_DIR", output)

    with (raw / "result_master_ISIS2.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "File name",
                "Satellite number",
                "Latitude",
                "Longitude",
                "Ground station code",
                "Ground station number",
                "Ground station name",
                "Timestamp",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "File name": "batch/B1-35-20 ISIS B D-409/Image0001.png",
                "Satellite number": "4",
                "Latitude": "69.7N",
                "Longitude": "18.9E",
                "Ground station code": "KSH",
                "Ground station number": "1",
                "Ground station name": "Kashima",
                "Timestamp": "1972-01-03 02:44:04",
            }
        )

    with (raw / "microapp_ISIS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "max_depth", "fmin"])
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "B1-35-20 ISIS B D-409/Image0001.png",
                "max_depth": "300",
                "fmin": "0.5",
            }
        )

    with (raw / "orbitcheck_isis_2.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Subdirectory",
                "Filename",
                "TLE_Epoch",
                "Flag",
                "Sat_Lat",
                "Sat_Lon",
                "Sat_Height",
                "Station_Alt",
                "Station_Distance",
                "Events",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Subdirectory": "B1-35-20 ISIS B D-409",
                "Filename": "Image0001.png",
                "TLE_Epoch": "1972-01-01",
                "Flag": "ok",
                "Sat_Lat": "69.7",
                "Sat_Lon": "18.9",
                "Sat_Height": "1400",
                "Station_Alt": "10",
                "Station_Distance": "5",
                "Events": "none",
            }
        )

    stats, _ = csa.build()

    assert stats["rows"] == 1
    assert stats["in_microapp"] == 1
    assert stats["in_orbitcheck"] == 1
    with (output / "csa_master.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["max_depth"] == "300"
    assert row["timestamp_trust"] == "ok"


def test_nasa_download_failure_is_nonfatal(tmp_path, monkeypatch):
    monkeypatch.setattr(nasa, "RAW_DIR", tmp_path / "headers")

    def fail_fetch(url):
        raise OSError("offline")

    monkeypatch.setattr(nasa, "fetch", fail_fetch)

    assert nasa.download_one(("RES_75N_265E", "1972", "missing.asc")) is False


def test_phase1_alignment_records_subprocess_failure(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.csv"
    output = tmp_path / "landmarks"
    csa_path = tmp_path / "film.png"
    cdf_path = tmp_path / "scan.cdf"
    csa_path.write_bytes(b"film")
    cdf_path.write_bytes(b"cdf")
    with pairs.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pair_name", "csa_link", "cdf_link"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_name": "offline_pair",
                "csa_link": "film.png",
                "cdf_link": "scan.cdf",
            }
        )

    class FailedProcess:
        returncode = 1
        stderr = "aligner failed\n"
        stdout = ""

    monkeypatch.setattr(aligner, "ALIGN", tmp_path / "align_landmarks.py")
    monkeypatch.setattr(
        aligner.subprocess, "run", lambda *args, **kwargs: FailedProcess()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "align_phase1_pairs",
            "--pairs",
            str(pairs),
            "--out",
            str(output),
            "--workers",
            "1",
            "--reuse",
            str(tmp_path / "empty_reuse"),
        ],
    )

    aligner.main()

    with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["status"] == "failed"
    assert row["error"] == "aligner failed"
