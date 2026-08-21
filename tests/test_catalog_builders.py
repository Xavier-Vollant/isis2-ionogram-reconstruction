import sys
from pathlib import Path

import pytest

DATASET_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "dataset"
sys.path.insert(0, str(DATASET_SCRIPTS))

import build_csa_master as csa  # noqa: E402
import build_nasa_master as nasa  # noqa: E402


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
