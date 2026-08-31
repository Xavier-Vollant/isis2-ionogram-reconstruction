"""Tests for Phase 6 preparation input validation."""

import csv
import json
import sys

import numpy as np
import pytest

from scripts.dataset import package_amplitude_dataset as package
from scripts.pipeline import standardize_film_only_512 as standardize
from scripts.training import prepare_phase6_512_image_targets as targets
from scripts.training import prepare_phase6_usable_512 as usable


def write_manifest(path, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def test_target_preparation_rejects_empty_corpus(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_manifest(corpus / "manifest.csv", ["pair_name", "split"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_phase6_512_image_targets",
            "--corpus",
            str(corpus),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit, match="target corpus manifest contains no rows"):
        targets.main()


def test_usable_corpus_preparation_rejects_empty_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_manifest(
        dataset / "dataset_index.csv",
        ["pair_name", "selected_route", "csa_warped", "nasa_cdf"],
    )
    phase1 = tmp_path / "phase1.csv"
    write_manifest(phase1, ["pair_name"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_phase6_usable_512",
            "--dataset",
            str(dataset),
            "--phase1",
            str(phase1),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit, match="dataset contains no film-only rows"):
        usable.main()


def test_package_rejects_empty_quality_gate(tmp_path, monkeypatch):
    final = tmp_path / "final.csv"
    pairs = tmp_path / "pairs.csv"
    write_manifest(final, ["status", "pair_name"])
    write_manifest(pairs, ["pair_name"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_amplitude_dataset",
            "--final",
            str(final),
            "--pairs",
            str(pairs),
            "--out",
            str(tmp_path / "out"),
            "--min-usable-pairs",
            "0",
        ],
    )

    with pytest.raises(SystemExit, match="quality gate produced no usable pairs"):
        package.main()


def test_package_rejects_quality_rows_without_packageable_pairs(tmp_path, monkeypatch):
    final = tmp_path / "final.csv"
    pairs = tmp_path / "pairs.csv"
    write_manifest(final, ["status", "pair_name", "pair_number"])
    with final.open("a", encoding="utf-8") as handle:
        handle.write("usable,missing,1\n")
    write_manifest(pairs, ["pair_name"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_amplitude_dataset",
            "--final",
            str(final),
            "--pairs",
            str(pairs),
            "--out",
            str(tmp_path / "out"),
            "--min-usable-pairs",
            "0",
        ],
    )

    with pytest.raises(SystemExit, match="could package no usable pairs"):
        package.main()


def test_standardize_review_row_keeps_warnings_and_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(standardize, "load_image", lambda path: np.zeros((10, 20)))
    monkeypatch.setattr(
        standardize,
        "extract_structure",
        lambda image: {"vertical_markers": {"candidates": [{"x": 1.0}]}},
    )
    monkeypatch.setattr(
        standardize,
        "fit_from_profile",
        lambda observed, shape, profile, metadata: {
            "status": "review",
            "warnings": ["frequency_needs_review"],
        },
    )
    monkeypatch.setattr(
        standardize,
        "fit_height_from_profile",
        lambda structure, profile, frequency: {
            "status": "usable",
            "warnings": ["height_needs_review"],
        },
    )
    monkeypatch.setattr(
        standardize,
        "warp_one",
        lambda image, frequency, height, structure, frequency_bins, height_bins: (
            {
                "status": "review",
                "warnings": ["low_warp_valid_coverage"],
                "valid_coverage": 0.5,
            },
            None,
        ),
    )

    row = standardize.process(tmp_path / "scan.png", {}, tmp_path / "out")

    assert row["reason"] == "frequency or height calibration requires review"
    assert row["valid_coverage"] == 0.5
    assert (
        row["warnings"]
        == "frequency_needs_review;height_needs_review;low_warp_valid_coverage"
    )


def test_standardize_batch_continues_after_one_file_fails(tmp_path, monkeypatch):
    bad = tmp_path / "bad.png"
    good = tmp_path / "good.png"
    bad.write_bytes(b"bad")
    good.write_bytes(b"good")
    output = tmp_path / "out"

    monkeypatch.setattr(standardize, "input_paths", lambda args: [bad, good])
    monkeypatch.setattr(standardize, "load_profile", dict)

    def fake_process(path, profile, destination):
        if path == bad:
            raise RuntimeError("malformed scan")
        return {"film_file": str(path), "status": "usable"}

    monkeypatch.setattr(standardize, "process", fake_process)
    monkeypatch.setattr(
        sys, "argv", ["standardize_film_only_512", "--output", str(output)]
    )

    standardize.main()

    with (output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["not_usable", "usable"]
    assert rows[0]["reason"] == "malformed scan"
    assert json.loads((output / "summary.json").read_text())["usable"] == 1
