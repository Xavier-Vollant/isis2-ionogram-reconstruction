"""Summaries and comparisons for NASA CDF and model-derived CDF files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cdflib
import numpy as np


def _hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16]


def _value_summary(value: np.ndarray) -> dict:
    value = np.asarray(value)
    summary = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _hash(value),
    }
    if value.ndim == 0:
        item = value.item()
        if isinstance(item, (int, float, str, bool)):
            summary["value"] = item
    if np.issubdtype(value.dtype, np.number):
        numeric = value.astype(float, copy=False)
        finite = numeric[np.isfinite(numeric)]
        if finite.size:
            valid = finite[np.abs(finite) < 1e29]
            summary.update(
                {
                    "finite": int(finite.size),
                    "valid": int(valid.size),
                    "fill_values": int(finite.size - valid.size),
                }
            )
            if valid.size:
                summary.update(
                    {
                        "min": float(valid.min()),
                        "max": float(valid.max()),
                        "mean": float(valid.mean()),
                        "std": float(valid.std()),
                    }
                )
    return summary


def _inspect_cdf(cdf) -> dict:
    variables = list(cdf.cdf_info().zVariables)
    return {
        "variable_count": len(variables),
        "variables": variables,
        "global_attributes": sorted(cdf.globalattsget()),
        "fields": {
            name: _value_summary(np.asarray(cdf.varget(name))) for name in variables
        },
    }


def inspect_cdf(path) -> dict:
    """Return the file structure and compact value summaries."""
    return _inspect_cdf(cdflib.CDF(str(Path(path))))


def compare_cdf_content(nasa_path, model_path) -> dict:
    """Compare variables, metadata, axes, amplitudes, and global attributes."""
    nasa_cdf = cdflib.CDF(str(Path(nasa_path)))
    model_cdf = cdflib.CDF(str(Path(model_path)))
    nasa = _inspect_cdf(nasa_cdf)
    model = _inspect_cdf(model_cdf)
    nasa_names = set(nasa["variables"])
    model_names = set(model["variables"])
    fields = []
    for name in sorted(nasa_names | model_names):
        nasa_field = nasa["fields"].get(name)
        model_field = model["fields"].get(name)
        item = {
            "name": name,
            "nasa": nasa_field,
            "model": model_field,
            "same_shape": bool(
                nasa_field
                and model_field
                and nasa_field["shape"] == model_field["shape"]
            ),
            "same_dtype": bool(
                nasa_field
                and model_field
                and nasa_field["dtype"] == model_field["dtype"]
            ),
        }
        if nasa_field and model_field and item["same_shape"]:
            nasa_value = np.asarray(nasa_cdf.varget(name))
            model_value = np.asarray(model_cdf.varget(name))
            if np.issubdtype(nasa_value.dtype, np.number) and np.issubdtype(
                model_value.dtype, np.number
            ):
                a = nasa_value.astype(float, copy=False)
                b = model_value.astype(float, copy=False)
                finite = (
                    np.isfinite(a)
                    & np.isfinite(b)
                    & (np.abs(a) < 1e29)
                    & (np.abs(b) < 1e29)
                )
                if np.any(finite):
                    residual = b[finite] - a[finite]
                    item["numeric_difference"] = {
                        "finite": int(finite.sum()),
                        "max_abs": float(np.max(np.abs(residual))),
                        "mean_abs": float(np.mean(np.abs(residual))),
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                    }
            else:
                item["same_values"] = bool(
                    nasa_field["sha256"] == model_field["sha256"]
                )
        fields.append(item)

    nasa_attributes = set(nasa["global_attributes"])
    model_attributes = set(model["global_attributes"])
    return {
        "nasa": nasa,
        "model": model,
        "only_in_nasa": sorted(nasa_names - model_names),
        "only_in_model": sorted(model_names - nasa_names),
        "shared_variables": sorted(nasa_names & model_names),
        "global_attributes_only_in_nasa": sorted(nasa_attributes - model_attributes),
        "global_attributes_only_in_model": sorted(model_attributes - nasa_attributes),
        "fields": fields,
    }
