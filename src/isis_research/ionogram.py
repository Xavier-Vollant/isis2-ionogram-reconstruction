"""The one calibrated-ionogram artifact, and the rules it has to obey.

Three producers in this repository wrote three different arrays under three
different sets of keys, in two different orientations:

    warp_calibrated_scan.py   warped_intensity/valid_mask/frequency_mhz/virtual_height_km   (height, frequency)
    standardize_scan.py       warped/freq_axis/v_height                                     (frequency, height)
    warp_film_to_nasa.py      warped_film/nasa_ampl/freq/v_height                           (frequency, height)

No consumer could read more than one of them. The film-only deployment entry
point - the script meant for the ~313,000 archive scans with no NASA
counterpart - produced a file the production amplitude model raised KeyError
on, and the trace extractor could read neither of the other two.

The orientation split is the dangerous half. A Phase 6 grid is 512x512, so an
array stored the wrong way round still matches its own axis lengths and still
resamples to a well-formed 64x96 output. Nothing raises. Shape checking cannot
catch a square mistake, which is why orientation is a declared field here
rather than something inferred.

`write` emits only the canonical layout.  `read` remains available for audit
and migration of the historical layouts, but records their missing guarantees.
Production consumers use `read_validated`, which refuses every artifact that
does not meet the canonical contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "isis.ionogram.v1"
ORIENTATION = "height,frequency"
STATUSES = ("usable", "review", "not_usable")
ROUTES = ("film_only", "cdf_assisted")

# intensity key -> (mask key, frequency key, height key, stored orientation)
_LAYOUTS = {
    "warped_intensity": (
        "valid_mask",
        "frequency_mhz",
        "virtual_height_km",
        "height,frequency",
    ),
    "warped": (None, "freq_axis", "v_height", "frequency,height"),
    "warped_film": (None, "freq", "v_height", "frequency,height"),
}


@dataclass
class Ionogram:
    """A calibrated scan on a regular frequency x virtual-height grid.

    `intensity` is normalized film brightness in [0, 1] laid out as
    (height, frequency). Film traces are dark, so signal-positive readers use
    `1 - intensity`. `valid_mask` is False wherever no film supports the pixel.
    """

    intensity: np.ndarray
    valid_mask: np.ndarray
    frequency_mhz: np.ndarray
    virtual_height_km: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def support(self):
        return {
            "frequency_mhz": [
                float(self.frequency_mhz[0]),
                float(self.frequency_mhz[-1]),
            ],
            "virtual_height_km": [
                float(self.virtual_height_km[0]),
                float(self.virtual_height_km[-1]),
            ],
        }

    @property
    def coverage(self):
        """Fraction of grid pixels backed by real film."""
        return float(self.valid_mask.mean())


def _detect_layout(names):
    present = [key for key in _LAYOUTS if key in names]
    if not present:
        raise KeyError(
            f"no ionogram intensity array found; expected one of {sorted(_LAYOUTS)}, got {sorted(names)}"
        )
    if len(present) > 1:
        raise KeyError(
            f"ambiguous artifact: several intensity arrays present ({present})"
        )
    return present[0]


def _sidecar_meta(path):
    """Phase 6 wrote its metadata beside the NPZ rather than inside it."""
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def _embedded_meta(data):
    """Return embedded metadata plus where it came from, without defaults."""
    if "meta_json" not in data.files:
        return None, "absent"
    try:
        raw = np.asarray(data["meta_json"])
        if raw.ndim != 0:
            raise ValueError("meta_json must be a scalar JSON object")
        raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            raise ValueError("meta_json must contain JSON text")
        meta = json.loads(raw)
    except ValueError as error:
        return {"metadata_error": str(error)}, "invalid_embedded"
    if not isinstance(meta, dict):
        return {"metadata_error": "meta_json must decode to a JSON object"}, "invalid_embedded"
    return meta, "embedded"


def require_valid(artifact):
    """Return a conforming artifact or fail before a consumer can use it."""
    problems = validate(artifact)
    if problems:
        raise ValueError("invalid ionogram artifact: " + "; ".join(problems))
    return artifact


def read(path):
    """Read any historical layout for audit or migration, preserving violations."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        names = set(data.files)
        intensity_key = _detect_layout(names)
        mask_key, frequency_key, height_key, orientation = _LAYOUTS[intensity_key]
        required = (frequency_key, height_key)
        missing = [key for key in required if key not in names]
        if missing:
            raise ValueError(
                "ionogram is missing required axis fields: " + ", ".join(missing)
            )
        intensity = np.asarray(data[intensity_key], dtype=float)
        frequency = np.asarray(data[frequency_key], dtype=float)
        height = np.asarray(data[height_key], dtype=float)
        embedded, metadata_source = _embedded_meta(data)
        if embedded is None:
            meta = _sidecar_meta(path)
            metadata_source = "sidecar" if meta else metadata_source
        else:
            meta = {**_sidecar_meta(path), **embedded}
        if mask_key is not None and mask_key in names:
            valid = np.asarray(data[mask_key], dtype=bool)
            valid_mask_source = "artifact"
        else:
            valid = np.ones(intensity.shape, dtype=bool)
            valid_mask_source = "synthesized_absent_in_source_artifact"

    if orientation != ORIENTATION:
        intensity = intensity.T
        valid = valid.T
    meta.setdefault("source_layout", intensity_key)
    meta.setdefault("storage_orientation", orientation)
    meta.setdefault("valid_mask_source", valid_mask_source)
    meta["metadata_source"] = metadata_source
    return Ionogram(intensity, valid, frequency, height, meta)


def read_validated(path):
    """Read the sole production format, rejecting legacy or invalid artifacts."""
    return require_valid(read(path))


def validate(ionogram):
    """-> list of contract violations, empty when the artifact conforms."""
    problems = []
    intensity = np.asarray(ionogram.intensity)
    valid = np.asarray(ionogram.valid_mask)
    frequency = np.asarray(ionogram.frequency_mhz)
    height = np.asarray(ionogram.virtual_height_km)
    meta = ionogram.meta if isinstance(ionogram.meta, dict) else {}

    if not isinstance(ionogram.meta, dict):
        problems.append("metadata must be a dictionary")
    if meta.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}, got {meta.get('schema')!r}")
    if meta.get("source_layout") not in (None, "warped_intensity"):
        problems.append(
            "source_layout must be 'warped_intensity', got "
            f"{meta.get('source_layout')!r}"
        )
    if meta.get("valid_mask_source") not in (None, "artifact"):
        problems.append("valid_mask must be stored inside the artifact")
    if meta.get("metadata_source") not in (None, "embedded"):
        problems.append("metadata must be embedded inside the artifact")

    if intensity.ndim != 2:
        problems.append(f"intensity must be 2-D, got {intensity.ndim}-D")
        return problems
    if valid.shape != intensity.shape:
        problems.append(
            f"valid_mask {valid.shape} does not match intensity {intensity.shape}"
        )
    if height.ndim != 1:
        problems.append(f"virtual_height_km must be 1-D, got {height.ndim}-D")
    elif len(height) != intensity.shape[0]:
        problems.append(
            f"height axis {len(height)} does not match intensity rows {intensity.shape[0]}"
        )
    if frequency.ndim != 1:
        problems.append(f"frequency_mhz must be 1-D, got {frequency.ndim}-D")
    elif len(frequency) != intensity.shape[1]:
        problems.append(
            f"frequency axis {len(frequency)} does not match intensity columns {intensity.shape[1]}"
        )

    for name, axis in (("frequency_mhz", frequency), ("virtual_height_km", height)):
        if axis.ndim != 1:
            continue
        if not np.all(np.isfinite(axis)):
            problems.append(f"{name} contains non-finite values")
        elif len(axis) > 1 and not np.all(np.diff(axis) > 0):
            problems.append(f"{name} is not strictly increasing")

    if meta.get("orientation") != ORIENTATION:
        problems.append(
            f"orientation must be declared as {ORIENTATION!r}, got {meta.get('orientation')!r}"
        )
    status = meta.get("status")
    if status not in STATUSES:
        problems.append(f"status must be one of {STATUSES}, got {status!r}")
    route = meta.get("route")
    if route not in ROUTES:
        problems.append(f"route must be one of {ROUTES}, got {route!r}")

    confidence = meta.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        problems.append("confidence must be a finite number in [0, 1]")
    else:
        if isinstance(meta.get("confidence"), (bool, np.bool_)) or not (
            np.isfinite(confidence) and 0.0 <= confidence <= 1.0
        ):
            problems.append("confidence must be a finite number in [0, 1]")

    source = meta.get("source")
    if not isinstance(source, dict) or not source:
        problems.append("source must be a non-empty dictionary")
    provenance = meta.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("producer"):
        problems.append("provenance must be a dictionary with a producer")

    if valid.shape == intensity.shape and valid.any():
        supported = intensity[valid]
        if not np.all(np.isfinite(supported)):
            problems.append(
                "intensity contains non-finite values where valid_mask is True"
            )
        elif supported.min() < 0.0 or supported.max() > 1.0:
            problems.append(
                f"intensity outside [0, 1] where valid: [{supported.min():.3f}, {supported.max():.3f}]"
            )

    declared = meta.get("support")
    if declared and not isinstance(declared, dict):
        problems.append("support must be a dictionary")
    elif declared and frequency.ndim == height.ndim == 1 and len(frequency) and len(height):
        for name, axis in (("frequency_mhz", frequency), ("virtual_height_km", height)):
            bounds = declared.get(name)
            if bounds and not np.allclose(bounds, [axis[0], axis[-1]], atol=1e-6):
                problems.append(
                    f"declared {name} support {bounds} disagrees with the axis "
                    f"[{axis[0]}, {axis[-1]}]"
                )
    return problems


def write(path, intensity, valid_mask, frequency_mhz, virtual_height_km, **meta):
    """Validate, then write the canonical layout with its metadata inside it."""
    intensity = np.asarray(intensity, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    frequency_mhz = np.asarray(frequency_mhz, dtype=np.float64)
    virtual_height_km = np.asarray(virtual_height_km, dtype=np.float64)

    meta = {
        **meta,
        "schema": SCHEMA,
        "orientation": ORIENTATION,
        "source_layout": "warped_intensity",
    }
    ionogram = Ionogram(intensity, valid_mask, frequency_mhz, virtual_height_km, meta)
    meta.setdefault("support", ionogram.support)
    meta.setdefault("coverage", ionogram.coverage)
    problems = validate(ionogram)
    if problems:
        raise ValueError(
            "refusing to write a non-conforming ionogram: " + "; ".join(problems)
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        warped_intensity=intensity,
        valid_mask=valid_mask,
        frequency_mhz=frequency_mhz,
        virtual_height_km=virtual_height_km,
        meta_json=json.dumps(meta, sort_keys=True),
    )
    return path
