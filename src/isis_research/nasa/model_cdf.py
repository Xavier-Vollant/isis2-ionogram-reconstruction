"""Export an existing ISIS model prediction as a NASA-CDF-like ionogram."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import cdflib
import numpy as np
from cdflib import cdfwrite

CDF_EPOCH_MS = 149.896229
MODEL_KEYS = ("prediction", "amplitude", "model_output")
REQUIRED_HEADER = (
    "satellite_number",
    "satellite_height_km",
    "geographic_latitude_deg",
    "geographic_longitude_deg",
    "FH",
    "DIP",
    "CHI",
    "L",
    "GMLAT",
    "GMLONG",
    "INV_LAT",
    "year",
    "doy",
    "hr",
    "min",
    "sec",
)


def _scalar(value, name):
    value = np.asarray(value)
    if value.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return value.item()


def _axis(value, name, size):
    value = np.asarray(value, dtype=float).ravel()
    if value.size != size or value.size < 2:
        raise ValueError(f"{name} must contain {size} values")
    if not np.all(np.isfinite(value)) or not np.all(np.diff(value) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return value


def read_model_output(path, *, scale="unit"):
    """Read and validate the current model NPZ without changing the model."""
    with np.load(path, allow_pickle=False) as data:
        output = next((np.asarray(data[key], dtype=float) for key in MODEL_KEYS if key in data), None)
        if output is None:
            raise KeyError("model output must contain prediction, amplitude, or model_output")
        if output.ndim != 2 or not np.all(np.isfinite(output)):
            raise ValueError("model prediction must be a finite two-dimensional array")
        metadata = {}
        if "meta_json" in data:
            raw = _scalar(data["meta_json"], "meta_json")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            metadata = json.loads(raw)
            if not isinstance(metadata, dict):
                raise ValueError("meta_json must decode to an object")
        orientation = metadata.get("orientation", "height,frequency")
        if orientation == "frequency,height":
            output = output.T
        elif orientation != "height,frequency":
            raise ValueError(f"unsupported model orientation: {orientation!r}")
        frequency = _axis(data["frequency_mhz"], "frequency_mhz", output.shape[1])
        height = _axis(data["virtual_height_km"], "virtual_height_km", output.shape[0])
        if scale == "unit":
            if output.min() < -1e-6 or output.max() > 1.0 + 1e-6:
                raise ValueError("unit model output must lie in [0, 1]")
            output = np.clip(output, 0.0, 1.0)
        elif scale == "byte":
            if output.min() < -1e-6 or output.max() > 255.0 + 1e-6:
                raise ValueError("byte model output must lie in [0, 255]")
            output = np.clip(output / 255.0, 0.0, 1.0)
        else:
            raise ValueError("scale must be 'unit' or 'byte'")
        if "valid_mask" in data:
            valid = np.asarray(data["valid_mask"], dtype=bool)
            if orientation == "frequency,height":
                valid = valid.T
            if valid.shape != output.shape:
                raise ValueError("valid_mask must have the same shape as prediction")
        else:
            valid = np.ones(output.shape, dtype=bool)
    return output, valid, frequency, height, metadata


def _array(header, names, size, dtype=float):
    for name in names:
        if name in header:
            value = np.asarray(header[name], dtype=dtype).ravel()
            if value.size != size or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain {size} finite values")
            return value
    raise ValueError(f"pass header is missing one of: {', '.join(names)}")


def _epochs(header, size):
    if "epoch" in header:
        epoch = np.asarray(header["epoch"], dtype=float).ravel()
        if epoch.size != size or not np.all(np.isfinite(epoch)):
            raise ValueError(f"epoch must contain {size} finite values")
        return epoch
    if "epoch_start_ms" not in header or "scan_times_ms" not in header:
        raise ValueError("pass header needs epoch or epoch_start_ms plus scan_times_ms")
    times = np.asarray(header["scan_times_ms"], dtype=float).ravel()
    if times.size != size or not np.all(np.isfinite(times)):
        raise ValueError(f"scan_times_ms must contain {size} finite values")
    return float(header["epoch_start_ms"]) + times


def _header_value(header, names, default):
    for name in names:
        if name in header:
            return header[name]
    return default


def _required_header(header):
    missing = [name for name in REQUIRED_HEADER if name not in header]
    if missing:
        raise ValueError("pass header is missing: " + ", ".join(missing))


def _variable_spec(name, value, data_type, *, chars=False, record_vary=False):
    value = np.asarray(value)
    if chars:
        values = [str(item) for item in value.ravel()]
        width = max([len(item) for item in values] + [1])
        data = [item.ljust(width) for item in values]
        dimensions = [len(values)]
        elements = width
    else:
        data = value
        dimensions = list(value.shape)
        elements = 1
    if record_vary:
        dimensions = dimensions[1:]
    return {
        "spec": {
            "Variable": name,
            "Data_Type": data_type,
            "Num_Elements": elements,
            "Rec_Vary": cdfwrite.CDF.VARY if record_vary else cdfwrite.CDF.NOVARY,
            "Dim_Sizes": dimensions,
        },
        "data": data,
    }


def _write(cdf, name, value, data_type, attrs=None, *, chars=False, record_vary=False):
    variable = _variable_spec(
        name, value, data_type, chars=chars, record_vary=record_vary
    )
    cdf.write_var(variable["spec"], var_attrs=attrs or {}, var_data=variable["data"])


def _base_attrs(field, units=" "):
    return {"FIELDNAM": field, "UNITS": units, "VAR_TYPE": "data"}


def _assemble(model_output, header, *, scale="unit"):
    output, valid, frequency, height, model_meta = read_model_output(model_output, scale=scale)
    _required_header(header)
    ampl = np.clip(np.rint(np.where(valid, output, 0.0).T * 255.0), 0, 255).astype(np.uint8)
    n_scan, n_range = ampl.shape
    delay = np.asarray(header.get("delay_time", height / CDF_EPOCH_MS), dtype=float).ravel()
    if delay.size != n_range or not np.all(np.isfinite(delay)):
        raise ValueError(f"delay_time must contain {n_range} finite values")
    epoch = _epochs(header, n_scan)
    marker_times = _array(header, ("frequency_marker_times_ms", "Time_mark"), 22)
    marker_freq = _array(header, ("frequency_markers_mhz", "freq_mark"), 22)
    lmt = np.asarray(_header_value(header, ("local_mean_time", "LMT"), [header.get("hr", 0), header.get("min", 0)]), dtype=np.int32).ravel()
    glmt = np.asarray(_header_value(header, ("geomagnetic_local_time", "GMLMT"), lmt), dtype=np.int32).ravel()
    if lmt.size != 2 or glmt.size != 2:
        raise ValueError("LMT and GMLMT must contain hour and minute")
    geo = np.asarray([header["geographic_latitude_deg"], header["geographic_longitude_deg"], header["satellite_height_km"]], dtype=np.float32)
    values = {
        "satellite": int(header["satellite_number"]),
        "station_id": int(header.get("station_id", -1)),
        "power_code": int(header.get("power_code", -1)),
        "s/r_code": int(header.get("s/r_code", -1)),
        "f_range_code": int(header.get("f_range_code", -1)),
        "DMODE": int(header.get("DMODE", -1)),
        "GMODE": int(header.get("GMODE", -1)),
        "mixed_mode": int(header.get("mixed_mode", -1)),
        "AIT_mode": int(header.get("AIT_mode", -1)),
        "fix_freq": int(header.get("fix_freq", -1)),
        "year": int(header.get("year", -1)),
        "doy": int(header.get("doy", -1)),
        "hr": int(header.get("hr", -1)),
        "min": int(header.get("min", -1)),
        "sec": float(header.get("sec", -1.0)),
        "LMT": lmt,
        "geo_coord": geo,
        "GMLMT": glmt,
        "GMLAT": float(header["GMLAT"]),
        "GMLONG": float(header["GMLONG"]),
        "FH": float(header["FH"]),
        "INV_LAT": float(header["INV_LAT"]),
        "DIP": float(header["DIP"]),
        "CHI": float(header["CHI"]),
        "L": float(header["L"]),
        "sun": int(header.get("sun", -1)),
        "CEP": int(header.get("CEP", -1)),
        "VLF": int(header.get("VLF", -1)),
        "RPA": int(header.get("RPA", -1)),
        "IMS": int(header.get("IMS", -1)),
        "SPS": int(header.get("SPS", -1)),
        "EPD": int(header.get("EPD", -1)),
        "RLP": int(header.get("RLP", -1)),
        "ASP": int(header.get("ASP", -1)),
        "swept_start": 1,
        "Time_mark": marker_times,
        "freq_mark": marker_freq,
        "vh_num": n_scan,
        "f_num": n_range,
        "delay_time": delay,
        "v_height": height,
        "Epoch": epoch,
        "ampl": ampl,
        "freq": frequency,
        "valid_mask": valid.T.astype(np.uint8),
        "label_LMT": ["LMT(hh)", "LMT(mm)"],
        "unit_LMT": ["hrs", "min"],
        "label_geo": ["latitude", "longitude", "height"],
        "unit_geo": ["deg", "deg", "km"],
        "label_GMLMT": ["GMLMT(hh)", "GMLMT(mm)"],
    }
    csa_only = header.get("metadata_source") == "csa_scan_only"
    provenance = {
        "ampl": "model_prediction_quantized_to_uint8",
        "freq": "csa_artifact_axis" if csa_only else "model_output_axis",
        "v_height": "csa_artifact_axis" if csa_only else "model_output_axis",
        "delay_time": "pass_header" if "delay_time" in header else "derived_from_v_height",
        "Epoch": "csa_pair_name" if csa_only else "pass_header",
        "valid_mask": "model_output_or_all_true",
        "swept_start": "derived_model_grid_has_no_fixed_prefix",
        "pass_metadata": header.get("metadata_source", "external_header"),
        "csa_station": header.get("csa_station", "unknown"),
        "source": header.get("source", "external_header"),
        "unknown_pass_fields": header.get("unknown_fields", []),
        "model_metadata": model_meta,
    }
    return values, provenance


def export_model_cdf(model_output, header, destination, *, scale="unit"):
    """Write a model-derived CDF and return its assembled values/provenance."""
    values, provenance = _assemble(model_output, header, scale=scale)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    cdf = cdfwrite.CDF(destination, cdf_spec={"Encoding": cdfwrite.CDF.NETWORK_ENCODING, "Majority": "Row_major"})
    try:
        cdf.write_globalattrs(
            {
                "Project": {0: "FINAL ISIS"},
                "Data_type": {0: "MODEL_DERIVED_AVERAGE_IONOGRAM"},
                "TITLE": {0: "Model-derived NASA-CDF-like ionogram"},
                "FINALISIS_SCHEMA": {0: "final_isis.model_derived_cdf.v1"},
                "FINALISIS_PROVENANCE": {0: json.dumps(provenance, sort_keys=True)},
            }
        )
        ints = ("satellite", "station_id", "power_code", "s/r_code", "f_range_code", "DMODE", "GMODE", "mixed_mode", "AIT_mode", "fix_freq", "year", "doy", "hr", "min", "DIP", "CHI", "sun", "CEP", "VLF", "RPA", "IMS", "SPS", "EPD", "RLP", "ASP", "swept_start", "vh_num", "f_num")
        int_vectors = ("LMT", "GMLMT")
        float32_scalars = ("GMLAT", "GMLONG", "FH", "INV_LAT", "L")
        float64_scalars = ("sec",)
        vectors = ("Time_mark", "freq_mark", "delay_time", "v_height", "Epoch", "freq")
        strings = ("label_LMT", "unit_LMT", "label_geo", "unit_geo", "label_GMLMT")
        for name in ints:
            _write(cdf, name, np.asarray(values[name], dtype=np.int32), cdfwrite.CDF.CDF_INT4, _base_attrs(name))
        for name in int_vectors:
            _write(cdf, name, np.asarray(values[name], dtype=np.int32), cdfwrite.CDF.CDF_INT4, _base_attrs(name))
        for name in float32_scalars:
            _write(cdf, name, np.asarray(values[name], dtype=np.float32), cdfwrite.CDF.CDF_FLOAT, _base_attrs(name))
        for name in float64_scalars:
            _write(cdf, name, np.asarray(values[name], dtype=np.float64), cdfwrite.CDF.CDF_DOUBLE, _base_attrs(name))
        for name in vectors:
            _write(cdf, name, np.asarray(values[name], dtype=np.float64), cdfwrite.CDF.CDF_DOUBLE, _base_attrs(name), record_vary=name in {"Epoch", "freq"})
        _write(cdf, "geo_coord", np.asarray(values["geo_coord"], dtype=np.float32), cdfwrite.CDF.CDF_FLOAT, _base_attrs("Geographic coordinates", "deg,deg,km"))
        for name in strings:
            _write(cdf, name, values[name], cdfwrite.CDF.CDF_CHAR, _base_attrs(name), chars=True)
        _write(cdf, "ampl", values["ampl"], cdfwrite.CDF.CDF_UINT1, _base_attrs("Sounder amplitude", ""), record_vary=True)
        _write(cdf, "valid_mask", values["valid_mask"], cdfwrite.CDF.CDF_UINT1, _base_attrs("Model valid mask", ""), record_vary=True)
    finally:
        cdf.close()
    return values, provenance


def header_from_csa(pair_name, station, frequency, height):
    """Build export metadata from a CSA pair name and calibrated CSA axes only."""
    name = Path(str(pair_name)).stem
    station = str(station or "").strip()
    if station.lower() in {"", "<blank>", "unknown"}:
        station_match = re.search(r"(?:^|_)i2_av_([^_]+)_\d{13}(?:_v\d+)?$", name, re.IGNORECASE)
        station = station_match.group(1).upper() if station_match else "unknown"
    match = re.search(r"(?<!\d)(\d{13})(?:_v\d+)?$", name)
    if not match:
        raise ValueError(
            "CSA pair name must end with YYYYDDDHHMMSS[_vNN] to derive observation time"
        )
    digits = match.group(1)
    year = int(digits[:4])
    doy = int(digits[4:7])
    hour = int(digits[7:9])
    minute = int(digits[9:11])
    second = int(digits[11:13])
    timestamp = datetime(year, 1, 1) + timedelta(
        days=doy - 1, hours=hour, minutes=minute, seconds=second
    )
    epoch = float(
        cdflib.cdfepoch.compute_epoch(
            [
                timestamp.year,
                timestamp.month,
                timestamp.day,
                timestamp.hour,
                timestamp.minute,
                timestamp.second,
                0,
            ]
        )
    )
    frequency = np.asarray(frequency, dtype=float).ravel()
    height = np.asarray(height, dtype=float).ravel()
    for label, axis in (("frequency", frequency), ("height", height)):
        if axis.size < 2 or not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
            raise ValueError(f"CSA {label} axis must be finite and strictly increasing")

    unknown_ints = (
        "station_id",
        "power_code",
        "s/r_code",
        "f_range_code",
        "DMODE",
        "GMODE",
        "mixed_mode",
        "AIT_mode",
        "fix_freq",
        "sun",
        "CEP",
        "VLF",
        "RPA",
        "IMS",
        "SPS",
        "EPD",
        "RLP",
        "ASP",
    )
    unknown_floats = (
        "satellite_height_km",
        "geographic_latitude_deg",
        "geographic_longitude_deg",
        "GMLAT",
        "GMLONG",
        "FH",
        "INV_LAT",
        "DIP",
        "CHI",
        "L",
    )
    marker_frequencies = np.linspace(frequency[0], frequency[-1], 22).tolist()
    return {
        "satellite_number": -1,
        "station_id": -1,
        "power_code": -1,
        "s/r_code": -1,
        "f_range_code": -1,
        "DMODE": -1,
        "GMODE": -1,
        "mixed_mode": -1,
        "AIT_mode": -1,
        "fix_freq": -1,
        "year": year % 100,
        "doy": doy,
        "hr": hour,
        "min": minute,
        "sec": float(second),
        "local_mean_time": [-1, -1],
        "geomagnetic_local_time": [-1, -1],
        "geographic_latitude_deg": -1.0,
        "geographic_longitude_deg": -1.0,
        "satellite_height_km": -1.0,
        "GMLAT": -1.0,
        "GMLONG": -1.0,
        "FH": -1.0,
        "INV_LAT": -1.0,
        "DIP": -1,
        "CHI": -1,
        "L": -1.0,
        "sun": -1,
        "CEP": -1,
        "VLF": -1,
        "RPA": -1,
        "IMS": -1,
        "SPS": -1,
        "EPD": -1,
        "RLP": -1,
        "ASP": -1,
        "epoch": np.full(frequency.size, epoch, dtype=float).tolist(),
        "frequency_marker_times_ms": np.full(22, -1e31, dtype=float).tolist(),
        "frequency_markers_mhz": marker_frequencies,
        "metadata_source": "csa_scan_only",
        "source": f"CSA/{station or 'unknown_station'}/{name}",
        "csa_station": station or "unknown",
        "frequency_marker_table": "evenly_spaced_reference_from_csa_axis",
        "unknown_fields": [
            "satellite_number",
            *unknown_ints,
            *unknown_floats,
            "LMT",
            "GMLMT",
            "Time_mark",
        ],
    }


def header_from_cdf(path):
    """Extract a complete exporter header from a NASA CDF for tests/audits."""
    cdf = cdflib.CDF(str(path))

    def flat(name):
        return np.asarray(cdf.varget(name)).ravel()

    geo = flat("geo_coord")
    return {
        "satellite_number": int(flat("satellite")[0]),
        "station_id": int(flat("station_id")[0]),
        "power_code": int(flat("power_code")[0]),
        "s/r_code": int(flat("s/r_code")[0]),
        "f_range_code": int(flat("f_range_code")[0]),
        "DMODE": int(flat("DMODE")[0]),
        "GMODE": int(flat("GMODE")[0]),
        "mixed_mode": int(flat("mixed_mode")[0]),
        "AIT_mode": int(flat("AIT_mode")[0]),
        "fix_freq": int(flat("fix_freq")[0]),
        "year": int(flat("year")[0]),
        "doy": int(flat("doy")[0]),
        "hr": int(flat("hr")[0]),
        "min": int(flat("min")[0]),
        "sec": float(flat("sec")[0]),
        "local_mean_time": flat("LMT").astype(int).tolist(),
        "geomagnetic_local_time": flat("GMLMT").astype(int).tolist(),
        "geographic_latitude_deg": float(geo[0]),
        "geographic_longitude_deg": float(geo[1]),
        "satellite_height_km": float(geo[2]),
        "GMLAT": float(flat("GMLAT")[0]),
        "GMLONG": float(flat("GMLONG")[0]),
        "FH": float(flat("FH")[0]),
        "INV_LAT": float(flat("INV_LAT")[0]),
        "DIP": float(flat("DIP")[0]),
        "CHI": float(flat("CHI")[0]),
        "L": float(flat("L")[0]),
        "sun": int(flat("sun")[0]),
        "CEP": int(flat("CEP")[0]),
        "VLF": int(flat("VLF")[0]),
        "RPA": int(flat("RPA")[0]),
        "IMS": int(flat("IMS")[0]),
        "SPS": int(flat("SPS")[0]),
        "EPD": int(flat("EPD")[0]),
        "RLP": int(flat("RLP")[0]),
        "ASP": int(flat("ASP")[0]),
        "epoch": flat("Epoch").tolist(),
        "frequency_marker_times_ms": flat("Time_mark").tolist(),
        "frequency_markers_mhz": flat("freq_mark").tolist(),
    }
