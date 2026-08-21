#!/usr/bin/env python3
"""Export an existing ISIS model prediction as a NASA-CDF-like ionogram."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cdflib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research import ionogram  # noqa: E402
from isis_research.nasa.model_cdf import (  # noqa: E402
    export_model_cdf,
    header_from_csa,
    read_model_output,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_output", type=Path, help="model prediction .npz")
    parser.add_argument("--csa", type=Path, required=True, help="validated CSA artifact (.npz)")
    parser.add_argument(
        "--pair-name",
        help="CSA pair name containing YYYYDDDHHMMSS; defaults to the raw CSA filename",
    )
    parser.add_argument("--station", default="", help="CSA station label, if known")
    parser.add_argument("--output", type=Path, required=True, help="destination .cdf")
    parser.add_argument("--scale", choices=("unit", "byte"), default="unit")
    args = parser.parse_args(argv)

    scan = ionogram.read_validated(args.csa)
    _, _, model_frequency, model_height, _ = read_model_output(args.model_output, scale=args.scale)
    if not np.allclose(model_frequency, scan.frequency_mhz) or not np.allclose(
        model_height, scan.virtual_height_km
    ):
        raise ValueError("model output axes must match the calibrated CSA axes")

    pair_name = args.pair_name or _pair_name_from_scan(args.csa, scan)
    station = args.station or str(scan.meta.get("station", ""))
    header = header_from_csa(pair_name, station, scan.frequency_mhz, scan.virtual_height_km)
    values, _ = export_model_cdf(args.model_output, header, args.output, scale=args.scale)
    cdf = cdflib.CDF(str(args.output))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ampl_shape": list(values["ampl"].shape),
                "frequency_count": int(values["freq"].size),
                "range_count": int(values["v_height"].size),
                "variable_count": len(cdf.cdf_info().zVariables),
            }
        )
    )


def _pair_name_from_scan(path, scan):
    source = scan.meta.get("source", {})
    if isinstance(source, dict) and source.get("raw_csa"):
        return Path(source["raw_csa"]).stem
    return Path(path).stem


if __name__ == "__main__":
    main()
