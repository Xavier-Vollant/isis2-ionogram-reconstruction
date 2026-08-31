"""Parse station names and coordinates from NASA pass-header directories."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

STATION_DIRS = (
    "ACN_08S_346E",
    "ADL_67S_140E",
    "AME_23N_073E",
    "BRZ_04S_015E",
    "BUR_26S_028E",
    "CNA_28N_345E",
    "KER_49S_070E",
    "KRU_05N_307E",
    "KSH_36N_141E",
    "KWA_09N_168E",
    "LAU_45S_170E",
    "LIM_12S_283E",
    "ODG_14N_359E",
    "ORR_36S_149E",
    "OTT_45N_284E",
    "QUI_01S_281E",
    "RES_75N_265E",
    "SNT_33S_298E",
    "SOD_67N_027E",
    "SOL_52S_302E",
    "SYO_69S_040E",
    "TRO_70N_019E",
    "ULA_65N_212E",
    "WNK_51N_359E",
)


def parse_station_dir(name):
    """Parse a station code, latitude, and longitude from a directory name."""
    code, lat_text, lon_text = name.split("_")
    lat = float(lat_text[:-1]) * (-1 if lat_text[-1] == "S" else 1)
    lon = float(lon_text[:-1]) * (-1 if lon_text[-1] == "W" else 1)
    return code, lat, (lon - 360.0 if lon > 180 else lon)


STATIONS = {code: (lat, lon) for code, lat, lon in map(parse_station_dir, STATION_DIRS)}


def separation_km(lat1, lon1, lat2, lon2):
    """Return the great-circle distance between two coordinates in kilometres."""
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * 6371.0 * asin(sqrt(a))


def reconcile(code, lat, lon, tolerance_km=400.0):
    """Check a station code against its coordinates.

    Return `(normalized_code, conflict)`. Coordinates are used when the code
    and location disagree.
    """
    if lat is None or lon is None:
        return (code if code in STATIONS else ""), (
            "" if code in STATIONS else "unknown_code"
        )

    nearest, best_km = "", float("inf")
    for candidate, (clat, clon) in STATIONS.items():
        km = separation_km(lat, lon, clat, clon)
        if km < best_km:
            nearest, best_km = candidate, km

    if best_km > tolerance_km:
        return "", "no_station_within_tolerance"
    if code not in STATIONS:
        return nearest, "unknown_code"
    if code != nearest:
        return nearest, "code_coord_mismatch"
    return code, ""
