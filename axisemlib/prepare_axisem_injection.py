"""Prepare AxiSEM input files for one injection event without submitting it."""

import argparse
import re
from pathlib import Path

import numpy as np

from .utils import geodetic_to_geocentric


def _replace_parameter(contents: str, name: str, value: str, path: Path,
                       insert_after: str | None = None) -> str:
    """Replace one active parameter line, preserving all other lines."""
    pattern = re.compile(rf"^{re.escape(name)}\s+.*$", re.MULTILINE)
    result, count = pattern.subn(f"{name}        {value}", contents)
    if count == 0 and insert_after is not None:
        anchor = re.compile(rf"^({re.escape(insert_after)}\s+.*)$", re.MULTILINE)
        result, count = anchor.subn(
            lambda match: f"{match.group(1)}\n{name}        {value}", contents,
        )
    if count != 1:
        raise ValueError(f"Expected one {name} entry in {path}, found {count}")
    return result


def _convert_cmt(contents: str, path: Path) -> tuple[str, float, float]:
    """Convert event latitude and return its geocentric coordinates."""
    coordinates = {}
    lines = []
    for line in contents.splitlines(keepends=True):
        for key in ("latitude", "longitude"):
            if line.startswith(key + ":"):
                coordinates[key] = float(line.split(":", 1)[1])
                if key == "latitude":
                    line = f"latitude:      {geodetic_to_geocentric(coordinates[key]):.4f}\n"
        lines.append(line)

    if set(coordinates) != {"latitude", "longitude"}:
        raise ValueError(f"Missing latitude or longitude in {path}")
    return (
        "".join(lines),
        float(geodetic_to_geocentric(coordinates["latitude"])),
        coordinates["longitude"],
    )


def _convert_stations(contents: str) -> str:
    """Convert station latitudes in SPECFEM STATIONS rows."""
    lines = []
    for line in contents.splitlines(keepends=True):
        fields = line.split()
        if len(fields) >= 4 and not line.lstrip().startswith("#"):
            fields[2] = f"{geodetic_to_geocentric(float(fields[2])):.6f}"
            line = " ".join(fields) + "\n"
        lines.append(line)
    return "".join(lines)


def _distance_range(event_lat: float, event_lon: float,
                    region_lat: np.ndarray, region_lon: np.ndarray,
                    buffer_deg: float) -> tuple[float, float]:
    """Return the sampled epicentral distance range with the requested margin."""
    event_lat = np.deg2rad(event_lat)
    event_lon = np.deg2rad(event_lon)
    region_lat = np.deg2rad(region_lat)
    region_lon = np.deg2rad(region_lon)
    cos_arc = (np.sin(region_lat) * np.sin(event_lat)
               + np.cos(region_lat) * np.cos(event_lat)
               * np.cos(region_lon - event_lon))
    distances = np.rad2deg(np.arccos(np.clip(cos_arc, -1.0, 1.0)))
    return float(np.min(distances) - buffer_deg), float(np.max(distances) + buffer_deg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--region-box", nargs=5, type=float, required=True,
        default=argparse.SUPPRESS,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX", "MAX_DEPTH_KM"),
        help="Geographic study region and maximum depth in kilometres",
    )
    parser.add_argument("--t0-injection", type=float, required=True,
                        default=argparse.SUPPRESS,
                        help="Injection arrival time in seconds")
    parser.add_argument("--pre-buffer-seconds", type=float, default=50.0,
                        help="Time saved before injection in seconds")
    parser.add_argument("--post-buffer-seconds", type=float, default=300.0,
                        help="Time saved after injection in seconds")
    parser.add_argument("--region-buffer-deg", type=float, default=10.0,
                        help="Angular margin around the study region in degrees")
    parser.add_argument("--cmtsolution", type=Path,
                        default=Path("CMT_DIR/CMTSOLUTION"),
                        help="Geographic CMTSOLUTION input")
    parser.add_argument("--stations", type=Path,
                        default=Path("CMT_DIR/STATIONS"),
                        help="Geographic STATIONS input")
    parser.add_argument("--inparam-basic", type=Path,
                        default=Path("inparam_basic"),
                        help="Basic parameter input")
    parser.add_argument("--inparam-advanced", type=Path,
                        default=Path("inparam_advanced"),
                        help="Advanced parameter input")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory for the four prepared solver inputs")
    args = parser.parse_args()

    lon_min, lon_max, lat_min, lat_max, max_depth = args.region_box
    if lon_min > lon_max or lat_min > lat_max or not -90 <= lat_min <= lat_max <= 90:
        parser.error("Region bounds must be ascending and latitudes within [-90, 90]")
    if not 0 <= max_depth < 6371:
        parser.error("MAX_DEPTH_KM must be between 0 and 6371")
    if min(args.pre_buffer_seconds, args.post_buffer_seconds,
           args.region_buffer_deg) < 0:
        parser.error("Buffer sizes must be nonnegative")
    if args.t0_injection < args.pre_buffer_seconds:
        parser.error("T0_INJECTION must be at least PRE_BUFFER_SECONDS")

    output = args.output_dir
    if args.cmtsolution.resolve() == (output / "CMTSOLUTION").resolve():
        parser.error("CMTSOLUTION input and output must differ to avoid repeat conversion")
    if args.stations.resolve() == (output / "STATIONS").resolve():
        parser.error("STATIONS input and output must differ to avoid repeat conversion")

    lon = np.linspace(lon_min, lon_max, 100)
    lat = geodetic_to_geocentric(np.linspace(lat_min, lat_max, 100))
    region_lon, region_lat = np.meshgrid(lon, lat, indexing="ij")
    basic_template = args.inparam_basic.read_text()
    advanced_template = args.inparam_advanced.read_text()

    # Convert the event coordinates and determine its wavefield sampling bounds.
    cmt, event_lat, event_lon = _convert_cmt(
        args.cmtsolution.read_text(), args.cmtsolution,
    )
    stations = _convert_stations(args.stations.read_text())
    distance_min, distance_max = _distance_range(
        event_lat, event_lon, region_lat, region_lon, args.region_buffer_deg,
    )

    # Substitute the single event's time and region parameters.
    basic = _replace_parameter(
        basic_template, "SEISMOGRAM_LENGTH",
        f"{args.t0_injection + args.post_buffer_seconds:g}", args.inparam_basic,
    )
    advanced = advanced_template
    values = {
        "DUMP_T0": f"{args.t0_injection - args.pre_buffer_seconds:g}",
        "KERNEL_COLAT_MIN": f"{distance_min:f}",
        "KERNEL_COLAT_MAX": f"{distance_max:f}",
        "KERNEL_RMIN": f"{6371 - max_depth:g}",
        "KERNEL_RMAX": "7000",
    }
    for parameter, value in values.items():
        advanced = _replace_parameter(
            advanced, parameter, value, args.inparam_advanced,
            insert_after="KERNEL_SPP" if parameter == "DUMP_T0" else None,
        )

    # Write the four solver input files to the selected directory.
    output.mkdir(parents=True, exist_ok=True)
    for filename, contents in (
        ("inparam_basic", basic),
        ("inparam_advanced", advanced),
        ("CMTSOLUTION", cmt),
        ("STATIONS", stations),
    ):
        (output / filename).write_text(contents)
    print(f"Prepared AxiSEM inputs in {output.resolve()}")


if __name__ == "__main__":
    main()
