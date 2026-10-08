"""Merge SPECFEM boundary faces into AxiSEM's boundary_faces.dat format."""

from pathlib import Path
from argparse import ArgumentParser
import re

import numpy as np

from utils import cart2sph, geodetic_to_geocentric


EARTH_RADIUS_M = 6371000.0
POINTS_PER_FACE = 25
INPUT_PATTERNS = {
    'cube2sph': re.compile(r'proc(\d+)_wavefield_discontinuity_faces'),
    'cart': re.compile(r'proc(\d+)_normal\.txt'),
}


def _processor_files(input_dir, system):
    """Find input files in numeric processor order, including empty files."""
    pattern = INPUT_PATTERNS[system]
    files = []
    for path in Path(input_dir).iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            files.append((int(match.group(1)), path))
    if not files:
        raise FileNotFoundError(f'No {system} boundary files in {input_dir}')
    return sorted(files)


def _read_points(path, skip_header, allowed_columns):
    """Read coordinates while retaining the original 25-point face order."""
    rows = []
    with path.open() as stream:
        if skip_header:
            next(stream, None)
        for line_number, line in enumerate(stream, 2 if skip_header else 1):
            if not line.strip():
                continue
            columns = line.split()
            if len(columns) not in allowed_columns or (rows and len(columns) != len(rows[0])):
                expected = 'x y z [nx ny nz]' if 3 in allowed_columns else 'x y z nx ny nz'
                raise ValueError(f'{path}:{line_number}: expected {expected}')
            try:
                rows.append([float(value) for value in columns])
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: invalid coordinate or normal') from exc

    if len(rows) % POINTS_PER_FACE:
        raise ValueError(f'{path}: {len(rows)} points is not a multiple of {POINTS_PER_FACE}')
    width = len(rows[0]) if rows else min(allowed_columns)
    points = np.asarray(rows, dtype=float).reshape(-1, width)
    if not np.isfinite(points).all():
        raise ValueError(f'{path}: nonfinite coordinate or normal')
    return points


def merge_surfaces(input_dir, output_file, system, utm_zone=None):
    """Write lon/lat/depth faces and return ``(proc, local_face)`` per iface.

    The 0-based iface increases by counting complete 25-row faces in ascending
    processor order. Any normal columns are omitted from the output.
    """
    if system not in INPUT_PATTERNS:
        raise ValueError("system must be 'cube2sph' or 'cart'")
    if system == 'cart':
        from pyproj import Proj

        if utm_zone is None or not 1 <= int(utm_zone) <= 60:
            raise ValueError('cart requires a UTM zone from 1 to 60')
        projection = Proj(proj='utm',zone=int(utm_zone),ellps='WGS84')
    elif utm_zone is not None:
        raise ValueError('UTM zone applies only to cart input')

    converted = []
    face_sources = []
    for proc, path in _processor_files(input_dir, system):
        points = _read_points(path, skip_header=(system == 'cart'),
                              allowed_columns=(6,) if system == 'cart' else (3,6))
        if len(points):
            if system == 'cube2sph':
                radius, latitude, longitude = cart2sph(*points[:, :3].T)
                depth_km = (EARTH_RADIUS_M - radius) / 1000.0
            else:
                longitude, geographic_latitude = projection(
                    points[:, 0], points[:, 1], inverse=True)
                latitude = geodetic_to_geocentric(geographic_latitude)
                depth_km = -points[:, 2] / 1000.0

            if not (np.isfinite(longitude).all() and np.isfinite(latitude).all()
                    and np.isfinite(depth_km).all()
                    and np.all((-90 <= latitude) & (latitude <= 90))
                    and np.all((0 <= depth_km) & (depth_km <= EARTH_RADIUS_M/1000))):
                raise ValueError(f'{path}: invalid converted longitude, latitude, or depth')
            rows = np.column_stack((longitude, latitude, depth_km))
        else:
            rows = np.empty((0, 3))

        converted.append(rows)
        face_sources.extend((proc, local_face)
                            for local_face in range(len(rows) // POINTS_PER_FACE))

    if not face_sources:
        raise ValueError(f'No boundary faces in {input_dir}')

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w') as stream:
        for rows in converted:
            np.savetxt(stream, rows, fmt='%.12g')
    return face_sources


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('system', choices=('cube2sph', 'cart'))
    parser.add_argument('input_dir', help='SPECFEM DATABASES_MPI directory')
    parser.add_argument('output_file', help='AxiSEM boundary_faces.dat path')
    parser.add_argument('--utm-zone', type=int, help='required for cart input')
    args = parser.parse_args()
    if args.system == 'cart' and args.utm_zone is None:
        parser.error('--utm-zone is required for cart input')
    if args.system == 'cube2sph' and args.utm_zone is not None:
        parser.error('--utm-zone applies only to cart input')
    sources = merge_surfaces(args.input_dir, args.output_file,
                             args.system, args.utm_zone)
    print(f'Wrote {len(sources)} faces to {args.output_file}')
