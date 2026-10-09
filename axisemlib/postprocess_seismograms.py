"""Write N/E/Z seismograms from AxiSEM's recorded receiver traces.

Usage: axisemlib seismogram RUN_DIR [--output-dir SEISMOGRAMS]

The run may be a single-source run or a four-run moment simulation. No
wavefield database or boundary-face file is needed.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


MOMENT_RUNS = ("MZZ", "MXX_P_MYY", "MXZ_MYZ", "MXY_MXX_M_MYY")


@dataclass
class RunInfo:
    source_type: str
    simulation_type: str
    source_colat: float
    source_lon: float
    magnitude: float
    nrec: int
    nt: int
    dt: float
    shift: float


def read_run_info(attributes):
    """Read source and seismogram metadata from the NetCDF root."""
    def string(name):
        value = attributes[name]
        return value.decode().strip() if isinstance(value, bytes) else str(value).strip()

    return RunInfo(
        source_type=string("source type"),
        simulation_type=string("simulation type"),
        source_colat=float(attributes["Source colatitude"]),
        source_lon=float(attributes["Source longitude"]),
        magnitude=float(attributes["scalar source magnitude"]),
        nrec=int(attributes["number of receivers"]),
        nt=int(attributes["length of seismogram  in time samples"]),
        dt=float(attributes["seismogram sampling in sec"]),
        shift=float(attributes["source shift factor in sec"]),
    )


def read_moment_tensor(path):
    """Return (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp) in N m from CMTSOLUTION."""
    values = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in ("Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp"):
            values[key.strip()] = float(value.split()[0].replace("D", "E")) * 1e-7
    return np.array([values[key] for key in ("Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp")])


def single_moment_tensor(source_type, magnitude):
    """Represent a single simulation's source amplitude as a moment tensor."""
    tensor = np.zeros(6)
    if source_type == "mrr":
        tensor[0] = magnitude
    elif source_type == "mtt_p_mpp":
        tensor[1:3] = magnitude
    elif source_type in ("mtr", "mrt"):
        tensor[3] = magnitude
    elif source_type in ("mpr", "mrp"):
        tensor[4] = magnitude
    elif source_type in ("mtp", "mpt"):
        tensor[5] = magnitude
    elif source_type == "mtt_m_mpp":
        tensor[1:3] = (magnitude, -magnitude)
    elif source_type == "explosion":
        tensor[:3] = magnitude
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
    return tensor


def read_receiver_names(dataset, nrec):
    """Decode NetCDF's fixed-width receiver names from HDF5 character data."""
    values = dataset[:]
    if values.ndim == 1:
        return [value.decode().strip(" \x00") if isinstance(value, bytes)
                else str(value).strip(" \x00") for value in values]
    # Fortran (receiver, character) dimensions are reversed in HDF5.
    if values.shape == (40, nrec):
        values = values.T
    elif values.shape != (nrec, 40):
        raise ValueError(f"Unexpected receiver_name shape: {values.shape}")
    return [b"".join(row).decode("utf-8").strip(" \x00") for row in values]


def read_recordings(path):
    """Return metadata, first recorded time, receiver positions, and traces."""
    with h5py.File(path) as nc:
        info = read_run_info(nc.attrs)
        group = nc["Seismograms"]
        first_time = float(group["time"][0])
        theta = np.deg2rad(group["theta"][:])
        phi = np.deg2rad(group["phi"][:])
        names = read_receiver_names(group["receiver_name"], len(theta))
        displacement = group["displacement"][:]

    # NetCDF Fortran (time, component, receiver) is HDF5 (receiver, component, time).
    if displacement.shape == (info.nrec, 3, info.nt):
        displacement = displacement.transpose(0, 2, 1)
    elif displacement.shape == (info.nt, 3, info.nrec):
        displacement = displacement.transpose(2, 0, 1)
    else:
        raise ValueError(f"{path}: unexpected displacement shape {displacement.shape}")
    if len(names) != info.nrec or len(phi) != info.nrec or len(theta) != info.nrec:
        raise ValueError(f"{path}: receiver dimensions disagree")
    return info, first_time, names, theta, phi, displacement


def radiation_prefactor(source_type, moment, magnitude, phi):
    """Azimuthal factors for the raw cylindrical (s, phi, z) traces."""
    if magnitude == 0:
        raise ValueError("Source magnitude must be nonzero")
    mrr, mtt, mpp, mrt, mrp, mtp = moment / magnitude
    if source_type == "mrr":
        return np.array((mrr, 0.0, mrr))
    if source_type == "mtt_p_mpp":
        return np.array((mtt + mpp, 0.0, mtt + mpp))
    if source_type in ("mtr", "mrt", "mpr", "mrp"):
        radial = mrt * np.cos(phi) + mrp * np.sin(phi)
        azimuthal = -mrt * np.sin(phi) + mrp * np.cos(phi)
        return np.array((radial, azimuthal, radial))
    if source_type in ("mtp", "mpt", "mtt_m_mpp"):
        radial = (mtt - mpp) * np.cos(2 * phi) + 2 * mtp * np.sin(2 * phi)
        azimuthal = (mpp - mtt) * np.sin(2 * phi) + 2 * mtp * np.cos(2 * phi)
        return np.array((radial, azimuthal, radial))
    if source_type == "explosion":
        return np.full(3, (mrr + mtt + mpp) / 3)
    raise ValueError(f"Unsupported source type: {source_type}")


def source_rotation(colat, lon):
    """Map source-at-pole Cartesian vectors into Earth Cartesian vectors."""
    ct, st = np.cos(colat), np.sin(colat)
    cp, sp = np.cos(lon), np.sin(lon)
    return np.array(((ct * cp, -sp, st * cp),
                     (ct * sp, cp, st * sp),
                     (-st, 0.0, ct)))


def rotate_nez(seis, theta, phi, source_colat, source_lon):
    """Rotate cylindrical traces into North, East, and vertical displacement."""
    xyz = np.empty_like(seis)
    xyz[:, 0] = np.cos(phi) * seis[:, 0] - np.sin(phi) * seis[:, 1]
    xyz[:, 1] = np.sin(phi) * seis[:, 0] + np.cos(phi) * seis[:, 1]
    xyz[:, 2] = seis[:, 2]
    rotation = source_rotation(source_colat, source_lon)
    xyz = xyz @ rotation.T

    location = rotation @ np.array(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))
    actual_theta = np.arccos(np.clip(location[2], -1.0, 1.0))
    actual_phi = np.arctan2(location[1], location[0])
    ct, st = np.cos(actual_theta), np.sin(actual_theta)
    cp, sp = np.cos(actual_phi), np.sin(actual_phi)
    radial = cp * xyz[:, 0] + sp * xyz[:, 1]
    north = -ct * radial + st * xyz[:, 2]
    east = -sp * xyz[:, 0] + cp * xyz[:, 1]
    vertical = st * radial + ct * xyz[:, 2]
    return north, east, vertical


def output_name(receiver_name):
    """Convert AxiSEM's STATION_NETWORK name to NETWORK.STATION."""
    station, separator, network = receiver_name.rpartition("_")
    return f"{network}.{station}" if separator else receiver_name


def process(run_dir, output_dir):
    moment_run = (run_dir / "MZZ" / "Data" / "axisem_output.nc4").is_file()
    subdirs = MOMENT_RUNS if moment_run else (".",)
    recordings = []
    for subdir in subdirs:
        path = run_dir / subdir / "Data" / "axisem_output.nc4"
        print(f"Reading {path}")
        recordings.append(read_recordings(path))

    first_info, first_time, names, theta, phi, _ = recordings[0]
    expected_type = "moment" if moment_run else "single"
    if any(recording[0].simulation_type != expected_type for recording in recordings):
        raise ValueError("NetCDF simulation type does not match the run directory")
    for info, time0, other_names, other_theta, other_phi, _ in recordings[1:]:
        if (info.nt != first_info.nt or info.nrec != first_info.nrec
                or not np.isclose(info.dt, first_info.dt)
                or not np.isclose(info.shift, first_info.shift)
                or not np.isclose(time0, first_time)
                or not np.isclose(info.source_colat, first_info.source_colat)
                or not np.isclose(info.source_lon, first_info.source_lon)
                or other_names != names
                or not np.allclose(other_theta, theta)
                or not np.allclose(other_phi, phi)):
            raise ValueError("Moment runs have inconsistent receiver or timing metadata")

    if moment_run:
        moment = read_moment_tensor(run_dir / "CMTSOLUTION")
    else:
        moment = single_moment_tensor(first_info.source_type, first_info.magnitude)

    # The NetCDF time variable starts at zero, while the source shift defines t0.
    # The seismogram sampling attribute gives dt (the stored time array may use
    # the solver step when recordings were decimated).
    t0 = first_time - first_info.shift
    time = t0 + np.arange(first_info.nt) * first_info.dt
    filenames = [output_name(name) for name in names]
    if any(not name or Path(name).name != name for name in filenames):
        raise ValueError("NetCDF contains an invalid receiver name")
    if len(set(filenames)) != len(filenames):
        raise ValueError("NetCDF contains duplicate receiver names")

    output_dir.mkdir(parents=True, exist_ok=True)
    for receiver, name in enumerate(filenames):
        cylindrical = np.zeros((first_info.nt, 3), dtype=float)
        for info, _, _, _, _, traces in recordings:
            cylindrical += traces[receiver] * radiation_prefactor(
                info.source_type, moment, info.magnitude, phi[receiver])
        north, east, vertical = rotate_nez(
            cylindrical, theta[receiver], phi[receiver],
            first_info.source_colat, first_info.source_lon)
        for component, values in (("E", east), ("N", north), ("Z", vertical)):
            np.savetxt(output_dir / f"{name}.BX{component}.dat",
                       np.column_stack((time, values)))

    print(f"Wrote {len(names)} receiver seismograms to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="AxiSEM run directory")
    parser.add_argument("--output-dir", type=Path, default=Path("SEISMOGRAMS"),
                        help="Output directory (default: ./SEISMOGRAMS)")
    args = parser.parse_args()
    process(args.run_dir.resolve(), args.output_dir)


if __name__ == "__main__":
    main()
