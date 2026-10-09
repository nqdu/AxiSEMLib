"""Transpose AxiSEM wavefields into float32 binary files."""

import os
from pathlib import Path
import subprocess
import sys

import h5py
from mpi4py import MPI
import numpy as np
from tqdm import tqdm

SOURCE_DIRS = ("MZZ", "MXZ_MYZ", "MXY_MXX_M_MYY", "MXX_P_MYY", "PX", "PY", "PZ")
STANDARD_FIELDS = ("disp_s", "disp_p", "disp_z")
BOUNDARY_FIELDS = ("disp_s", "disp_p", "disp_z", "chi")


def write_trans_data_dof(infile: Path, field: str, out_dir: Path, size_gb: float) -> None:
    """Transpose Snapshots/(time, point) to (point, time)."""
    with h5py.File(infile, "r") as source:
        values = source[f"Snapshots/{field}"]
        nt, npts = values.shape
        points_per_block = max(1, int(size_gb * 1024**3 / (2 * nt * np.dtype("f4").itemsize)))
        output = np.memmap(out_dir / f"{field}.bin", dtype="f4", mode="w+", shape=(npts, nt))
        try:
            for first in tqdm(range(0, npts, points_per_block), desc=str(infile.name) + "/" + field):
                last = min(first + points_per_block, npts)
                output[first:last] = np.asarray(values[:, first:last], dtype="f4").T
            output.flush()
        finally:
            del output


def write_trans_data_boundary(infile: Path, field: str, out_dir: Path, size_gb: float) -> None:
    """Transpose (time, unique_element, i, j) to (unique_element, i, j, time)."""
    with h5py.File(infile, "r") as source:
        values = source[field]
        nt, nelem, ni, nj = values.shape
        elements_per_block = max(
            1, int(size_gb * 1024**3 / (2 * nt * ni * nj * np.dtype("f4").itemsize))
        )
        output = np.memmap(out_dir / f"{field}.bin", dtype="f4", mode="w+", shape=(nelem, ni, nj, nt))
        try:
            for first in tqdm(range(0, nelem, elements_per_block), desc=str(infile.name) + "/" + field):
                last = min(first + elements_per_block, nelem)
                block = np.asarray(values[:, first:last, :, :], dtype="f4")
                output[first:last] = np.moveaxis(block, 0, -1)
            output.flush()
        finally:
            del output


def find_inputs(path: Path):
    """Use boundary data when available in each Data directory."""
    if path.is_file():
        if path.name in ("boundary_wavefields.nc4", "axisem_output.nc4"):
            yield path
        return
    data_dirs = [path, path / "Data"]
    data_dirs.extend(path / source / "Data" for source in SOURCE_DIRS)
    for data_dir in data_dirs:
        for name in ("boundary_wavefields.nc4", "axisem_output.nc4"):
            candidate = data_dir / name
            if candidate.is_file():
                yield candidate
                break


def repack_after_removing_fields(infile: Path, fields: set[str], boundary: bool) -> None:
    prefix = "" if boundary else "Snapshots/"
    with h5py.File(infile, "a") as source:
        for field in fields:
            print(f"Deleting {prefix}{field} from {infile}", flush=True)
            del source[prefix + field]
    backup = infile.with_name(infile.name + ".bak")
    os.replace(infile, backup)
    try:
        subprocess.run(["h5repack", str(backup), str(infile)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if infile.exists():
            infile.unlink()
        os.replace(backup, infile)
        raise
    backup.unlink()


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) < 2:
        print("Usage: axisemlib transpose sizeGB_per_rank RUN_DIR [RUN_DIR ...]", file=sys.stderr)
        raise SystemExit(1)
    try:
        size_gb_per_rank = float(arguments[0])
    except ValueError:
        print("sizeGB_per_rank must be a positive number", file=sys.stderr)
        raise SystemExit(1)
    if size_gb_per_rank <= 0:
        print("sizeGB_per_rank must be a positive number", file=sys.stderr)
        raise SystemExit(1)

    jobs: list[tuple[Path, str, bool]] = []
    seen: set[Path] = set()
    for path in arguments[1:]:
        directory = Path(path)
        for infile in find_inputs(directory):
            resolved = infile.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            boundary = infile.name == "boundary_wavefields.nc4"
            fields = BOUNDARY_FIELDS if boundary else STANDARD_FIELDS
            with h5py.File(infile, "r") as source:
                group = source if boundary else source.get("Snapshots")
                if group is None:
                    continue
                for field in fields:
                    if field in group:
                        jobs.append((infile, field, boundary))

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    failures = []
    for index in range(rank, len(jobs), size):
        infile, field, boundary = jobs[index]
        print(f"Rank {rank}: transposing {infile}/{field}", flush=True)
        try:
            if boundary:
                write_trans_data_boundary(infile, field, infile.parent, size_gb_per_rank)
            else:
                write_trans_data_dof(infile, field, infile.parent, size_gb_per_rank)
        except Exception as exc:
            failures.append(f"{infile}/{field}: {exc}")
    all_failures = comm.allgather(failures)
    if any(all_failures):
        for rank_failures in all_failures:
            for failure in rank_failures:
                print(f"Transpose failed: {failure}", flush=True)
        raise SystemExit(1)

    repack_failure = None
    if rank == 0:
        try:
            for infile in sorted({infile for infile, _, _ in jobs}):
                repack_after_removing_fields(
                    infile,
                    {field for path, field, _ in jobs if path == infile},
                    infile.name == "boundary_wavefields.nc4",
                )
        except Exception as exc:
            repack_failure = str(exc)
    repack_failure = comm.bcast(repack_failure, root=0)
    if repack_failure:
        raise SystemExit(f"Repack failed: {repack_failure}")


if __name__ == "__main__":
    main()
