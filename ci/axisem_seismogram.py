"""Build and compare a four-part AxiSEM moment run in CI."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

MOMENT_PARTS = ("MZZ", "MXX_P_MYY", "MXZ_MYZ", "MXY_MXX_M_MYY")
MESH_NAME = "ak135.ci50"
RUN_NAME = "ak135.ci50-moment"


def set_parameter(path: Path, name: str, value: str, *, equals: bool = False) -> None:
    """Replace exactly one active assignment in an AxiSEM input file."""
    separator = r"\s*=\s*" if equals else r"\s+"
    pattern = re.compile(rf"^{re.escape(name)}{separator}.*$", re.MULTILINE)
    contents, count = pattern.subn(f"{name}{' = ' if equals else ' '}{value}", path.read_text())
    if count != 1:
        raise ValueError(f"Expected one active {name} entry in {path}; found {count}")
    path.write_text(contents)


def run_command(*command: str, cwd: Path) -> None:
    print(f"Running in {cwd}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def configure(axisem: Path) -> None:
    """Initialize the documented templates for a small 50 s moment run."""
    run_command("bash", "copytemplates.sh", cwd=axisem)

    macros = axisem / "make_axisem.macros"
    for name, value in {
        "USE_NETCDF": "true",
        "USE_PAR_NETCDF": "false",
        "SERIAL": "true",
        "FC": "gfortran",
        "FFLAGS": "-O2 -cpp -fallow-argument-mismatch -ffree-line-length-none -fbacktrace",
        "CFLAGS": "-O2 -Wno-implicit-function-declaration",
        "LDFLAGS": "-O2",
    }.items():
        set_parameter(macros, name, value, equals=True)

    mesh = axisem / "MESHER" / "inparam_mesh"
    for name, value in {
        "BACKGROUND_MODEL": "external",
        "EXT_MODEL": "ak135.smooth.bm",
        "DOMINANT_PERIOD": "50.0",
        "NTHETA_SLICES": "1",
        "NRADIAL_SLICES": "1",
        "WRITE_VTK": "false",
    }.items():
        set_parameter(mesh, name, value)

    solver = axisem / "SOLVER"
    basic = solver / "inparam_basic"
    for name, value in {
        "SIMULATION_TYPE": "moment",
        "SEISMOGRAM_LENGTH": "300.0",
        "RECFILE_TYPE": "stations",
        "MESHNAME": MESH_NAME,
        "ATTENUATION": "false",
    }.items():
        set_parameter(basic, name, value)

    advanced = solver / "inparam_advanced"
    for name, value in {
        "USE_NETCDF": "true",
        "KERNEL_WAVEFIELDS": "false",
        "SAVE_BDRY_FACES": "false",
        "DUMP_T0": "0.0",
    }.items():
        set_parameter(advanced, name, value)

    (solver / "STATIONS").write_text(
        "CI01 XX 40.0 -75.0 0.0 0.0\n"
        "CI02 XX 34.0 -80.0 0.0 0.0\n"
        "CI03 XX 38.0 -83.0 0.0 0.0\n"
    )
    print("Configured external ak135 50 s mesh and four-part moment run", flush=True)


def wait_for_outputs(paths: list[Path], marker: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        finished = [path for path in paths if path.is_file() and marker in path.read_text(errors="replace")]
        if len(finished) == len(paths):
            print(f"Completed: {', '.join(path.parent.name for path in paths)}", flush=True)
            return
        print(f"Waiting for {marker}: {len(finished)}/{len(paths)} complete", flush=True)
        time.sleep(20)
    for path in paths:
        print(f"--- {path} ---", flush=True)
        print(path.read_text(errors="replace")[-4000:] if path.exists() else "missing", flush=True)
    raise TimeoutError(f"Timed out waiting for {marker}")


def mesh(axisem: Path) -> None:
    mesher = axisem / "MESHER"
    if not (mesher / "ak135.smooth.bm").is_file():
        raise FileNotFoundError("Generate ak135.smooth.bm before running the mesher")
    run_command("csh", "submit.csh", cwd=mesher)
    if not (mesher / "xmesh").is_file():
        raise RuntimeError("Mesher compilation did not create xmesh")
    wait_for_outputs([mesher / "OUTPUT"], "DONE WITH MESHER", 1200)
    run_command("csh", "movemesh.csh", MESH_NAME, cwd=mesher)
    if not list((axisem / "SOLVER" / "MESHES" / MESH_NAME).glob("meshdb.dat*")):
        raise RuntimeError("Completed mesh was not moved to SOLVER/MESHES")


def solver(axisem: Path) -> None:
    solver_dir = axisem / "SOLVER"
    # Build UTILS explicitly: its target name collides with the directory on
    # case-insensitive filesystems, and submit.csh needs xpost_processing.
    run_command("make", "-j4", "all", cwd=solver_dir / "UTILS")
    run_command("make", "-j4", cwd=solver_dir)
    for binary in (solver_dir / "axisem", solver_dir / "UTILS" / "xpost_processing"):
        if not binary.is_file():
            raise RuntimeError(f"Build did not create {binary}")
    run_command("csh", "submit.csh", RUN_NAME, cwd=solver_dir)
    run = solver_dir / RUN_NAME
    if not (run / "xpost_processing").is_file():
        raise RuntimeError("submit.csh did not copy xpost_processing into the run")
    logs = [run / part / f"OUTPUT_{part}" for part in MOMENT_PARTS]
    wait_for_outputs(logs, "FINISHED", 2400)
    for part in MOMENT_PARTS:
        output = run / part / "Data" / "axisem_output.nc4"
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Missing NetCDF seismograms: {output}")


def read_trace(path: Path) -> np.ndarray:
    import numpy as np

    values = np.loadtxt(path, ndmin=2)
    if values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid two-column seismogram: {path}")
    return values


def compare(axisem: Path) -> None:
    import numpy as np

    run = axisem / "SOLVER" / RUN_NAME
    params = run / "param_post_processing"
    if not params.is_file():
        raise FileNotFoundError(f"Solver did not write {params}")
    for name, value in {
        "REC_COMP_SYS": "enz",
        "CONV_PERIOD": "0.",
        "LOAD_SNAPS": "F",
        "DATA_DIR": '"./Data_Postprocessing"',
        "NEGATIVE_TIME": "T",
    }.items():
        set_parameter(params, name, value)

    run_command("csh", "post_processing.csh", cwd=run)
    legacy_dir = run / "Data_Postprocessing" / "SEISMOGRAMS"
    if not list(legacy_dir.glob("*_disp_post_mij_conv*_N.dat")):
        raise RuntimeError("post_processing.csh produced no NEZ seismograms")

    python_dir = run / "PYTHON_SEISMOGRAMS"
    run_command("axisemlib", "seismogram", str(run), "--output-dir", str(python_dir), cwd=run)

    traces = sorted(legacy_dir.glob("*_disp_post_mij_conv*_?.dat"))
    if len(traces) != 9:
        raise AssertionError(f"Expected three receivers × NEZ, found {len(traces)} traces")
    global_peak = max(float(np.max(np.abs(read_trace(path)[:, 1]))) for path in traces)
    if global_peak <= 0:
        raise AssertionError("All legacy seismograms are zero")

    report = []
    failures = []
    for legacy_path in traces:
        receiver, separator, suffix = legacy_path.name.partition("_disp_post_mij_conv")
        if not separator or not suffix.endswith(("_N.dat", "_E.dat", "_Z.dat")):
            raise ValueError(f"Unexpected legacy filename: {legacy_path.name}")
        station, separator, network = receiver.rpartition("_")
        if not separator:
            raise ValueError(f"Missing station/network separator: {legacy_path.name}")
        component = suffix[-5]
        python_path = python_dir / f"{network}.{station}.BX{component}.dat"
        old = read_trace(legacy_path)
        new = read_trace(python_path)
        if old.shape != new.shape:
            raise AssertionError(f"Sample count differs: {legacy_path.name}: {old.shape} vs {new.shape}")
        time_error = float(np.max(np.abs(old[:, 0] - new[:, 0])))
        amplitude_error = float(np.max(np.abs(old[:, 1] - new[:, 1])))
        trace_scale = max(float(np.max(np.abs(old[:, 1]))), global_peak * 1e-4)
        relative_error = amplitude_error / trace_scale
        report.append(
            f"{network}.{station}.BX{component}: samples={len(old)} "
            f"time_error={time_error:.3e} amplitude_error={amplitude_error:.3e} "
            f"relative_to_trace_scale={relative_error:.3e}"
        )
        if time_error > 5e-4 or relative_error > 1e-3:
            failures.append(report[-1])

    output = run / "seismogram-comparison.txt"
    output.write_text("\n".join(report) + "\n")
    print(output.read_text(), flush=True)
    if failures:
        raise AssertionError("Legacy/Python mismatch:\n" + "\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("configure", "mesh", "solver", "compare"))
    parser.add_argument("axisem", type=Path, help="AxiSEM checkout")
    args = parser.parse_args()
    axisem = args.axisem.resolve()
    {"configure": configure, "mesh": mesh, "solver": solver, "compare": compare}[args.phase](axisem)


if __name__ == "__main__":
    main()
