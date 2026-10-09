# Command-line reference

Follow the [installation guide](installation.md) to clone the solver and
install the package. The base installation
includes NumPy, SciPy, Numba, h5py, and pyproj. It contains no compiled Python
extension; Numba compiles numerical kernels when they are first called.

MPI coupling also needs `mpi4py` and PyYAML. Wavefield transposition needs
`mpi4py` and tqdm. Install both with
`python -m pip install -e ".[coupling,transpose]"` in an environment whose MPI library matches the one
used to launch the jobs. `.[all]` also installs Matplotlib for reciprocity
plots. The AxiSEM solver itself is built from the separate axisem repository.

## `axisemlib merge`

```bash
axisemlib merge cube2sph SPECFEM_DB ../axisem/SOLVER/boundary_faces.dat
axisemlib merge cart SPECFEM_DB ../axisem/SOLVER/boundary_faces.dat --utm-zone 10
```

The command reads processor files in numeric order and writes 25 rows per
face as `longitude latitude depth_km`. The `cart` input uses UTM coordinates
and requires a zone. See [the workflow guide](workflow.md#run-axisem-simulation)
for input filenames and coordinate conventions.

## `axisemlib seismogram`

```bash
axisemlib seismogram AXISEM_RUN --output-dir SEISMOGRAMS
```

For a four-part moment run, it reads each part's `Data/axisem_output.nc4`
and the run's `CMTSOLUTION`. For a single-source run, it reads the run's own
NetCDF file. It sums and rotates recorded receiver traces to North, East, and
vertical. The default output directory is `./SEISMOGRAMS`; files are named
`NETWORK.STATION.BXN.dat`, `.BXE.dat`, and `.BXZ.dat`.

## `axisemlib coupling`

```bash
mpirun -n 8 axisemlib coupling param.yaml
```

The YAML file supplies the AxiSEM run, SPECFEM database, output directory,
time window, coordinate system, and coupling method. Example configurations
are in `EXAMPLES/cube2sph` and `EXAMPLES/specfem3d_cart`. See
[the workflow guide](workflow.md#getting-started-with-examples) for the file
formats and output quantities.

## `axisemlib transpose`

```bash
mpirun -n 8 axisemlib transpose 2.0 AXISEM_RUN [ANOTHER_RUN ...]
```

The first argument is the read buffer size in GiB per rank. The command writes
element-major boundary fields or standard DOF fields as binary files, removes
the converted fields from NetCDF, then uses `h5repack` on `PATH` to repack the
files.

## Other installed commands

`axisemlib-prepare` creates one event's `CMTSOLUTION`, `STATIONS`, and
parameter files; see [the preparation guide](prepare_axisem_injection.md).

Use the commands above or import the Python API from `axisemlib`.
