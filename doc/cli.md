# Command-line reference

Follow the [installation guide](installation.md) to clone the solver and
install the package. The default installation includes NumPy, SciPy, Numba,
h5py, pyproj, mpi4py, PyYAML, tqdm, and Matplotlib, covering every runtime
module. Use an environment whose MPI library matches the job launcher. The
package contains no compiled Python extension; Numba compiles numerical
kernels when first called. The AxiSEM solver is built from its separate
repository.

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

## `axisemlib model`

Generate a smoothed PREM or ak135 model for the AxiSEM mesher:

```bash
axisemlib model --model ak135 --sigma-km 5 --output-dir ../axisem/MESHER
```

The default model is PREM. The command writes `<model>.smooth.bm` for
`EXT_MODEL`, `<model>.txt` as a radial depth profile, and `smooth.jpg`.
Use `--ngll`, `--element-size-km`, or `--no-plot` to change the smoothing
grid or outputs. The default grid spacing is 1 km.

## Other installed commands

`axisemlib-prepare` creates one event's `CMTSOLUTION`, `STATIONS`, and
parameter files; see [the preparation guide](prepare_axisem_injection.md).

Use the commands above or import the Python API from `axisemlib`.
