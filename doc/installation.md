# Installation

## Clone the AxiSEM solver

Clone the [AxiSEM solver repository](https://github.com/nqdu/axisem) first.
The `AxiSEMLib` branch contains the boundary-face wavefield support used by
this package:

```bash
git clone --branch AxiSEMLib https://github.com/nqdu/axisem.git
```

Initialize the solver inputs from its templates before editing any parameter
files. `copytemplates.sh` replaces `make_axisem.macros` and the mesher and
solver input files:

```bash
cd axisem
./copytemplates.sh
```

## Install the Python package

Return to the parent directory, clone AxiSEMLib beside the `axisem` checkout,
then install it in a Python 3.10 or newer environment:

```bash
cd ..
git clone --branch devel https://github.com/nqdu/AxiSEMLib.git
cd AxiSEMLib
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

With these sibling checkouts, the solver is at `../axisem` relative to the
AxiSEMLib repository. Use the full solver path in configuration files when
running from another directory.

The default installation includes dependencies for MPI coupling, wavefield
transposition, and reciprocity plotting. Use an environment whose MPI library
matches the job launcher. Check the commands with `axisemlib --help` and
`axisemlib prepare --help`.
Wavefield transposition also needs `h5repack` on `PATH`.

## Build the AxiSEM mesh

Set `USE_NETCDF = true` and the correct `NETCDF_PATH` in
`../axisem/make_axisem.macros`, then
configure `../axisem/MESHER/inparam_mesh` for the model, period, and processor
count. The copied mesher template selects an external `ak135.smooth.bm` file.
Generate it from the AxiSEMLib checkout:

```bash
axisemlib smooth --model ak135 --output-dir ../axisem/MESHER
```

You can instead select a built-in model such as `BACKGROUND_MODEL ak135`.
Build and run the mesh from `MESHER`:

```bash
cd ../axisem/MESHER
./submit.csh
```

Wait until `OUTPUT` reports `DONE WITH MESHER`. Move the completed mesh to the
solver, then set `MESHNAME` in `SOLVER/inparam_basic` to the same name:

```bash
./movemesh.csh my_mesh
```

The solver is a separate Fortran program. Continue with the
[solver and coupling workflow](workflow.md) after the mesh is available.
AxiSEMLib's Python installation does not build the solver.

## Build these docs

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html -W --keep-going doc doc/_build/html
```

Open `doc/_build/html/index.html` after the build.
