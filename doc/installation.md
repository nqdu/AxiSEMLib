# Installation

## Clone the AxiSEM solver

Clone the [AxiSEM solver repository](https://github.com/nqdu/axisem) first.
The `AxiSEMLib` branch contains the boundary-face wavefield support used by
this package:

```bash
git clone --branch AxiSEMLib https://github.com/nqdu/axisem.git
```

The solver is a separate Fortran program. Configure its
`make_axisem.macros`, build the mesh, and run the solver as described in the
[solver and coupling workflow](workflow.md). AxiSEMLib's Python installation
does not build the solver.

## Install the Python package

Clone AxiSEMLib beside the `axisem` checkout, then install it in a Python
3.10 or newer environment:

```bash
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
`axisemlib-prepare --help`.
Wavefield transposition also needs `h5repack` on `PATH`.

## Build these docs

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html -W --keep-going doc doc/_build/html
```

Open `doc/_build/html/index.html` after the build.
