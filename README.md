# AxiSEMLib

AxiSEMLib is a Python companion to AxiSEM. It reads AxiSEM wavefields,
produces receiver seismograms, prepares models and injection inputs, and
builds SPECFEM coupling files.

[Read the documentation](https://nqdu.github.io/AxiSEMLib/).

## Install

Follow the [installation guide](doc/installation.md) to clone the
[AxiSEM solver](https://github.com/nqdu/axisem), run `copytemplates.sh`,
generate a mesh, and install this package. Python 3.10 or newer is required.
From the AxiSEMLib checkout:

```bash
python -m pip install -e .
```

The default installation includes coupling, transposition, and reciprocity
routines. Use an environment with an MPI library available.

## Commands

```bash
axisemlib merge cube2sph SPECFEM_DB ../axisem/SOLVER/boundary_faces.dat
axisemlib seismogram AXISEM_RUN --output-dir SEISMOGRAMS
axisemlib transpose 2.0 AXISEM_RUN
axisemlib smooth --model ak135 --output-dir ../axisem/MESHER
axisemlib prepare --region-box 114 132 41 46 450 --t0-injection 1166.926174
mpirun -n 8 axisemlib coupling param.yaml
```

`merge cart` also requires `--utm-zone`. Seismograms are written as
`NETWORK.STATION.BX{N,E,Z}.dat`. The coupling command reads paths and time
settings from its YAML file. Run `axisemlib --help` or a subcommand's `--help`
for arguments. `axisemlib prepare` prepares one event's solver inputs.

The Python API starts with:

```python
from axisemlib import AxiBasicDB
```

## Guides

- [Sphinx documentation](doc/index.md)
- [Installation](doc/installation.md)
- [Solver, boundary-face, and coupling workflow](doc/workflow.md)
- [Command-line reference and dependencies](doc/cli.md)
- [Prepare one injection event](doc/prepare_axisem_injection.md)

The AxiSEM solver is maintained separately at
[nqdu/axisem](https://github.com/nqdu/axisem). This repository is licensed
under the [GNU LGPL v3 or later](license.md).
