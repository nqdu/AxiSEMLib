# AxiSEMLib

**AxiSEMLib** is a python library that provides several extensions for the [AxiSEM](https://github.com/geodynamics/axisem):

* Synthesize accurate seismograms/strain/stress at any point of the earth.
* Teleseismic injection interfaces between **AxiSEM**, [SPECFEM3D](https://github.com/SPECFEM/specfem3d) and [SPECFEM3D-injection](https://github.com/tianshi-liu/specfem3D-injection)
* Reciprocity validation and [Instaseis](https://github.com/krischer/instaseis)-like Database. 

And there are several modifications in [AxiSEM](https://github.com/geodynamics/axisem)
* Fix source location problem
* Dump elastic parameters in discontinuous form.

Part of the code are adapted from [Instaseis](https://github.com/krischer/instaseis), so LGPL license is applied.
 

## Download required packages
1. **Compilers:** C++/Fortran compilers which support c++14 (tested on `GCC >=7.5`, `ICC >=18.4.0`), `cmake >= 3.12`, and MPI libraries.

2. create a new environment with conda:
```bash
conda create -n axisem_lib python=3.8 
conda activate axisem_lib
conda install numpy scipy numba pyproj tqdm
pip install pybind11-global
```
3. Install several packages:
* [parallel-hdf5](https://support.hdfgroup.org/HDF5/PHDF5/) using your installed MPI libraries. 
* [netcdf-fortran](https://docs.unidata.ucar.edu/netcdf-fortran/current/), only serial version.
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html): You can build it by using existing mpi libraries:
```bash 
MPICC=mpicc pip install mpi4py --no-binary mpi4py
```
* [h5py-mpi](https://docs.h5py.org/en/stable/mpi.html) using existing `parallel-hdf5` libraries:
```bash
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-binary=h5py h5py
```

4. build `AxiSEMLib` by using:
```bash
mkdir -p build; cd build;
cmake .. -DCXX=g++ -DFC=gfortran  -DPYTHON_EXECUTABLE=`which python`
make -j4; make install 
```

## Prepare AxiSEM Mesh
- Go to directory `axisem` and change compiler options in `make_axisem.macros`. Remember to set `USE_NETCDF = true`, set `NETCDF_PATH`.

- Go to `MESHER/`, set parameters including `DOMINANT_PERIOD` and number of slices in `inparam_mesh`. If you want to use a smoothed version of ak135/prem model, you can run scripts under `smooth_model/main.py`, and set the parameters like:
```bash
BACKGROUND_MODEL external
EXT_MODEL ak135.smooth.bm
```

- Run mesh generation `./submit.csh` and `./movemesh.csh mesh_name`. Then the mesh files will be moved to `SOLVER/MESHES` as the `mesh_name` you set.

## Prepare AxiSEM Solver files
There are two files you should edit: `inparam_basic` and `inparam_advanced`.

In `param_basic` you should set `SEISMOGRAM_LENGTH` as you required, and you should set `ATTENUATION` to `false` because the current version only support isotropic elastic model. And you can set some other parameters like `SIMULATION_TYPE`. If `SIMULATION_TYPE` is not `moment`, you should also edit `inparam_source`.

In `inparam_advanced`, you should set the part of several parameters as below:
```bash 
# GLL points to save, starting and ending GLL point index 
# (overwritten with 0 and npol for dumptype displ_only)
KERNEL_IBEG         0
KERNEL_IEND         4
KERNEL_JBEG         0
KERNEL_JEND         4

KERNEL_WAVEFIELDS   true
KERNEL_DUMPTYPE     displ_only
KERNEL_SPP          8/16/32 (depend on your dominant frequency)

# you should add this one 
# KERNEL  dump  after DUMP_T0
DUMP_T0       200. 

# epicenter distance
KERNEL_COLAT_MIN   25.
KERNEL_COLAT_MAX   100.

# minimal and maximal radius in km for kernel wavefields
# (only for dumptype displ_only)
KERNEL_RMIN        5000.
KERNEL_RMAX        6372.
```
Then you can prepare your `CMTSOLUTION` and `STATIONS`. You can follow examples in `run_all_events.sh`, which will submit
all events in `CMT_DIR`

## Run AxiSEM simulation
Run it on your cluster.

## Transpose the output field for better performance.
Set the variables in `submit_transpose.sh`, then:
```bash
bash submit_transpose.sh 
```

## Try examples in `EXAMPLES/` !
