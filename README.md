# AxiSEMLib

**AxiSEMLib** is a python library that provides several extensionals for the [AxiSEM](https://github.com/geodynamics/axisem):

* Synthesize accurate seismograms/strain/stress at any points of the earth.
* Teleseismic injection interfaces between **AxiSEM**, [SPECFEM3D](https://github.com/SPECFEM/specfem3d) and [SPECFEM3D-injection](https://github.com/tianshi-liu/specfem3D-injection)

Part of the code are adapted from [Instaseis](https://github.com/krischer/instaseis), so LGPL license is applied.
 

## 1. Download required packages
1. **Compilers:** c++/Fortran Compilers which support c++14 (tested on `GCC >=7.5`, `ICC >=19.2.0`), `cmake >= 3.12`, and MPI libraries.

2. create a new environment with conda:
```bash
conda create -n axisem_lib python=3.8 
conda activate axisem_lib
conda install numpy scipy numba pyproj
pip install pybind11-global
```
3. Install [parallel-hdf5](https://support.hdfgroup.org/HDF5/PHDF5/), [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html) and [h5py-mpi](https://docs.h5py.org/en/stable/mpi.html), and [netcdf-fortran](https://docs.unidata.ucar.edu/netcdf-fortran/current/).

4. build `AxiSEMLib` by using:
```bash
mkdir -p build; cd build;
cmake .. -DCXX=g++ -DFC=gfortran  -DPYTHON_EXECUTABLE=`which python`
make -j4; make install 
```

## 2. Prepare AxiSEM Mesh
- Go to directory `axisem` and change compiler options in `make_axisem.macros`. Remember to set `USE_NETCDF = true`, set `NETCDF_PATH`.

- Go to `MESHER/`, set parameters including `DOMINANT_PERIOD` and number of slices in `inparam_mesh`. If you want to use a smoothed version of ak135 model, you can run scripts under `smooth_model/ak135.py`, and set the parameters like:
```bash
BACKGROUND_MODEL external
EXT_MODEL ak135.smooth.bm
```

- Run mesh generation `./submit.csh` and `./movemesh.csh mesh_name`. Then the mesh files will be moved to `SOLVER/MESHES` as the `mesh_name` you set.

## Prepare AxiSEM Solver files
There are two files you should edit: `inparam_basic` and `inparam_advanced`.

In `param_basic` you should set `SEISMOGRAM_LENGTH` as you required, and you should set `ATTENUATION` to `false` because the current version only support isotropic model. And you can set some other parameters like `SIMULATION_TYPE`. If `SIMULATION_TYPE` is not `moment`, you should also edit `inparam_source`.

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
# KERNEL  dump  after KERNEL_T0
KERNEL_T0       200. 

# epicenter distance
KERNEL_COLAT_MIN   25.
KERNEL_COLAT_MAX   100.

# minimal and maximal radius in km for kernel wavefields
# (only for dumptype displ_only)
KERNEL_RMIN        5000.
KERNEL_RMAX        6372.
```
Then you can prepare your `CMTSOLUTION` and `STATIONS`

## Run AxiSEM simulation
Run it on your cluster.

## Transpose the output field.
Set the variables in `submit_transpose.sh`, then:
```bash
bash submit_transpose.sh 
```

## Check examples in `test/` !
