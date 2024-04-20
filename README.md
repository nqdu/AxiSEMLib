# AxiSEMLib

**AxiSEMLib** is a python library that enables several extensional functionalies of the [AxiSEM](https://github.com/geodynamics/axisem):

* Synthesize accurate seismograms/strain/stress at any points of the earth.
* Teleseismic injection interfaces between **AxiSEM**, [SPECFEM3D](https://github.com/SPECFEM/specfem3d) and [SPECFEM3D-injection](https://github.com/tianshi-liu/specfem3D-injection)
 

## This is the test version, user manual and examples will be updated later ...

## Download required packages
1. **Compilers:** c++/Fortran Compilers which support c++14 (tested on `GCC >=7.5`, `ICC >=19.2.0`), `cmake >= 3.12`, and MPI libraries.

2. create a new environment with conda:
```bash
conda create -n axisem_lib python=3.8 
conda activate axisem_lib
conda install numpy scipy numba
conda install -c conda-forge pybind11
```
3. Install [parallel-hdf5](https://support.hdfgroup.org/HDF5/PHDF5/), [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html) and [h5py-mpi](https://docs.h5py.org/en/stable/mpi.html), and .

4. build `AxiSEMLib` by using:
```bash
mkdir -p build; cd build;
cmake .. -DCXX=g++ -DFC=gfortran  -DPYTHON_EXECUTABLE=`which python`
make -j4; make install 
```

## Download AxiSEM-1.4
Download version 1.4 of [axisem](https://github.com/geodynamics/axisem). And install all required libraries from it's manual.

Ssubstitute `nc_routines.F90` in `axisem/SOLVER/` by the same file in `nc_routines.tar.gz`.

## Prepare AxiSEM Files
In `SOLVER/inparam_advanced`, you should set the several parameters:
```bash 
KERNEL_WAVEFIELDS   true
KERNEL_DUMPTYPE     displ_only
KERNEL_SPP          8/16/32 (depend on your dominant frequency)

# epicenter distance
KERNEL_COLAT_MIN   25.
KERNEL_COLAT_MAX   100.

# minimal and maximal radius in km for kernel wavefields
# (only for dumptype displ_only)
KERNEL_RMIN        5000.
KERNEL_RMAX        6372.
```
Then you can edit several files: `inparam_basic`, `inparam_source`,`CMTSOLUTION`,`STATIONS`

## Run AxiSEM simulation
Build **AxiSEM** with `USE_NETCDF` mode, and run it on your cluster. 

## Transpose the output field.
Set the variables in `submit_transpose.sh`, then:
```bash
bash submit_transpose.sh 
```

## Check examples in `test/` !
