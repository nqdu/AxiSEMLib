# AXISEM-SPECFEM3D Coupling Notes

## This is the test version, use manual and examples will be updated later ...

## step1 : Download axisem-1.4
Download version 1.4 of [axisem](https://github.com/geodynamics/axisem). 

If you want to use the fast version, you should substitute `nc_routines.F90` in `axisem/SOLVER/` with that in `src/`.

## step2: download required software
1. create a new environment with conda:
```bash
conda create -n axisem_lib python=3.8 
conda activate axisem_lib
conda install numpy scipy numba mpi4py
conda install -c conda-forge pybind11  
```
2. You should install [parallel-hdf5](https://support.hdfgroup.org/HDF5/PHDF5/), and [h5py](https://docs.h5py.org/en/stable/mpi.html). Then you can change the `EIGEN_INC` in `compile.sh`, and compile this package by :
```bash
bash compie.sh
```


## step3: Prepare AXISEM Files
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
Then you can edit several files: `inparam_basic`, `inparam_source`,`CMTSOLUTION`

## step4: run simulation
Build axisem with `USE_NETCDF` mode, and run it.

## step5: Prepare Boundary Points
your should provide two files `proc*_wavefield_discontinuity_faces` and `proc*_wavefield_discontinuity_points`. The first file is with format: 
```bash
x y z
```
where `x/y/z` are Cartesian coordinates for each boundary points with dimension `m`. Note that this file should be of shape (nbds), only keep unique points. For second:
```bash
x y z nx ny nz
```
where `x/y/z` are Cartesian coordinates for each boundary points with dimension `m` and `n_x/y/z` are the normal vector ouside the study region. It should be of the shape (nelmnts,NGLL3)

## step7 transpose the output field.
change the variables in `submit_transpose.sh`, then:
```bash
bash submit_transpose.sh 
```

## step8 compute coupling field:
change the variables in `bash submit_prepare_fields.sh`, then:
```bash
bash submit_prepare_fields.sh 
```
