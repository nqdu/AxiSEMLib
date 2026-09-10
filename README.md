
# AxiSEMLib

**AxiSEMLib** is a Python library designed to extend the capabilities of [AxiSEM](https://github.com/geodynamics/axisem). It facilitates high-precision seismic modeling,hybrid simulation integration, and efficient database management.

## Key Features

* **High-Precision Synthesis:** Generate accurate seismograms, strain, and stress tensors at any arbitrary point within the Earth.
* **Teleseismic Injection:** Seamless interfaces for wavefield injection between **AxiSEM**, [SPECFEM3D](https://github.com/SPECFEM/specfem3d), and [SPECFEM3D-injection](https://github.com/tianshi-liu/specfem3D-injection).
* **Reciprocity & Databases:** Tools for reciprocity validation and the creation of [Instaseis](https://github.com/krischer/instaseis)-style databases.
* **Enhanced AxiSEM Core:** Includes specific modifications to the original AxiSEM code:
    * Resolved source location inaccuracies.
    * Support for dumping elastic parameters in discontinuous forms.

> **Note on Licensing:** Parts of this codebase are adapted from [Instaseis](https://github.com/krischer/instaseis); therefore, this library is distributed under the **LGPL license**.

---

## Installation

### 1. System Requirements
Before installing the Python package, ensure your system has the following:
* **Compilers:** C++ and Fortran compilers supporting **C++14** (Tested on `GCC >=7.5` and `ICC >=18.4.0`).
* **Build Tools:** `cmake >= 3.12`.
* **Libraries:** MPI libraries (e.g., OpenMPI, MPICH).
* **External Dependencies:** * [HDF5](https://support.hdfgroup.org/HDF5)
    * [netcdf-fortran](https://docs.unidata.ucar.edu/netcdf-fortran/current/) (Serial version only).

### 2. Environment Setup
We recommend using a Conda environment to manage Python dependencies:

```bash
# Create and activate environment
conda create -n axisem_lib python=3.12
conda activate axisem_lib

# Install core dependencies
pip install numpy scipy numba pyproj tqdm pyyaml pybind11-global
```

### 3. Install external packages
To ensure the library functions correctly with parallel I/O and scientific data formats, install the following dependencies:

* **[HDF5](https://support.hdfgroup.org/HDF5):** Required for high-performance data management.
* **[netcdf-fortran](https://docs.unidata.ucar.edu/netcdf-fortran/current/):** Required for AxiSEM compatibility (**Note:** Use the serial version only).
* **[mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html):** Build this using your existing system MPI libraries to ensure consistent parallel execution:
    ```bash 
    MPICC=mpicc pip install mpi4py --no-binary mpi4py
    ```
* **[h5py](https://docs.h5py.org/en/stable/mpi.html):** Build this from source linked against your specific `HDF5` installation:
    ```bash
    HDF5_DIR=/path/to/your/hdf5 pip install --no-binary=h5py h5py
    ```

### 4. Build and Install AxiSEMLib
Finally, compile and install the library using `cmake`. Ensure that your environment is activated so that the correct Python path is detected.

```bash
# Create build directory and compile
mkdir -p build && cd build
cmake .. -DCXX=g++ -DFC=gfortran -DPYTHON_EXECUTABLE=$(which python)

# Build using 4 cores and install
make -j4
make install
```

## Prepare AxiSEM Mesh
- Go to directory `axisem` and change compiler options in `make_axisem.macros`. Remember to set `USE_NETCDF = true`, set `NETCDF_PATH`.

- Go to `MESHER/`, set parameters including `DOMINANT_PERIOD` and number of slices in `inparam_mesh`. Please make sure this dominant period is slightly shorter than the minimum period used in SEM(you can find it in `SPECFEM`'s `output_generate_databases.txt`) If you want to use a smoothed version of ak135/prem model, you can run scripts under `smooth_model/main.py`, and set the parameters like:
```bash
BACKGROUND_MODEL external
EXT_MODEL ak135.smooth.bm
```

- Run mesh generation `./submit.csh` and `./movemesh.csh mesh_name`. Then the mesh files will be moved to `SOLVER/MESHES` as the `mesh_name` you set.

## Prepare AxiSEM Mesh

1.  **Configure Macros:** Navigate to the `axisem` directory and update the compiler options in `make_axisem.macros`.
    * Ensure `USE_NETCDF = true`.
    * Set the correct `NETCDF_PATH`.

2.  **Configure Mesher:** Navigate to `MESHER/` and edit `inparam_mesh`.
    * Set `DOMINANT_PERIOD` and the number of slices. 
    * **Note:** Ensure the dominant period is slightly shorter than the minimum period used in the SEM (refer to `output_generate_databases.txt` in SPECFEM).
    * **Optional:** To use a smoothed version of the **ak135/prem** model, run the scripts in `smooth_model/main.py` and configure the parameters as follows:
        ```bash
        BACKGROUND_MODEL external
        EXT_MODEL ak135.smooth.bm
        ```

3.  **Generate Mesh:** Execute the generation and migration scripts:
    ```bash
    ./submit.csh
    ./movemesh.csh <mesh_name>
    ```
    The mesh files will be moved to `SOLVER/MESHES/<mesh_name>`.

---

## Prepare AxiSEM Solver Files

You must edit two primary configuration files: `inparam_basic` and `inparam_advanced`.

### 1. `inparam_basic`
* Set `SEISMOGRAM_LENGTH` to your requirements.
* Set `ATTENUATION` to `false` (the current version supports anisotropic elastic models only).
* Adjust `SIMULATION_TYPE`. If it is not set to `moment`, you must also edit `inparam_source`.

### 2. `inparam_advanced`
Configure the wavefield kernel parameters as follows:

```fortran
# GLL points to save (starting and ending indices)
KERNEL_IBEG         0
KERNEL_IEND         4
KERNEL_JBEG         0
KERNEL_JEND         4

KERNEL_WAVEFIELDS   true
KERNEL_DUMPTYPE     displ_only

# Samples per period (choose based on dominant frequency)
KERNEL_SPP          16  # or 16/32

# Time to start dumping
DUMP_T0             200. 

# Epicenter distance range
KERNEL_COLAT_MIN    25.
KERNEL_COLAT_MAX    100.

# Depth range (min/max radius in km)
KERNEL_RMIN         5000.
KERNEL_RMAX         6372.
```

### 3. Source and Station Setup
Prepare your `CMTSOLUTION` and `STATIONS` files. You can refer to the examples in `run_all_events.py`, which automates the submission for all events in the `CMT_DIR`. 

**Use geographic coordinates** for all latitudes (event latitude in `CMTSOLUTION`, station latitudes in `STATIONS`, and the study region passed to the script). `run_all_events.py` automatically converts them from geodetic to geocentric latitudes before they are passed to AxiSEM, which assumes a spherical Earth. The original files in `CMT_DIR` are left unchanged.
```bash
cd SOLVER
python run_all_events.py lonmin lonmax latmin latmax maxdep  # e.g. 114 132 41 46 450
```

## Run AxiSEM Simulation
Execute the simulation on your cluster according to your local scheduler (e.g., SLURM, PBS). Ensure your environment is correctly loaded before submission (`SOLVER/submit.csh`).

## Transpose Output Field
For significantly improved data access performance in post-processing, transpose the generated wavefield. 

1. Edit the variables in `submit_transpose.sh` to match your simulation paths.
2. Execute the script:
   ```bash
   bash submit_transpose.sh
   ```

## Getting Started with Examples

To explore the library's capabilities, navigate to the `EXAMPLES/` directory. Configuration is handled via a `YAML` file, which allows you to define paths, coupling methods, and time windows.

Two examples are provided, one for each SPECFEM coordinate system:

| Example | `SPECFEM_SYSTEM` | `coupling_method` | Target code |
|---|---|---|---|
| `EXAMPLES/specfem3d_cart` | `cart` | `wd` | SPECFEM3D (Cartesian, UTM) |
| `EXAMPLES/cube2sph` | `cube2sph` | `wd` | SPECFEM3D with Cube2sph (spherical, Earth-centered) |

Each example directory contains `param.yaml` (configuration) and `create_interfaces.sh` (SLURM job running `mpirun -np <N> python ../../run_coupling.py param.yaml`). The number of MPI ranks does not need to match the number of SPECFEM processors.

### Required Files

#### 1. AxiSEM database (`AXISEM_DIR`)
Both examples need the same AxiSEM output, i.e. the simulation directory created by `submit.csh` (e.g. `SOLVER/ak135.<event>`):
```
AXISEM_DIR/
├── CMTSOLUTION                       # source used by the simulation
├── MZZ/Data/axisem_output.nc4        # mesh and metadata are read from here
├── MZZ/Data/disp_{s,p,z}.bin         # transposed wavefield
├── MXX_P_MYY/Data/disp_{s,p,z}.bin
├── MXZ_MYZ/Data/disp_{s,p,z}.bin
└── MXY_MXX_M_MYY/Data/disp_{s,p,z}.bin
```
The `disp_*.bin` files are produced by the [Transpose Output Field](#transpose-output-field) step, so it must be run first.

#### 2. Boundary points (`SPECFEM_DB`)
These files describe the injection boundary and are read from `SPECFEM_DB` (usually `${SPECFEM_DIR}/DATABASES_MPI`). One file (or pair of files) is needed per SPECFEM processor.

**`specfem3d_cart`**: `proc??????_normal.txt`

Generated by SPECFEM3D in `LOCAL_PATH` when the mesh is created with `COUPLE_WITH_INJECTION_TECHNIQUE = .TRUE.` and `INJECTION_TYPE = 3` in `Par_file`. The first line is a header and is skipped; each following line is one boundary point:
```
x y z nx ny nz
```
- `x`, `y`: UTM coordinates in m, in the zone given by `UTM_ZONE`
- `z`: elevation in m (0 at the surface, negative below)
- `nx`, `ny`, `nz`: outward normal in (E, N, Z)

**`cube2sph`**: `proc??????_wavefield_discontinuity_points` and `proc??????_wavefield_discontinuity_faces`

Both use Earth-centered Cartesian coordinates in m, with no header line.
- `*_points`: unique boundary points, shape `(nbds, 3)`
  ```
  x y z
  ```
- `*_faces`: GLL points on each boundary face, shape `(nspec_bd * NGLL2, 6)`, with an outward normal
  ```
  x y z nx ny nz
  ```
  Files may be empty for processors that do not touch the boundary.

#### 3. Output (`OUTPUT_DIR`)
| Example | Output files | Content per time step |
|---|---|---|
| `specfem3d_cart` | `proc??????_sol_axisem` | velocity and traction |
| `cube2sph` | `proc??????_wavefield_discontinuity.bin` | displacement, acceleration and traction |

For `cube2sph` with `DOWN_SAMPLING: True`, `wavefield_discontinuity_info.txt` is also written with the new time step and number of steps.

### Example Configuration (`param.yaml`)

```yaml
# Parameter settings for generating the injection field

# Working Directories
AXISEM_DIR: ../axisem/SOLVER/ak135 
SPECFEM_DB: ~/SPECFEM/DATABASES_MPI    # Path to ${SPECFEM_DIR}/DATABASES_MPI
SPECFEM_DATA: ~/SPECFEM/DATA           # Only needed for equivalent forces coupling
SPECFEM_SYSTEM: 'cube2sph'             # Options: 'cart' or 'cube2sph'
OUTPUT_DIR: './OUTPUT_DIR'

# Signal Processing
DOWN_SAMPLING: False                    # If true, down-samples to AxiSEM mesh T0/2 

# Coupling Methods
coupling_method: 'wd'                  # Options: 'wd' (Wavefield Discontinuity) or 'ef' (Equivalent Forces)
UTM_ZONE: 10                           # UTM zone for cartesian systems (only if SYSTEM = 'cart')
only_eq_force: False                   # Set to true to only compute equivalent forces

# Time Window Settings
t0: 124.3                              # Starting time (t0 = 0 is the earthquake origin time)
dt: 0.025                              # Time step
nt: 1000                               # Number of time steps
```

> **Note:** `coupling_method: 'ef'` (Equivalent Forces, with `only_eq_force` and `SPECFEM_DATA`) is **experimental**. It is not covered by the examples and has not been fully tested. Use `'wd'` unless you know what you are doing.