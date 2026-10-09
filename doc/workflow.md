# AxiSEM solver and coupling workflow

Start with the [installation guide](installation.md) to clone the AxiSEM
solver and install AxiSEMLib. The commands below assume sibling `axisem` and
`AxiSEMLib` checkouts unless another working directory is stated.

## Prepare AxiSEM Mesh

1.  **Copy the templates:** From the AxiSEMLib checkout, initialize the
    solver's default input files before changing them:

    ```bash
    cd ../axisem
    ./copytemplates.sh
    cd ../AxiSEMLib
    ```

    This replaces `make_axisem.macros`, `MESHER/inparam_mesh`, and the solver
    parameter files.

2.  **Configure Macros:** Update the compiler options in
    `../axisem/make_axisem.macros`.
    * Ensure `USE_NETCDF = true`.
    * Set the correct `NETCDF_PATH`.

3.  **Configure Mesher:** Edit `../axisem/MESHER/inparam_mesh`.
    * Set `DOMINANT_PERIOD` and the number of slices.
    * **Note:** Ensure the dominant period is slightly shorter than the minimum period used in the SEM (refer to `output_generate_databases.txt` in SPECFEM).
    * The copied template selects `BACKGROUND_MODEL external` and
      `EXT_MODEL ak135.smooth.bm`. Put that file in `MESHER`, or use a built-in
      model such as `BACKGROUND_MODEL ak135`.
    * If you have a smoothed **ak135/prem** model, configure it as an external model:
        ```bash
        BACKGROUND_MODEL external
        EXT_MODEL ak135.smooth.bm
        ```

4.  **Generate Mesh:** Run the mesher:

    ```bash
    cd ../axisem/MESHER
    ./submit.csh
    ```

    Wait for `OUTPUT` to report `DONE WITH MESHER`, then move the completed
    mesh:

    ```bash
    ./movemesh.csh <mesh_name>
    cd ../../AxiSEMLib
    ```
    The mesh files will be moved to `../axisem/SOLVER/MESHES/<mesh_name>`.
    Set `MESHNAME` in `../axisem/SOLVER/inparam_basic` to `<mesh_name>`.

---

## Prepare AxiSEM Solver Files

You must edit two primary configuration files: `inparam_basic` and `inparam_advanced`.

### 1. `inparam_basic`
* Set `SEISMOGRAM_LENGTH` to your requirements.
* Set `ATTENUATION` to `false` (the current version supports anisotropic elastic models only).
* Adjust `SIMULATION_TYPE`. If it is not set to `moment`, you must also edit `inparam_source`.

### 2. `inparam_advanced`
Configure the wavefield kernel parameters as follows:

```text
# GLL points to save (starting and ending indices)
KERNEL_IBEG         0
KERNEL_IEND         4
KERNEL_JBEG         0
KERNEL_JEND         4

KERNEL_WAVEFIELDS   true
SAVE_BDRY_FACES     false  # set true to save the listed boundary faces
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

> **Note:** To save boundary faces, set
> `SAVE_BDRY_FACES true` alongside `KERNEL_WAVEFIELDS true` and `USE_NETCDF true`,
> and put `boundary_faces.dat` in the solver's `SOLVER` directory before
> running `submit.csh`.
> Each consecutive 25 rows defines one face as `longitude latitude depth_km`;
> its 13th point determines whether the face is solid or fluid. The solver saves
> the distinct touched elements to `boundary_wavefields.nc4` instead of the
> usual kernel wavefield dump. `DUMP_T0` and `KERNEL_SPP` still set the sample
> times; the other kernel dump selection settings do not select boundary points.

### 3. Source and Station Setup
Prepare your `CMTSOLUTION` and `STATIONS` files. `axisemlib-prepare`
prepares the input files for one event at a time; see the
[injection preparation guide](prepare_axisem_injection.md) for all options.

**Use geographic coordinates** for all latitudes (event latitude in `CMTSOLUTION`, station latitudes in `STATIONS`, and the study region passed to the command). `axisemlib-prepare` converts them from geodetic to geocentric latitudes for AxiSEM, which assumes a spherical Earth. The original files in `CMT_DIR` are left unchanged.

## Run AxiSEM Simulation

Run the solver from `../axisem/SOLVER` after preparing the mesh and the input
files described above. If `SAVE_BDRY_FACES` is enabled, place
`boundary_faces.dat` there first. `submit.csh` builds the solver, creates a
new run directory, copies the inputs, and starts the simulation. The run
directory name must not already exist.

Use `axisemlib merge` to construct `../axisem/SOLVER/boundary_faces.dat` from
SPECFEM's `DATABASES_MPI` files:

```bash
axisemlib merge cube2sph SPECFEM_DB ../axisem/SOLVER/boundary_faces.dat
axisemlib merge cart SPECFEM_DB ../axisem/SOLVER/boundary_faces.dat --utm-zone 10
```

For `cube2sph`, the script reads `proc??????_wavefield_discontinuity_faces`
with either `x y z` or `x y z nx ny nz` rows, and converts the Earth-centered
Cartesian `x y z` coordinates to longitude,
geocentric latitude, and depth below a 6371 km Earth. For `cart`, it reads
`proc??????_normal.txt`, inverts the specified UTM projection to geographic
longitude and latitude, converts latitude to geocentric, and sets depth from
the file's elevation `z` in meters. Cartesian rows contain
`x y z nx ny nz`; any normal columns are omitted from the output. The script skips the first
header line of each `proc*_normal.txt` file.
The three-column `cube2sph` form is sufficient to build the AxiSEM input;
the coupling driver still needs face normals to calculate traction.

Each consecutive 25 input rows becomes one face. Files are merged by ascending
numeric processor number, and `iface` is the zero-based cumulative face count;
no coordinate lookup determines it. Empty processor files add no faces. The
script rejects files whose data rows are not a multiple of 25. Its
`merge_surfaces(...)` return value maps each `iface` to
`(processor_number, local_face_index)`. `submit.csh` copies
`boundary_faces.dat` into the run directory when `SAVE_BDRY_FACES` is enabled.

For one run on the local machine:

```bash
cd ../axisem/SOLVER
./submit.csh ak135.my_event
```

Local runs start in the background; check the `OUTPUT_*` files in the new run directory for progress. For a scheduler, adapt `submit.csh` to your cluster.

To prepare one event, run from the AxiSEM `SOLVER` directory:

```bash
axisemlib-prepare \
    --region-box 114 132 41 46 450 \
    --t0-injection 1166.926174 \
    --cmtsolution CMT_DIR/CMTSOLUTION_SKS_1 \
    --stations CMT_DIR/STATIONS_SKS_1
```

The five `--region-box` values are geographic `lonmin lonmax latmin latmax`
in degrees and maximum depth in kilometres. `--t0-injection` is the event's
injection time in seconds. By default, the script starts dumping 50 seconds
before injection, sets `SEISMOGRAM_LENGTH` to 300 seconds after injection, and
adds a 10-degree margin to the angular region. Override these with
`--pre-buffer-seconds`, `--post-buffer-seconds`, and `--region-buffer-deg`.

The default input paths are `CMT_DIR/CMTSOLUTION`, `CMT_DIR/STATIONS`,
`inparam_basic`, and `inparam_advanced`; the first two are overridden above
to select a particular event. The script writes converted `CMTSOLUTION` and
`STATIONS` files plus substituted parameter files directly to the current
directory by default. Use `--output-dir` to select a different destination.
Source `CMTSOLUTION` and `STATIONS` paths must differ from their output paths,
so geographic coordinates are not converted twice. The script does not submit
the simulation. To run the prepared event, call `submit.csh` explicitly:

```bash
./submit.csh ak135.SKS_1
```

To create receiver seismograms directly from the `Seismograms` group in each
`axisem_output.nc4`, including runs with `SAVE_BDRY_FACES` enabled, run:

```bash
axisemlib seismogram ak135.SKS_1
```

The command reads source metadata and recorded traces from NetCDF, sums the
moment simulations, and rotates the result to North, East, and vertical. It
writes `NETWORK.STATION.BXN.dat`, `.BXE.dat`, and `.BXZ.dat` files to
`./SEISMOGRAMS`, using the established trace naming. Time starts at the first
recorded time minus the source shift and uses the NetCDF seismogram sampling
interval. Use `--output-dir` to choose another directory. `AxiBasicDB` provides wavefield-based synthesis at arbitrary locations.

## Transpose Output Field
For significantly improved data access performance in post-processing, transpose the generated wavefield.

Run the transpose command with the read buffer size in GiB per MPI rank and
one or more AxiSEM run directories:

```bash
mpirun -n 8 axisemlib transpose 2.0 RUN_DIR [ANOTHER_RUN ...]
```

The command checks each Data directory for `boundary_wavefields.nc4`
first, then `axisem_output.nc4`. Boundary fields are written as element-major
`disp_*.bin` and `chi.bin`; standard fields retain the DOF layout.

For a run with boundary surfaces, `AxiBasicDB` reads the boundary mesh and
uses either the transposed binary fields or the wavefields still in NetCDF:

```python
from axisemlib import AxiBasicDB

db = AxiBasicDB()
db.read_basic("RUN_DIR/MZZ/Data/axisem_output.nc4")
db.set_iodata("RUN_DIR")
wavefield = db.syn_surface_wavefield(0, cmtfile="RUN_DIR/CMTSOLUTION")
derived = db.syn_surface_derived_fields(0, cmtfile="RUN_DIR/CMTSOLUTION")
db.close()
```

Face indices start at zero. The first method prints and returns displacement
`(25, 3, nt)` for a solid face or `chi (25, nt)` for a fluid face. The second
prints and returns XYZ stress `(25, 6, nt)` for a solid face or XYZ acoustic
displacement `(25, 3, nt)` for a fluid face. Solid displacement coordinates
can be selected with `comp='enz'` (default), `'xyz'`, or `'spz'`.

When `boundary_wavefields.nc4` is present, `axisemlib.driver` selects separate
boundary workers for Cartesian and `cube2sph` wavefield coupling. The
`cube2sph` worker maps SPECFEM's unique points back to the ascending face
indices; the Cartesian worker uses the cumulative 25-row face count.
Equivalent-force output uses the corresponding
face stress or acoustic fields.
The boundary workers keep the existing record shapes and use these quantities:

| Worker | Solid face | Fluid face |
|---|---|---|
| Cartesian | velocity / traction | `dchi` / acoustic displacement |
| `cube2sph` | displacement / acceleration / traction | `chi` / `ddchi` / acoustic displacement |

For fluid points, the scalar `dchi`, `chi`, or `ddchi` is repeated in all three
components of its record. The third `cube2sph` record is ordered by face
points. Equivalent-force coupling still derives acoustic traction from `ddchi`.

## Getting Started with Examples

To explore the library's capabilities, navigate to the `EXAMPLES/` directory. Configuration is handled via a `YAML` file, which allows you to define paths, coupling methods, and time windows.

Two examples are provided, one for each SPECFEM coordinate system:

| Example | `SPECFEM_SYSTEM` | `coupling_method` | Target code |
|---|---|---|---|
| `EXAMPLES/specfem3d_cart` | `cart` | `wd` | SPECFEM3D (Cartesian, UTM) |
| `EXAMPLES/cube2sph` | `cube2sph` | `wd` | SPECFEM3D with Cube2sph (spherical, Earth-centered) |

Each example directory contains `param.yaml`. From that directory, run
`mpirun -n <N> axisemlib coupling param.yaml` after setting its paths. The
number of MPI ranks does not need to match the number of SPECFEM processors.
The example `AXISEM_DIR` path assumes the sibling checkout layout in the
[installation guide](installation.md) and is resolved from the directory in
which you launch the command.

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

### Example Configuration (`param.yaml`)

```yaml
# Parameter settings for generating the injection field

# Working Directories
AXISEM_DIR: ../../../axisem/SOLVER/ak135
SPECFEM_DB: ~/SPECFEM/DATABASES_MPI    # Path to ${SPECFEM_DIR}/DATABASES_MPI
SPECFEM_DATA: ~/SPECFEM/DATA           # Only needed for equivalent forces coupling
SPECFEM_SYSTEM: 'cube2sph'             # Options: 'cart' or 'cube2sph'
OUTPUT_DIR: './OUTPUT_DIR'

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
