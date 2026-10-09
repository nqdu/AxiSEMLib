# Prepare one AxiSEM injection event

`axisemlib-prepare` converts a
geographic `CMTSOLUTION` and `STATIONS` file to geocentric latitudes and sets
the time and wavefield region parameters for one event. It writes the four
solver input files and does not start a simulation.

## Example geographic inputs

Example `CMTSOLUTION`:

```text
 PDEW2015  8 10  7 43 37.30 -19.6400 -174.8500  46.3 0.0 5.6 TONGA ISLANDS
event name:     201508100743A
time shift:      8.9700
half duration:   1.6000
latitude:      -19.6200
longitude:    -174.4700
depth:         109.3700
Mrr:      -4.480000e+23
Mtt:       1.590000e+24
Mpp:      -1.140000e+24
Mrt:      -9.660000e+22
Mrp:      -3.650000e+24
Mtp:      -1.960000e+23
```

Example `STATIONS` (ten rows):

```text
CDT BU 41.015300 117.915800 117.900000 0.
SHZ HL 44.062000 128.956000 330.000000 0.
BNX HL 45.739000 127.403000 165.000000 0.
BST JL 41.945200 126.368500 551.900000 0.
QAT JL 45.006100 124.004000 164.400000 0.
CBS JL 42.070000 128.070000 1790.000000 0.
PST JL 42.950100 126.078000 408.800000 0.
SPT JL 43.199700 124.534900 256.600000 0.
SMT JL 42.188200 128.174000 1149.000000 0.
WQT JL 43.690000 129.010000 358.000000 0.
```

The station columns are name, network, geographic latitude, longitude,
elevation in metres, and burial depth in metres. A `STATIONS` file contains
one such row per station. The preparation script converts the centroid
`latitude:` and each station latitude to geocentric coordinates.

## Run from the solver directory

Install AxiSEMLib first. Prepare the mesh and the event's geographic
source and station files, then run the command from the solver's `SOLVER`
directory:

```bash
cd ../axisem/SOLVER
axisemlib-prepare \
    --region-box 114 132 41 46 450 \
    --t0-injection 1166.926174 \
    --cmtsolution CMT_DIR/CMTSOLUTION_SKS_1 \
    --stations CMT_DIR/STATIONS_SKS_1
```

`--region-box` requires five values: minimum and maximum longitude, minimum
and maximum **geographic** latitude (degrees), and maximum depth (km). The
example study region spans 114–132° longitude, 41–46° latitude, and 0–450 km
depth. `--t0-injection` is the event's injection arrival time in seconds on
the solver's time axis.

The script writes `CMTSOLUTION`, `STATIONS`, `inparam_basic`, and
`inparam_advanced` directly into the current directory by default. It reads
all inputs before writing. The source `CMTSOLUTION` and `STATIONS` paths must
differ from their output paths, which prevents converting an already converted
latitude again. Working `inparam_basic` and `inparam_advanced` files in the
output directory are updated in place when used as the inputs.
Each invocation replaces any existing output files with these four names.

After checking the prepared files, start the event explicitly:

```bash
./submit.csh ak135.SKS_1
```

If `SAVE_BDRY_FACES` is enabled, place `boundary_faces.dat` in `SOLVER` before
calling `submit.csh`.

## Input paths and optional buffers

| Option | Default | Purpose |
| --- | --- | --- |
| `--region-box LON_MIN LON_MAX LAT_MIN LAT_MAX MAX_DEPTH_KM` | Required | Geographic study region |
| `--t0-injection SECONDS` | Required | Injection arrival time |
| `--cmtsolution PATH` | `CMT_DIR/CMTSOLUTION` | Geographic event source file |
| `--stations PATH` | `CMT_DIR/STATIONS` | Geographic station file |
| `--inparam-basic PATH` | `inparam_basic` | Basic parameter input |
| `--inparam-advanced PATH` | `inparam_advanced` | Advanced parameter input |
| `--output-dir PATH` | `.` | Destination for the four solver input files |
| `--pre-buffer-seconds SECONDS` | `50` | Start dumping this long before injection |
| `--post-buffer-seconds SECONDS` | `300` | End the simulation this long after injection |
| `--region-buffer-deg DEGREES` | `10` | Add this margin to both angular bounds |

The default `CMT_DIR/CMTSOLUTION` and `CMT_DIR/STATIONS` names are generic
input paths. Pass the event-specific paths, as in the example, when your
source directory contains names such as `CMTSOLUTION_SKS_1` and
`STATIONS_SKS_1`.

For a narrower time and angular window, append:

```bash
--pre-buffer-seconds 40 --post-buffer-seconds 250 --region-buffer-deg 5
```

All buffer sizes must be nonnegative. The injection time must be at least the
pre-injection buffer, so `DUMP_T0` is not negative.

## What the script changes

The output `CMTSOLUTION` uses a geocentric `latitude:` value. The latitude in
each data row of `STATIONS` is also converted to geocentric degrees. The
geographic input files are left unchanged when they are separate from the
output paths.

The parameter values are calculated as follows:

| Parameter | Value |
| --- | --- |
| `DUMP_T0` | `t0_injection - pre_buffer_seconds` |
| `SEISMOGRAM_LENGTH` | `t0_injection + post_buffer_seconds` |
| `KERNEL_COLAT_MIN` | Minimum event-to-region epicentral distance minus `region_buffer_deg` |
| `KERNEL_COLAT_MAX` | Maximum event-to-region epicentral distance plus `region_buffer_deg` |
| `KERNEL_RMIN` | `6371 - MAX_DEPTH_KM` |
| `KERNEL_RMAX` | `7000` km |

The angular limits come from a 100 × 100 sample of the geographic region
after its latitudes are converted to geocentric coordinates. If the advanced
parameter input has no `DUMP_T0` line, the script inserts it after
`KERNEL_SPP`.

Use `--output-dir` to prepare files outside `SOLVER`. To submit that event,
copy the four prepared files into `SOLVER` before running `submit.csh`; the
solver submission script reads its inputs from that directory.
