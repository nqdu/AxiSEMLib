#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --partition=compute

# load your modules
module load gcc openmpi python/3.9.8 parallel-hdf5/gcc-8.3.0

# set your params
axisem_data_dir=.. # like /path/to/axisem/SOLVER/ak135
output_dir=..  # OUTPUT traction/displ/accel to this dir
coordir=.. # wave_discon_*coordinates 

# time window
nt=1000
dt=0.025
t0=0.  # started from earthquake origin time
UTM_ZONE=10

mpirun -np 120 python ../../coupling_specfem.py  $axisem_data_dir $coordir $UTM_ZONE $t0 $dt $nt $outdir
