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
outdir=..  # OUTPUT traction/displ/accel to this dir
coordir=.. # wave_discon_*coordinates 

# time window
nt=1000
dt=0.025
t0=123.4  # t0 = 0. is earthquake origin time

# the cores can be different from # of discon files.
mpirun -np 120 python ../../coupling.py  $axisem_data_dir $coordir $t0 $dt $nt $outdir
