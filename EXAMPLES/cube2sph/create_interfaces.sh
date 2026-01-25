#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --partition=compute

# load your modules
module load gcc openmpi python/3.9.8 parallel-hdf5/gcc-8.3.0


# the cores can be different from # of discon files.
mpirun -np 120 python ../../run_coupling.py param.yaml
