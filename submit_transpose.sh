#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=compute
#SBATCH --time=01:30:00
#SBATCH --mem=0

# load your own libs
module load gcc openmpi python/3.9.8 parallel-hdf5/gcc-8.3.0

# solver dir
axisem_data_dir=.. # like /path/to/axisem/SOLVER/ak135
mpirun -np 8 python ./transpose_fields.py $axisem_data_dir MZZ MXZ_MYZ MXY_MXX_M_MYY MXX_P_MYY