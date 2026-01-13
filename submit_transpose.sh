#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=compute
#SBATCH --time=01:30:00
#SBATCH --mem=0

# load your own libs
module load fwi/gcc

# solver dir
axisem_list=.. # like /path/to/axisem/SOLVER/ak135.*
size_gb_per_rank=2.0  # size in GB per rank for buffer
mpirun -np 8 python ./transpose_fields.py 1 $size_gb_per_rank $axisem_list