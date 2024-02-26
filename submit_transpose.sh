#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --array=1-6
#SBATCH --partition=compute
#SBATCH --account=rrg-liuqy
#SBATCH --time=01:30:00
#SBATCH --mem=0
#module load gcc openmpi python/3.9.8 parallel-hdf5/gcc-8.3.0


mpirun -np 8 python transpose_fields.py ..//SOLVER/ak135.$SLURM_ARRAY_TASK_ID MZZ MXZ_MYZ MXY_MXX_M_MYY MXX_P_MYY
#mpirun -np 32 python transpose_fields.py ..//SOLVER/ak135_th MZZ MXZ_MYZ MXY_MXX_M_MYY MXX_P_MYY
