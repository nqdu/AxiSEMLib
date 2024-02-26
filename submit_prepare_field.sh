#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --time=02:00:00
#SBATCH --array=4-4
#SBATCH --account=rrg-liuqy
#SBATCH --mem=0
#SBATCH --partition=compute

#module load gcc openmpi python/3.9.8 parallel-hdf5/gcc-8.3.0
basedir=/home/l/liuqy/nqdu/scratch/axisem/SOLVER/ak135.$SLURM_ARRAY_TASK_ID/
outdir=output.$SLURM_ARRAY_TASK_ID
coordir=/home/l/liuqy/nqdu/scratch/NEChina_mesh/DATABASES_MPI/

#basedir=/home/l/liuqy/nqdu/scratch/axisem/SOLVER/ak135_th
#outdir=output.th
#coordir=/scratch/l/liuqy/liutia97/cube2sph_injection_test/SPECFEM3D-with-Cube2sph-and-PML/utils/cube2sph/cube2sph_examples/northeast_china/DATABASES_MPI/ 
mpirun -np 120 python coupling.py  $basedir $coordir P 250 100 $outdir
