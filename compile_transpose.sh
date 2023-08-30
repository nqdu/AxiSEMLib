module purge 
module load gcc openmpi parallel-hdf5/gcc 
MPICXX=mpic++
includes="-I./src -I$PH5_PATH/include -I$EIGEN_INC"
set -x
# transpose1 
$MPICXX -DUSE_MPI src/transpose1.cpp -O3 -o ./bin/transpose $includes  -L$PH5_PATH/lib -lhdf5 