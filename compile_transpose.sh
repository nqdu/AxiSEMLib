module purge 
module load gcc openmpi parallel-hdf5/gcc 

includes="-I./src -I$PH5_PATH/include -I$EIGEN_INC"
set -x
# transpose 
$MPICXX -DUSE_MPI src/hdf5file.cpp src/transpose.cpp -O3 -o ./bin/transpose $includes  -L$PH5_PATH/lib -lhdf5 