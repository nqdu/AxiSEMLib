# Axisem_lib
library for axisem database.

## Workflow
- compile code (with `pybind11`)
```
bash compile.sh
```
- transpose fields: first load modules
```
module add openmpi parallel-hdf5
```
then compile the code `bash compile_tranpose.sh`. After that, tranpose the wavefields by using:
```
mpirun -np 8 ./bin/tranpose basedir forcetype(MZZ MXY_YX ...)
```
The output file will be `basedir/forcetype/Data/axisem_fields.h5`
    
