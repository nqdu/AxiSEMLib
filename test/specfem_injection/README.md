# Prepare Boundary Points
your should provide two files `proc*_wavefield_discontinuity_faces` and `proc*_wavefield_discontinuity_points`. The first file is with format: 
```bash
x y z
```
where `x/y/z` are Cartesian coordinates for each boundary points with dimension `m`. Note that this file should be of shape (nbds), only keep unique points. For second:
```bash
x y z nx ny nz
```
where `x/y/z` are Cartesian coordinates for each boundary points with dimension `m` and `n_x/y/z` are the normal vector ouside the study region. It should be of the shape (nspec_bd,NGLL2)
