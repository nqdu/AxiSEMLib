# Prepare Boundary Points
your should provide files `proc*_normal.txt`. This file can be obtained from `LOCAL_PATH` in **SPECFEM3D**'s `Par_file`. Your can generate all these files by set `COUPLE_WITH_INJECTION_TECHNIQUE = .TRUE.` and `INJECTION_TYPE = 3` before you generate your mesh.  

# OUTPUT
The program will print `proc*_axisem_sol` files that contain velocity/tractions at each time step. You can set parameters in `create_interfaces.sh`.