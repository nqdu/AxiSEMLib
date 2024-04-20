# Prepare Boundary Points
your should provide files `proc*_cords.bin`. This file can be obtained from `src/specfem3D/couple_with_injection.f90` in **SPECFEM3D**. The fortran code to write them to disk is like

```fortran
integer,parameter :: FKTRAC_IO = 12314
character(len=256) :: fktracfile
write(fktracfile,'(a,i6.6,a)')trim(FKMODEL_FILE)//'_field/proc',myrank,'cord.bin'

open(UNIT=FKTRAC_IO,file=trim(fktracfile),form='UNFORMATTED',action='write')
write(FKTRAC_IO) xx(1:npt)
write(FKTRAC_IO) yy(1:npt)
write(FKTRAC_IO) zz(1:npt)
write(FKTRAC_IO) nmx(1:npt)
write(FKTRAC IO) nmy(1:npt)
write(FKTRAC IO) nmz(1:npt)

close(FKTRAC_IO)
```

# injection Fields
The code will print the traction/velocity at the injection boundaries, the fortran code to read them is like:
```fortran
integer,parameter :: FKTRAC_IO = 12314
character(len=256) :: fktracfile
write(fktracfile,'(a,i6.6,a)')trim(FKMODEL_FILE)//'_field/proc',myrank,'.bin'
open(UNIT=KTRAC_IO,file=trim(fktracfile),form='UNFORMATTED',action='read')
read(FKTRAC_IO) Vx_t
read(FKTRAC_IO) Vy_t
read(FKTRAC_IO) Vz_t
read(FKTRAC_IO) Tx_t
read(FKTRAC_IO) Ty_t
read(FKTRAC_IO) Tz_t
close(FKTRAC_IO)
```