from database import AxiBasicDB
import numpy as np 
import os  
from mpi4py import MPI
from utils import cart2sph,allocate_task
from utils import resample_axisem,geodetic_to_geocentric
from FortranIO import FortranIO  
from pathlib import Path
from collections.abc import Sequence
from typing import Any, TextIO
import re


EARTH_RADIUS_M = 6371000.0
NGLL2 = 25
FACE_FILE = re.compile(r'proc(\d+)_wavefield_discontinuity_faces')


def _processor_face_files(specfem_db: str | Path) -> list[tuple[int, Path]]:
    """List cube2sph face files by numeric SPECFEM processor number."""
    paths = []
    for path in Path(specfem_db).glob('proc*_wavefield_discontinuity_faces'):
        match = FACE_FILE.fullmatch(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    if not paths:
        raise FileNotFoundError(f'No cube2sph face files in {specfem_db}')
    return sorted(paths)


def _read_face_points(path: Path) -> np.ndarray:
    """Read Cartesian coordinates and normals in their original GLL order."""
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            columns = line.split()
            if len(columns) != 6:
                raise ValueError(f'{path}:{line_number}: expected x y z nx ny nz')
            try:
                rows.append([float(value) for value in columns])
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: invalid coordinate or normal') from exc

    if len(rows) % NGLL2:
        raise ValueError(f'{path}: {len(rows)} points is not a multiple of {NGLL2}')
    points = np.asarray(rows, dtype=float).reshape(-1, 6)
    if not np.isfinite(points).all():
        raise ValueError(f'{path}: nonfinite coordinate or normal')
    return points


def _cube2sph_faces(specfem_db: str | Path) -> tuple[
        list[tuple[int, np.ndarray]], list[tuple[int, int]]]:
    """Return faces and their zero-based (processor, local face) sources."""
    blocks = []
    sources = []
    for proc, path in _processor_face_files(specfem_db):
        points = _read_face_points(path)
        blocks.append((proc, points))
        sources.extend((proc, local_face) for local_face in range(len(points) // NGLL2))
    if not sources:
        raise ValueError(f'No boundary faces in {specfem_db}')
    return blocks, sources


def _validate_boundary_faces(db: AxiBasicDB,
                             blocks: list[tuple[int, np.ndarray]]) -> None:
    """Reject a solver boundary file built from different SPECFEM faces."""
    points = np.concatenate([block for _, block in blocks], axis=0)
    radius, latitude, longitude = cart2sph(*points[:, :3].T)
    depth_km = (EARTH_RADIUS_M - radius) / 1000.0
    if (len(points) != len(db.point_longitude)
            or not np.allclose(longitude, db.point_longitude, atol=1e-5, rtol=0)
            or not np.allclose(latitude, db.point_latitude, atol=1e-5, rtol=0)
            or not np.allclose(depth_km, db.point_depth_km, atol=1e-3, rtol=0)):
        raise ValueError('AxiSEM boundary faces differ from the cube2sph processor files')


def _surface_traction(stress: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Multiply XYZ Voigt stress by outward normals at the 25 face points."""
    traction = np.empty((NGLL2, 3, stress.shape[-1]))
    nx, ny, nz = normals.T
    traction[:, 0] = stress[:, 0]*nx[:, None] + stress[:, 5]*ny[:, None] + stress[:, 4]*nz[:, None]
    traction[:, 1] = stress[:, 5]*nx[:, None] + stress[:, 1]*ny[:, None] + stress[:, 3]*nz[:, None]
    traction[:, 2] = stress[:, 4]*nx[:, None] + stress[:, 3]*ny[:, None] + stress[:, 2]*nz[:, None]
    return traction


def _fluid_surface_moment(db: AxiBasicDB, iface: int, normals: np.ndarray,
                          displacement: np.ndarray) -> np.ndarray:
    """Build the isotropic acoustic moment from normal displacement."""
    from sem_funcs import lagrange_interpol_2D_td

    _, points = db._surface_points(iface)
    moment = np.zeros((NGLL2, 6, db.nt))
    for row, point in enumerate(points):
        elem = int(db.point_mesh_index[point])
        nodes_xi = db.glj if db.axis[elem] == 1 else db.gll
        bulk_modulus = lagrange_interpol_2D_td(
            nodes_xi, db.gll, db.xlamda[elem].T[None, :, :],
            db.point_xi[point], db.point_eta[point])[0]
        pressure_jump = bulk_modulus * (normals[row] @ displacement[row])
        moment[row, 0:3] = pressure_jump
    return moment


def _boundary_face_fields(db: AxiBasicDB, iface: int, cmtfile: str,
                          comp: str = 'xyz') -> tuple[np.ndarray, np.ndarray]:
    """Return solid displacement in comp, or XYZ acoustic displacement."""
    if db.face_phase[iface] == 0:
        displacement = db.syn_surface_wavefield(
            iface, comp=comp, cmtfile=cmtfile, print_output=False)
        stress = db.syn_surface_derived_fields(
            iface, cmtfile=cmtfile, print_output=False)
        return displacement, stress

    displacement = db.syn_surface_derived_fields(
        iface, cmtfile=cmtfile, print_output=False)
    chi = db.syn_surface_wavefield(
        iface, cmtfile=cmtfile, print_output=False)
    return displacement, chi


def read_boundary_points(
        coordir: str, iproc: int) -> tuple[np.ndarray, ...] | list[list[float]]:
    """
    read specfem3D boundary points from proc*_normal.txt

    coordir: str
        coordinate directory, specfem3D's DATABASES_MPI
    iproc: int
        current proc id
    """

    # read points
    filename = coordir + '/proc%06d_normal.txt' %(iproc)
    data = np.loadtxt(filename,dtype='f4',skiprows=1,ndmin=2)
    if data.shape[0] == 0:
        return [[] for i in range(6)]
    
    xx,yy,zz,nnx,nny,nnz = np.loadtxt(filename,dtype='f4',skiprows=1,unpack=True)

    return xx,yy,zz,nnx,nny,nnz 



def get_field_proc_cart(args: tuple[int, str, str, str, np.ndarray, int]) -> None:
    from pyproj import Proj
    from utils import rotate_EN_to_UTM
    from utils import rotation_matrix,rotate_tensor2

    # unpack input paramters
    iproc,basedir,coordir,outdir,tvec,UTM_ZONE = args

    # read database
    db = AxiBasicDB()
    db.read_basic(basedir + "/MZZ/Data/axisem_output.nc4")
    db.set_iodata(basedir)

    # read boundary points
    xx,yy,zz,nnx,nny,nnz = read_boundary_points(coordir,iproc)
    npts = len(xx)

    # create dataset
    t0 = np.arange(db.nt) * db.dtsamp + db.t0
    t1 = tvec.copy()
    nt1 = len(t1)

    # allocate space for veloc/traction
    veloc_axi = np.zeros((nt1,npts,3),'f4')
    trac_axi = np.zeros((nt1,npts,3),'f4')

    # open file
    outbin = "%s/proc%06d_sol_axisem"%(outdir,iproc)
    f = FortranIO(outbin,'w')
    if npts == 0:
        for i in range(nt1):
            f.write_record(veloc_axi[i,...],trac_axi[i,...])
        f.close()
        return 0

    # convert to spherical coordinates
    p = Proj(proj='utm',zone=UTM_ZONE,ellps='WGS84')

    # nqdu added, change the working latitude from geographic to geocentric
    stlo,stla = p(xx,yy,inverse=True) # this is in geographic!
    stla = geodetic_to_geocentric(stla)
    r = zz + 6371000
    stel = -6371000 + r

    fmax = 1.0 / db.dominant_T0
    if iproc == 0: print("synthetic traction/velocity ...")
    for ir in range(npts):
        #print(f"synthetic traction for point {ir+1} of {npts} in proc {iproc} ...")

        # get rotation matrix from (xyz) to (enz)
        R = rotation_matrix(np.deg2rad(90-stla[ir]),np.deg2rad(stlo[ir]))
        tmp = R[:,1] * 1.
        R[:,1] = -R[:,0] * 1. # \hat{e}_n is -\hat{\theta}
        R[:,0] = tmp * 1.
        R = R.T

        # get stress in ENZ
        sig_xyz = db.syn_stress(stla[ir],stlo[ir],stel[ir],basedir + '/CMTSOLUTION')
        sig_xyz = rotate_tensor2(sig_xyz,R)
        Tx = np.zeros((db.nt)); Ty = Tx *  1.; Tz = Tx * 1. 

        # synthetic displ in enz, note that enz is specfem3d's (xyz)
        ue,un,uz = db.syn_seismo(stla[ir],stlo[ir],stel[ir],'enz',basedir + '/CMTSOLUTION')

        # get meridian convergence angle in rad 
        gamma = p.get_factors(stlo[ir],stla[ir]).meridian_convergence
        gamma = np.deg2rad(gamma)

        # rotate ue,un to UTM
        ux,uy = rotate_EN_to_UTM(ue,un,gamma)

        # get velocity
        _,veloc_axi[:,ir,0] = resample_axisem(t0,ux,t1,
                                            deriv_order=1,
                                            f_dom=fmax)
        _,veloc_axi[:,ir,1] = resample_axisem(t0,uy,t1,
                                            deriv_order=1,
                                            f_dom=fmax)
        _,veloc_axi[:,ir,2] = resample_axisem(t0,uz,t1,
                                            deriv_order=1,
                                            f_dom=fmax)
                                             
        # traction
        nx = nnx[ir]; ny = nny[ir]; nz = nnz[ir] # note it's ENZ! 
        Te = sig_xyz[0,:] * nx + sig_xyz[5,:] * ny + sig_xyz[4,:] * nz 
        Tn = sig_xyz[5,:] * nx + sig_xyz[1,:] * ny + sig_xyz[3,:] * nz 
        Tz = sig_xyz[4,:] * nx + sig_xyz[3,:] * ny + sig_xyz[2,:] * nz 

        # rotate Tn/Te to UTM
        Tx,Ty = rotate_EN_to_UTM(Te,Tn,gamma)

        trac_axi[:,ir,0],_ = resample_axisem(t0,Tx,t1,
                                        deriv_order=0,
                                        f_dom=fmax)
        trac_axi[:,ir,1],_ = resample_axisem(t0,Ty,t1,
                                        deriv_order=0,
                                        f_dom=fmax)
        trac_axi[:,ir,2],_ = resample_axisem(t0,Tz,t1,
                                        deriv_order=0,
                                        f_dom=fmax)

    # write file
    for i in range(nt1):
        f.write_record(veloc_axi[i,...],trac_axi[i,...])
        
    f.close()

def get_field_proc_cart_boundary(
        args: tuple[int, str, str, str, np.ndarray, int, int]) -> None:
    """Write solid velocity/traction or fluid dchi/displacement by face."""
    from pyproj import Proj
    from utils import rotate_EN_to_UTM,rotation_matrix,rotate_tensor2

    iproc,basedir,coordir,outdir,tvec,UTM_ZONE,first_iface = args
    db = AxiBasicDB()
    db.read_basic(basedir + '/MZZ/Data/axisem_output.nc4')
    db.set_iodata(basedir)
    if not db._boundary_mode:
        raise ValueError('A boundary_wavefields.nc4 file is required')

    input_file = Path(coordir) / f'proc{iproc:06d}_normal.txt'
    with input_file.open() as stream:
        next(stream,None)
        rows = [line for line in stream if line.strip()]
    data = np.loadtxt(rows,ndmin=2) if rows else np.empty((0,6))
    npts = len(data)
    if npts % NGLL2:
        raise ValueError(f'{input_file}: expected 25 ordered points per face')
    nt = len(tvec)
    velocity = np.zeros((nt,npts,3),dtype='f4')
    traction = np.zeros((nt,npts,3),dtype='f4')

    if npts:
        if data.shape[1] != 6:
            raise ValueError(f'{input_file}: expected x y z nx ny nz')

        # Convert UTM positions for local rotations; iface comes from row counts.
        projection = Proj(proj='utm',zone=UTM_ZONE,ellps='WGS84')
        longitude,latitude = projection(data[:,0],data[:,1],inverse=True)
        latitude = geodetic_to_geocentric(latitude)
        colat = np.deg2rad(90-latitude)
        lonrad = np.deg2rad(longitude)

        # Each consecutive 25-row block is one saved face.
        cmtfile = basedir + '/CMTSOLUTION'
        t0 = db.sample_time
        fmax = 1.0 / db.dominant_T0
        for local_face in range(npts//NGLL2):
            iface = first_iface + local_face
            displacement,derived = _boundary_face_fields(
                db,iface,cmtfile,comp='enz')
            is_solid = db.face_phase[iface] == 0

            # Fill the 25 points while this face's synthesized fields are available.
            for row in range(NGLL2):
                ir = local_face*NGLL2 + row
                R = rotation_matrix(colat[ir],lonrad[ir])
                east = R[:,1].copy()
                R[:,1] = -R[:,0]
                R[:,0] = east
                R = R.T
                if is_solid:
                    ue,un,uz = displacement[row]
                else:
                    ue,un,uz = R @ displacement[row]
                gamma = np.deg2rad(
                    projection.get_factors(longitude[ir],latitude[ir]).meridian_convergence)
                ux,uy = rotate_EN_to_UTM(ue,un,gamma)

                if is_solid:
                    # Elastic faces carry velocity and traction in UTM coordinates.
                    for component,values in enumerate((ux,uy,uz)):
                        _,velocity[:,ir,component] = resample_axisem(
                            t0,values,tvec,deriv_order=1,f_dom=fmax)

                    nx,ny,nz = data[ir,3:]
                    stress = rotate_tensor2(derived[row],R)
                    te = stress[0]*nx + stress[5]*ny + stress[4]*nz
                    tn = stress[5]*nx + stress[1]*ny + stress[3]*nz
                    tz = stress[4]*nx + stress[3]*ny + stress[2]*nz
                    tx,ty = rotate_EN_to_UTM(te,tn,gamma)
                    for component,values in enumerate((tx,ty,tz)):
                        traction[:,ir,component],_ = resample_axisem(
                            t0,values,tvec,deriv_order=0,f_dom=fmax)
                else:
                    # Acoustic faces carry scalar dchi and vector displacement.
                    _,dchi = resample_axisem(
                        t0,derived[row],tvec,deriv_order=1,f_dom=fmax)
                    velocity[:,ir,:] = dchi[:,None]
                    for component,values in enumerate((ux,uy,uz)):
                        traction[:,ir,component],_ = resample_axisem(
                            t0,values,tvec,deriv_order=0,f_dom=fmax)

    output_file = Path(outdir) / f'proc{iproc:06d}_sol_axisem'
    with FortranIO(output_file,'w') as stream:
        for it in range(nt):
            stream.write_record(velocity[it],traction[it])
    db.close()

def get_wavefield_sph(args: tuple[int, str, str, str, np.ndarray, bool]) -> None:
    """
    get wavefield (displ/accel/traction) on the injection boundaries in spherical system

    Parameters
    -------------------
    args: tuple
        (iproc,basedir,coordir,outdir,tvec,downsample)
    """

    iproc,basedir,coordir,outdir,tvec,downsample = args
    datadir = coordir
    file_trac = datadir + "proc%06d_wavefield_discontinuity_faces"%iproc
    file_disp = datadir + "proc%06d_wavefield_discontinuity_points"%iproc
    outbin = "%s/proc%06d_wavefield_discontinuity.bin"%(outdir,iproc)

     # read database
    db = AxiBasicDB()
    db.read_basic(basedir + "/MZZ/Data/axisem_output.nc4")
    db.set_iodata(basedir)

    # time vector
    t0 = np.arange(db.nt) * db.dtsamp + db.t0
    t1 = tvec.copy()
    nt1 = len(t1)
    if downsample:
        dt_dsmp = min(db.dominant_T0 / 2 / 5., 0.5) # 1/5 of Nyquist freq = 1/(2T0) / 5
        nt1 = int((t1[-1] - t1[0]) / dt_dsmp) + 1

        # slightly lengthen t1 to [t1[0] - dt_dsmp, t1[-1] + dt_dsmp]
        tnew = np.arange(nt1 + 2) * dt_dsmp + t1[0] - dt_dsmp
        t1 = tnew.copy()
        nt1 = len(t1)

        # write info
        if iproc == 0:
            fio = open(outdir + "/wavefield_discontinuity_info.txt","w")
            fio.write("%f\n" % (dt_dsmp))
            fio.write("%d\n" % (nt1))
            fio.close()
    else: 
        # sanity check
        if iproc == 0:
            if os.path.exists(outdir + "/wavefield_discontinuity_info.txt"):
                # remove it
                os.remove(outdir + "/wavefield_discontinuity_info.txt")

    # get fmax 
    fmax = 1.0 / db.dominant_T0

    # create datafile for displ/accel
    if os.path.getsize(file_disp) != 0:
        data = np.loadtxt(file_disp,ndmin=2)
        r,stla,stlo = cart2sph(data[:,0],data[:,1],data[:,2])
        stel = -6371000 + r
        npts = len(r)
    else:
        npts = 0
    displ = np.zeros((nt1,npts,3),dtype='f4')
    accel = np.zeros((nt1,npts,3),dtype='f4')

    # compute displ/accel on the injection boundaries
    print(f"synthetic displ/accel for {file_disp} ...")
    for ir in range(npts):
        #print(f"synthetic displ/accel for point {ir+1} in proc {iproc} ...")
        ux1,uy1,uz1 = db.syn_seismo(stla[ir],stlo[ir],stel[ir],'xyz',basedir + '/CMTSOLUTION')

        # interpolate to t1 
        displ[:,ir,0],accel[:,ir,0] = resample_axisem(t0,ux1,t1,
                                                     deriv_order=2,
                                                     f_dom=fmax)
        displ[:,ir,1],accel[:,ir,1] = resample_axisem(t0,uy1,t1,
                                                     deriv_order=2,
                                                     f_dom=fmax)
        displ[:,ir,2],accel[:,ir,2] = resample_axisem(t0,uz1,t1,
                                                     deriv_order=2,
                                                     f_dom=fmax)

    # compute traction on the injection boundaries
    if os.path.getsize(file_trac) != 0:
        data = np.loadtxt(file_trac,ndmin=2)
        r,stla,stlo = cart2sph(data[:,0],data[:,1],data[:,2])
        stel = -6371000 + r
        npts = len(r)
    else:
        npts = 0
    tract = np.zeros((nt1,npts,3),dtype='f4')
    print(f"synthetic traction for {file_trac} ...")

    for ir in range(npts):
        #print(f"synthetic traction for point {ir+1} in proc {iproc} ...")
        sig_xyz = db.syn_stress(stla[ir],stlo[ir],stel[ir],basedir + '/CMTSOLUTION')
        Tx = np.zeros((db.nt)); Ty = Tx *  1.; Tz = Tx * 1. 

        nx = data[ir,3]; ny = data[ir,4]; nz = data[ir,5]
        Tx = sig_xyz[0,:] * nx + sig_xyz[5,:] * ny + sig_xyz[4,:] * nz 
        Ty = sig_xyz[5,:] * nx + sig_xyz[1,:] * ny + sig_xyz[3,:] * nz 
        Tz = sig_xyz[4,:] * nx + sig_xyz[3,:] * ny + sig_xyz[2,:] * nz 

        tract[:,ir,0],_ = resample_axisem(t0,Tx,t1,
                                        deriv_order=0,
                                        f_dom=fmax)
        tract[:,ir,1],_ = resample_axisem(t0,Ty,t1,
                                        deriv_order=0,
                                        f_dom=fmax)
        tract[:,ir,2],_ = resample_axisem(t0,Tz,t1,
                                        deriv_order=0,
                                        f_dom=fmax)

    # write final binary for specfem_injection
    displ = displ.astype('f4')
    accel = accel.astype('f4')
    tract = tract.astype('f4')
    fileio = FortranIO(outbin,"w")
    for it in range(nt1):
        fileio.write_record(displ[it,:,:])
        fileio.write_record(accel[it,:,:])
        fileio.write_record(tract[it,:,:])
    fileio.close()

def get_wavefield_sph_boundary(
        args: tuple[int, str, str, str, np.ndarray, bool,
                    list[int], np.ndarray, int]) -> None:
    """Write solid u/a/traction or fluid chi/ddchi/u by face."""
    from scipy.spatial import cKDTree

    iproc,basedir,coordir,outdir,tvec,downsample,face_ids,face_points,first_proc = args
    db = AxiBasicDB()
    db.read_basic(basedir + '/MZZ/Data/axisem_output.nc4')
    db.set_iodata(basedir)
    if not db._boundary_mode:
        raise ValueError('A boundary_wavefields.nc4 file is required')

    t0 = db.sample_time
    t1 = tvec.copy()
    if downsample:
        dt_dsmp = min(db.dominant_T0 / 10.0,0.5)
        nstep = int((t1[-1]-t1[0])/dt_dsmp) + 1
        t1 = np.arange(nstep+2)*dt_dsmp + t1[0] - dt_dsmp
        if iproc == first_proc:
            info_file = Path(outdir) / 'wavefield_discontinuity_info.txt'
            info_file.write_text(f'{dt_dsmp:f}\n{len(t1)}\n')
    elif iproc == first_proc:
        info_file = Path(outdir) / 'wavefield_discontinuity_info.txt'
        if info_file.exists():
            info_file.unlink()

    fmax = 1.0 / db.dominant_T0
    nt = len(t1)
    cmtfile = basedir + '/CMTSOLUTION'

    # Map every (global iface, GLL point) to its unique SPECFEM point ID.
    point_file = Path(coordir) / f'proc{iproc:06d}_wavefield_discontinuity_points'
    if point_file.stat().st_size:
        points = np.loadtxt(point_file,ndmin=2)
    else:
        points = np.empty((0,3))
    if len(face_points) != len(face_ids)*NGLL2:
        raise ValueError(f'Processor {iproc} has an inconsistent face point count')
    if len(face_points):
        if not len(points):
            raise ValueError(f'{point_file} is missing the processor face points')
        distance,unique_ids = cKDTree(points[:,:3]).query(face_points[:,:3])
        if np.any(distance > 1.0) or len(np.unique(unique_ids)) != len(points):
            raise ValueError(f'{point_file} differs from the processor face points')
        face_to_unique = dict(zip(face_ids,unique_ids.reshape(len(face_ids),NGLL2)))
    else:
        if len(points):
            raise ValueError(f'{point_file} has points but processor {iproc} has no faces')
        face_to_unique = {}

    displacement = np.zeros((nt,len(points),3),dtype='f4')
    acceleration = np.zeros_like(displacement)
    third = np.zeros((nt,len(face_points),3),dtype='f4')

    # Synthesize each face once and fill both unique-point and face-point output.
    for local_face,iface in enumerate(face_ids):
        face_start = local_face*NGLL2
        normals = face_points[face_start:face_start+NGLL2,3:]
        field,derived = _boundary_face_fields(db,iface,cmtfile)
        if db.face_phase[iface] == 0:
            stress_traction = _surface_traction(derived,normals)

        for row in range(NGLL2):
            unique_id = face_to_unique[iface][row]
            face_id = face_start + row
            if db.face_phase[iface] == 0:
                for component in range(3):
                    displacement[:,unique_id,component],acceleration[:,unique_id,component] = (
                        resample_axisem(t0,field[row,component],t1,
                                        deriv_order=2,f_dom=fmax))
                    third[:,face_id,component],_ = resample_axisem(
                        t0,stress_traction[row,component],t1,deriv_order=0,f_dom=fmax)
            else:
                # Repeat scalar chi and ddchi in all unique-point components.
                chi,ddchi = resample_axisem(
                    t0,derived[row],t1,deriv_order=2,f_dom=fmax)
                displacement[:,unique_id,:] = chi[:,None]
                acceleration[:,unique_id,:] = ddchi[:,None]
                for component in range(3):
                    third[:,face_id,component],_ = resample_axisem(
                        t0,field[row,component],t1,deriv_order=0,f_dom=fmax)

    output_file = Path(outdir) / f'proc{iproc:06d}_wavefield_discontinuity.bin'
    with FortranIO(output_file,'w') as stream:
        for it in range(nt):
            stream.write_record(displacement[it])
            stream.write_record(acceleration[it])
            stream.write_record(third[it])
    db.close()

def _write_force_file(f: TextIO,
                      cords: Sequence[float],
                      force: Sequence[float],
                      stf_file: str,
                      stf: np.ndarray) -> None:
    """Write a force source at (longitude, latitude, depth) and its time series."""

    f.write("FORCE 000\n")
    f.write("time shift:    0.\n")
    f.write("hdurorf0:    0.\n")
    f.write("latorUTM:  %.6f\n" % (cords[1]))
    f.write("longorUTM:  %.6f\n" % (cords[0]))
    f.write("depth:  %.6f\n" % (cords[2]))
    f.write("source time function:   0\n")
    f.write("factor force source:    1.0\n")
    f.write("component dir vect source E: %f\n" %(force[0]))
    f.write("component dir vect source N: %f\n" %(force[1]))
    f.write("component dir vect source Z: %f\n" %(force[2]))
    f.write(f"{stf_file}\n")
    f1 = open(stf_file,'wb')
    byte = stf.tobytes()
    f1.write(byte)
    f1.close()

def _write_mt_file(f: TextIO,
                   cords: Sequence[float],
                   mt: np.ndarray,
                   stf_file: str,
                   stf: np.ndarray) -> None:
    f.write("whoareyou\n")
    f.write("time shift:    0.")
    f.write("hdurorf0:    0.\n")
    f.write("latorUTM:  %.6f\n" % (cords[1]))
    f.write("longorUTM:  %.6f\n" % (cords[0]))
    f.write("depth:  %.6f\n" % (cords[2]))
    f.write("Mrr: %e\n" % (mt[0]))
    f.write("Mtt: %e\n" % (mt[1]))
    f.write("Mpp: %e\n" % (mt[2]))
    f.write("Mrt: %e\n" % (mt[3]))
    f.write("Mrp: %e\n" % (mt[4]))
    f.write("Mtp: %e\n" % (mt[5]))
    f.write(f"{stf_file}\n")
    f1 = open(stf_file,'wb')
    byte = stf.tobytes()
    f1.write(byte)
    f1.close()
    f.write("\n")

def equivalent_force_cube2sph(param: dict[str, Any]) -> None:
    """
    equivalent force coupling in cube2sph system
    
    :param param: Description
    :type param: dict
    """
    from jacobian import compute_jacobian_surface,moment_to_force

    # mpi 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # constants
    NGLL:int = 5

    # The input order defines the same global iface numbers as AxiSEM.
    blocks, face_sources = _cube2sph_faces(param['SPECFEM_DB'])
    nfaces = len(face_sources)
    points = np.concatenate([block for _, block in blocks], axis=0)
    cords = points[:, :3].reshape(nfaces, NGLL, NGLL, 3)
    norms = points[:, 3:].reshape(nfaces, NGLL, NGLL, 3)

    # allocate tasks
    startid,endid = allocate_task(nfaces,nprocs,rank)
    nfaces_loc = endid - startid + 1

    # compute jacobian2D  if required 
    only_eq_force:bool = param['only_eq_force']
    if not only_eq_force: # moment tensor will be used
        jaco = None
        dxi_dr = None
    else:
        jaco, dxi_dr = compute_jacobian_surface(cords[startid:endid+1,:,:,:])

    # read axisem database
    db = AxiBasicDB()
    db.read_basic(param['AXISEM_DIR'] + "/MZZ/Data/axisem_output.nc4")
    db.set_iodata(param['AXISEM_DIR'])
    if db._boundary_mode:
        _validate_boundary_faces(db, blocks)
    t0 = np.arange(db.nt) * db.dtsamp + db.t0
    t1 = np.arange(param['nt']) * param['dt'] + param['t0']
    nt1 = len(t1)

    # force components 
    stf = np.zeros((nfaces_loc,NGLL,NGLL,3,nt1),dtype=float)
    stf1 = None 
    stf_mt = np.zeros((nfaces_loc,NGLL,NGLL,6,nt1),dtype=float)

    # gather nfaces_loc from all procs
    nfaces_all = comm.allgather(nfaces_loc)
    nfaces_cum = np.cumsum([0] + nfaces_all)
    offset_f = nfaces_cum[rank] * NGLL * NGLL * 3
    offset_mt = nfaces_cum[rank] * NGLL * NGLL * 6
    if only_eq_force:
        offset_f *= 2

    # compute equivalent force on each face
    fmax = 1.0 / db.dominant_T0
    for iface in range(startid,endid+1):
        local_face = iface - startid
        if db._boundary_mode:
            face_normals = norms[iface].reshape(NGLL2, 3)
            if db.face_phase[iface] == 0:
                face_stress = db.syn_surface_derived_fields(
                    iface, cmtfile=param['AXISEM_DIR'] + '/CMTSOLUTION', print_output=False)
            else:
                face_displacement = db.syn_surface_derived_fields(
                    iface, cmtfile=param['AXISEM_DIR'] + '/CMTSOLUTION', print_output=False)
                face_chi = db.syn_surface_wavefield(
                    iface, cmtfile=param['AXISEM_DIR'] + '/CMTSOLUTION', print_output=False)
                face_moment = _fluid_surface_moment(db, iface, face_normals, face_displacement)
        #print(f"compute equivalent force for face {iface+1} of {nfaces} in proc {rank} ...")
        for i in range(NGLL):
            for j in range(NGLL):
                # get spherical coordinates
                r,stla,stlo = cart2sph(cords[iface,i,j,0],
                                       cords[iface,i,j,1],
                                       cords[iface,i,j,2])
                stel = -6371000 + r 

                # get stress 
                if db._boundary_mode and db.face_phase[iface] == 0:
                    sig_xyz = face_stress[i*NGLL+j]
                elif not db._boundary_mode:
                    sig_xyz = db.syn_stress(stla,stlo,stel,param['AXISEM_DIR'] + '/CMTSOLUTION')
                nx = norms[iface,i,j,0]; ny = norms[iface,i,j,1]; nz = norms[iface,i,j,2]
                if db._boundary_mode and db.face_phase[iface] == 1:
                    _, ddchi = resample_axisem(t0,face_chi[i*NGLL+j],t1,
                                               deriv_order=2,f_dom=fmax)
                    Tx,Ty,Tz = ddchi * nx, ddchi * ny, ddchi * nz
                else:
                    Tx = sig_xyz[0,:] * nx + sig_xyz[5,:] * ny + sig_xyz[4,:] * nz
                    Ty = sig_xyz[5,:] * nx + sig_xyz[1,:] * ny + sig_xyz[3,:] * nz
                    Tz = sig_xyz[4,:] * nx + sig_xyz[3,:] * ny + sig_xyz[2,:] * nz
                    Tx,_ = resample_axisem(t0,Tx,t1,deriv_order=0,f_dom=fmax)
                    Ty,_ = resample_axisem(t0,Ty,t1,deriv_order=0,f_dom=fmax)
                    Tz,_ = resample_axisem(t0,Tz,t1,deriv_order=0,f_dom=fmax)
                stf[local_face,i,j,0,:] = - Tx
                stf[local_face,i,j,1,:] = - Ty
                stf[local_face,i,j,2,:] = - Tz

                # get equivalent mt
                norm_xyz = np.array(norms[iface,i,j,:],dtype=float) 
                if db._boundary_mode and db.face_phase[iface] == 1:
                    mt_xyz = face_moment[i*NGLL+j]
                else:
                    mt_xyz = db.syn_eq_moment(
                        stla,stlo,stel,
                        norm_xyz,
                        param['AXISEM_DIR'] + '/CMTSOLUTION'
                    )

                # save mt 
                for k in range(6):
                    out,_ = resample_axisem(t0,mt_xyz[k,:],t1,
                                           deriv_order=0,
                                          f_dom=fmax)
                    stf_mt[local_face,i,j,k,:] = out

    # now we change to stf_mt to equivalent force if required
    if only_eq_force:
        stf1 = moment_to_force(
            stf_mt,
            jaco,
            dxi_dr
        )

    # now write data to txt files
    nforces = nfaces_loc * NGLL * NGLL * 3 
    ncmt = nfaces_loc * NGLL * NGLL * 6
    if only_eq_force:
        ncmt = 0
    nforces = comm.allreduce(nforces,op=MPI.SUM)
    ncmt = comm.allreduce(ncmt,op=MPI.SUM)
    
    for iproc in range(nprocs):
        if iproc == rank:
            outfile = "%s/FORCESOLUTION"%(param['OUTPUT_DIR'])

            if rank == 0:
                f = open(outfile,'w')
            else:
                f = open(outfile,'a')

            # write no. of forces
            idx = 0
            for iface in range(nfaces_loc):
                for i in range(NGLL):
                    for j in range(NGLL):
                        for idim in range(3):
                            force = [0,0,0]
                            force[idim] = 1.
                            id1 = offset_f + idx
                            stf_file = "%s/force_stf_%d.bin"%(param['OUTPUT_DIR'],id1)
                            r,lat,lon = cart2sph(*cords[startid + iface,i,j,:])
                            location = (lon,lat,(EARTH_RADIUS_M-r)/1000.0)
                            _write_force_file(
                                f,
                                location,
                                force,
                                stf_file,
                                stf[iface,i,j,idim,:]
                            )
                            idx += 1

                            if only_eq_force:
                                id1 += 1
                                stf_file = "%s/force_stf_eq_%d.bin"%(param['OUTPUT_DIR'],id1)
                                _write_force_file(
                                    f,
                                    location,
                                    force,
                                    stf_file,
                                    stf1[iface,i,j,idim,:]
                                )
                                idx += 1
            f.close()

            # write equivalent moment tensor if required
            if only_eq_force: continue

            outfile = "%s/CMT_SOLUTION"%(param['OUTPUT_DIR'])
            if rank == 0:
                f = open(outfile,'w')
            else:
                f = open(outfile,'a')
            idx = 0
            for iface in range(nfaces_loc):
                for i in range(NGLL):
                    for j in range(NGLL):
                        for idim in range(6):
                            id1 = offset_mt + idx
                            if id1 == 0:
                                f.write("PDE  1999 01 01 00 00 00.00  67000 67000 -25000 4.2 4.2 CPML_test\n")
                            mt = np.zeros((6,),dtype=float)
                            mt[idim] = 1.0
                            r,lat,lon = cart2sph(*cords[startid + iface,i,j,:])
                            location = (lon,lat,(EARTH_RADIUS_M-r)/1000.0)
                            _write_mt_file(
                                f,
                                location,
                                mt,
                                "%s/mt_stf_%d.bin"%(param['OUTPUT_DIR'],id1),
                                stf_mt[iface,i,j,idim,:]
                            )
                            idx += 1
            f.close()
        
        # synchronize
        comm.Barrier()
    db.close()
                        
def coupling_cart_stacey(param: dict[str, Any]) -> None:
    """
    stacey coupling in cartesian system
    
    :param param: Description
    :type param: dict
    """

    # Use actual processor numbers so gaps and empty boundary files are valid.
    files = [(int(match.group(1)),path)
             for path in Path(param['SPECFEM_DB']).glob('proc*_normal.txt')
             if (match := re.fullmatch(r'proc(\d+)_normal\.txt', path.name))]
    files.sort()
    if not files:
        raise FileNotFoundError(f'No Cartesian boundary files in {param["SPECFEM_DB"]}')
    ntasks = len(files)
    boundary_mode = (Path(param['AXISEM_DIR']) / 'MZZ/Data/boundary_wavefields.nc4').is_file()
    first_iface_by_proc = {}
    if boundary_mode:
        nfaces = 0
        for proc,path in files:
            with path.open() as stream:
                next(stream,None)
                npoints = sum(bool(line.strip()) for line in stream)
            if npoints % NGLL2:
                raise ValueError(f'{path}: expected 25 ordered points per face')
            first_iface_by_proc[proc] = nfaces
            nfaces += npoints // NGLL2

        db = AxiBasicDB()
        db.read_basic(str(Path(param['AXISEM_DIR']) / 'MZZ/Data/axisem_output.nc4'))
        if nfaces != len(db.face_phase):
            raise ValueError('Cartesian processor face counts differ from AxiSEM boundary faces')

    # mpi 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # allocate tasks
    startid,endid = allocate_task(ntasks,nprocs,rank)
    
    # time window 
    t1 = np.arange(param['nt']) * param['dt'] + param['t0']
    for i in range(startid,endid+1):
        proc = files[i][0]
        args = (proc,
                param['AXISEM_DIR'],
                param['SPECFEM_DB'],
                param['OUTPUT_DIR'],
                t1,
                param['UTM_ZONE'])
        if boundary_mode:
            get_field_proc_cart_boundary(args + (first_iface_by_proc[proc],))
        else:
            get_field_proc_cart(args)

def coupling_cube2sph(param: dict[str, Any]) -> None:
    """
    cube to spherical coupling
    
    :param param: Description
    :type param: dict
    """

    # SPECFEM processor numbers also determine the global boundary face order.
    point_files = sorted(Path(param['SPECFEM_DB']).glob('proc*_wavefield_discontinuity_points'))
    proc_ids = [int(match.group(1)) for path in point_files
                if (match := re.fullmatch(r'proc(\d+)_wavefield_discontinuity_points', path.name))]
    if not proc_ids:
        raise FileNotFoundError(f'No cube2sph point files in {param["SPECFEM_DB"]}')
    ntasks = len(proc_ids)

    boundary_file = Path(param['AXISEM_DIR']) / 'MZZ/Data/boundary_wavefields.nc4'
    face_by_proc = {}
    if boundary_file.is_file():
        blocks, sources = _cube2sph_faces(param['SPECFEM_DB'])
        db = AxiBasicDB()
        db.read_basic(str(boundary_file))
        _validate_boundary_faces(db, blocks)
        for proc, points in blocks:
            face_by_proc[proc] = ([], points)
        for iface, (proc, _) in enumerate(sources):
            face_by_proc[proc][0].append(iface)

    # mpi 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # allocate tasks
    startid,endid = allocate_task(ntasks,nprocs,rank)
    
    # time window 
    t1 = np.arange(param['nt']) * param['dt'] + param['t0']
    for i in range(startid,endid+1):
        proc = proc_ids[i]
        args = (proc,
                param['AXISEM_DIR'],
                param['SPECFEM_DB'],
                param['OUTPUT_DIR'],
                t1,
                param['DOWN_SAMPLING'])

        if face_by_proc:
            if proc not in face_by_proc:
                raise ValueError(f'No boundary face file for SPECFEM processor {proc}')
            args += face_by_proc[proc] + (proc_ids[0],)
            get_wavefield_sph_boundary(args)
        else:
            get_wavefield_sph(args)
