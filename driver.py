from database import AxiBasicDB
import numpy as np 
import os  
from mpi4py import MPI
from utils import cart2sph,allocate_task
from utils import prefilt_interp
from FortranIO import FortranIO  
from jacobian import compute_jacobian_surface

def read_boundary_points(coordir:str,iproc:int):
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

def get_field_proc_cart(args,intp_method ='savgol'):
    from pyproj import Proj
    from utils import rotation_matrix,rotate_tensor2

    # snity check
    if intp_method not in ['savgol','linear']:
        print("Error: intp_method should be 'savgol' or 'linear'")
        return -1

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
    dt1 = t1[1] - t1[0]
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
    stlo,stla = p(xx,yy,inverse=True)
    r = zz + 6371000
    stel = -6371000 + r

    method = intp_method  # 'savgol' or 'linear'
    if iproc == 0: print("synthetic traction/velocity ...")
    for ir in range(npts):
        #print(f"synthetic traction for point {ir+1} of {npts} in proc {iproc} ...")

        # get rotation matrix from (xyz) to (enz)
        R = rotation_matrix(np.deg2rad(90-stla[ir]),np.deg2rad(stlo[ir]))
        tmp = R[:,1] * 1.
        R[:,1] = -R[:,0] * 1. # \hat{e}_n is -\hat{\theta}
        R[:,0] = tmp * 1.
        R = R.T

        # get stress 
        sig_xyz = db.syn_stress(stla[ir],stlo[ir],stel[ir],basedir + '/CMTSOLUTION')
        sig_xyz = rotate_tensor2(sig_xyz,R)
        Tx = np.zeros((db.nt)); Ty = Tx *  1.; Tz = Tx * 1. 

        # synthetic displ in enz, note that enz is specfem3d's (xyz)
        ux,uy,uz = db.syn_seismo(stla[ir],stlo[ir],stel[ir],'enz',basedir + '/CMTSOLUTION')

        # get velocity
        _,veloc_axi[:,ir,0] = prefilt_interp(t0,ux,t1,
                                            method=method,
                                            fmax=1./db.dominant_T0,
                                            deriv=1)
        _,veloc_axi[:,ir,1] = prefilt_interp(t0,uy,t1,
                                            method=method,
                                            fmax=1./db.dominant_T0,
                                            deriv=1)
        _,veloc_axi[:,ir,2] = prefilt_interp(t0,uz,t1,
                                            method=method,
                                            fmax=1./db.dominant_T0,
                                            deriv=1)
                                             

        # traction
        nx = nnx[ir]; ny = nny[ir]; nz = nnz[ir]
        Tx = sig_xyz[0,:] * nx + sig_xyz[5,:] * ny + sig_xyz[4,:] * nz 
        Ty = sig_xyz[5,:] * nx + sig_xyz[1,:] * ny + sig_xyz[3,:] * nz 
        Tz = sig_xyz[4,:] * nx + sig_xyz[3,:] * ny + sig_xyz[2,:] * nz 

        trac_axi[:,ir,0],_ = prefilt_interp(t0,Tx,t1,
                                        method=method,
                                        fmax=1./db.dominant_T0,
                                        deriv=0)
        trac_axi[:,ir,1],_ = prefilt_interp(t0,Ty,t1,
                                        method=method,
                                        fmax=1./db.dominant_T0,
                                        deriv=0)
        trac_axi[:,ir,2],_ = prefilt_interp(t0,Tz,t1,
                                        method=method,
                                        fmax=1./db.dominant_T0,
                                        deriv=0)    

    # write file
    for i in range(nt1):
        f.write_record(veloc_axi[i,...],trac_axi[i,...])
        
    f.close()

def get_wavefield_sph(args,intp_method ='savgol'):
    """
    get wavefield (displ/accel/traction) on the injection boundaries in spherical system

    Parameters
    -------------------
    args: tuple
        (iproc,basedir,coordir,outdir,tvec,downsample)
    intp_method: str
        interpolation method: 'savgol' or 'linear'
    """
    # sanity check
    if intp_method not in ['savgol','linear']:
        print("Error: intp_method should be 'savgol' or 'linear'")
        return -1

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
    method = intp_method  # 'savgol' or 'linear'
    print(f"synthetic displ/accel for {file_disp} ...")
    for ir in range(npts):
        #print(f"synthetic displ/accel for point {ir+1} in proc {iproc} ...")
        ux1,uy1,uz1 = db.syn_seismo(stla[ir],stlo[ir],stel[ir],'xyz',basedir + '/CMTSOLUTION')

        # interpolate to t1 
        
        displ[:,ir,0],accel[:,ir,0] = prefilt_interp(t0,ux1,t1,
                                                     method=method,
                                                     fmax=1./db.dominant_T0,
                                                     deriv=2)
        displ[:,ir,1],accel[:,ir,1] = prefilt_interp(t0,uy1,t1,
                                                     method=method,
                                                     fmax=1./db.dominant_T0,
                                                     deriv=2)
        displ[:,ir,2],accel[:,ir,2] = prefilt_interp(t0,uz1,t1,
                                                     method=method,
                                                     fmax=1./db.dominant_T0,
                                                     deriv=2)

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

        tract[:,ir,0],_ = prefilt_interp(t0,Tx,t1,
                                        method=method,
                                        fmax=1./db.dominant_T0,
                                        deriv=0)
        tract[:,ir,1],_ = prefilt_interp(t0,Ty,t1,
                                        method=method,
                                        fmax=1./db.dominant_T0,
                                        deriv=0)
        tract[:,ir,2],_ = prefilt_interp(t0,Tz,t1,
                                        method=method,
                                        fmax=1./db.dominant_T0,
                                        deriv=0)

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

def _write_force_file(f,
                      cords:np.ndarray,
                      force:np.ndarray,
                      stf_file:str,
                      stf:np.ndarray):
    """
    write force file for cartesian coupling

    Parameters
    -------------------
    fio: FortranIO
        FortranIO object for writing
    cords: np.ndarray
        (nfaces, NGLL, NGLL, 3) array of (x,y,z) in m 
    stf: np.ndarray
        (nfaces_loc,NGLL, NGLL, 3,nt) equivalent force components on the surface element 
    startid: int
        starting face id for current proc 
    nfaces_loc: int
        number of faces for current proc 
    only_eq_force: bool
        whether only equivalent force is used 
    stf1: np.ndarray
        (nfaces_loc,NGLL, NGLL, 3,nt) equivalent force from moment tensor on the surface element 
    """

    f.write("FORCE 000\n")
    f.write("time shift:    0.\n")
    f.write("hdurorf0:    0.\n")
    f.write("latorUTM:  %.6f\n" % (cords[1]))
    f.write("longorUTM:  %.6f\n" % (cords[0]))
    f.write("depth:  %.6f\n" % (cords[2]))
    f.write("source time function:   0\n")
    f.write("factor force source:    1.0\n")
    f.write("component dir vect source E: %f" %(force[0]))
    f.write("component dir vect source N: %f" %(force[1]))
    f.write("component dir vect source Z: %f" %(force[2]))
    f.write(f"{stf_file}\n")
    f1 = open(stf_file,'wb')
    byte = stf.tobytes()
    f1.write(byte)
    f1.close()

def _write_mt_file(f,
        cords:np.ndarray,
        mt:np.ndarray,
        stf_file:str,
        stf:np.ndarray):
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
    f.write("\n")

def equivalent_force_cube2sph(param:dict):
    """
    equivalent force coupling in cube2sph system
    
    :param param: Description
    :type param: dict
    """
    # glob
    from glob import glob
    from jacobian import compute_jacobian_surface,moment_to_force

    # mpi 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # constants
    NGLL:int = 5

    # find how many procs used in specfem
    filenames = glob(param['SPECFEM_DB'] + '/proc*_wavefield_discontinuity_faces')
    npts = 0
    for f in filenames:
        # count no. of lines 
        with open(f,'r') as fin:
            npts += len(fin.readlines())
    if npts == 0:
        if rank == 0:
            print(f'please check proc*_wavefield_discontinuity_faces in {param["SPECFEM_DB"]}!')
        comm.Abort(1)
    
    # check if npts is divisible by NGLL^2
    if npts % (NGLL * NGLL) != 0:
        if rank == 0:
            print(f'Error: total no. of points {npts} is not divisible by {NGLL*NGLL}!')
        comm.Abort(1)

    # load everything into a coordinate system 
    nfaces = npts // (NGLL * NGLL)
    cords = np.zeros((nfaces,NGLL,NGLL,3),dtype=float) # x,y,z,nx,ny,nz
    norms = np.zeros((nfaces,NGLL,NGLL,3),dtype=float)
    iface=0
    for f in filenames:
        if os.path.getsize(f) == 0:
            continue
        # load data
        data = np.loadtxt(f,ndmin=2)

        # convert to spherical coordinates
        r,stla,stlo = cart2sph(data[:,0],data[:,1],data[:,2])
        stel = -6371000 + r 

        # store
        npts = len(r)
        nfaces1 = npts // (NGLL * NGLL)
        iface2 = iface + nfaces1
        cords[iface:iface2,:,0] = data[:,0].reshape((nfaces1,NGLL,NGLL))
        cords[iface:iface2,:,1] = data[:,1].reshape((nfaces1,NGLL,NGLL))
        cords[iface:iface2,:,2] = data[:,2].reshape((nfaces1,NGLL,NGLL))
        norms[iface:iface2,:,0] = data[:,3].reshape((nfaces1,NGLL,NGLL))
        norms[iface:iface2,:,1] = data[:,4].reshape((nfaces1,NGLL,NGLL))
        norms[iface:iface2,:,2] = data[:,5].reshape((nfaces1,NGLL,NGLL))
        iface = iface2

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
    t0 = np.arange(db.nt) * db.dtsamp + db.t0
    t1 = np.arange(param['nt']) * param['dt'] + param['t0']
    nt1 = len(t1)
    method = param['intp_method']  # 'savgol' or 'linear'

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
        offset = offset * 2 # for equivalent force only, no moment tensor

    # compute equivalent force on each face
    for iface in range(startid,endid+1):
        #print(f"compute equivalent force for face {iface+1} of {nfaces} in proc {rank} ...")
        for i in range(NGLL):
            for j in range(NGLL):
                # get spherical coordinates
                r,stla,stlo = cart2sph(cords[iface,i,j,0],
                                       cords[iface,i,j,1],
                                       cords[iface,i,j,2])
                stel = -6371000 + r 

                # get stress 
                sig_xyz = db.syn_stress(stla,stlo,stel,param['AXISEM_DIR'] + '/CMTSOLUTION')
                Tx = np.zeros((db.nt)); Ty = Tx *  1.; Tz = Tx * 1. 
                nx = norms[iface,i,j,0]; ny = norms[iface,i,j,1]; nz = norms[iface,i,j,2]
                Tx = sig_xyz[0,:] * nx + sig_xyz[5,:] * ny + sig_xyz[4,:] * nz 
                Ty = sig_xyz[5,:] * nx + sig_xyz[1,:] * ny + sig_xyz[3,:] * nz 
                Tz = sig_xyz[4,:] * nx + sig_xyz[3,:] * ny + sig_xyz[2,:] * nz 

                Tx,_ = prefilt_interp(t0,Tx,t1,
                                      method=method,
                                      fmax=1./db.dominant_T0,
                                      deriv=0)
                Ty,_ = prefilt_interp(t0,Ty,t1,
                                      method=method,
                                      fmax=1./db.dominant_T0,
                                      deriv=0)
                Tz,_ = prefilt_interp(t0,Tz,t1,
                                      method=method,
                                      fmax=1./db.dominant_T0,
                                      deriv=0)
                stf[iface,i,j,0,:] = - Tx 
                stf[iface,i,j,1,:] = - Ty
                stf[iface,i,j,2,:] = - Tz

                # get equivalent mt
                norm_xyz = np.array(norms[iface,i,j,:],dtype=float) 
                mt_xyz = db.syn_eq_moment(
                    stla,stlo,stel,
                    norm_xyz,
                    param['AXISEM_DIR'] + '/CMTSOLUTION'
                )

                # save mt 
                for k in range(6):
                    out,_ = prefilt_interp(t0,mt_xyz[k,:],t1,
                                          method=method,
                                          fmax=1./db.dominant_T0,
                                          deriv=0)
                    stf_mt[iface,i,j,k,:] = out 

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
                            id1 = offset_f[rank] + idx 
                            stf_file = "%s/force_stf_%d.bin"%(param['OUTPUT_DIR'],id1)
                            _write_force_file(
                                f,
                                cords[startid + iface,i,j,:],
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
                                    cords[startid + iface,i,j,:],
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
                            id1 = offset_mt[rank] + idx 
                            if id1 == 0:
                                f.write("PDE  1999 01 01 00 00 00.00  67000 67000 -25000 4.2 4.2 CPML_test\n")
                            mt = np.zeros((6,),dtype=float)
                            mt[idim] = 1.0
                            _write_mt_file(
                                f,
                                cords[startid + iface,i,j,:],
                                mt,
                                "%s/mt_stf_%d.bin"%(param['OUTPUT_DIR'],id1),
                                stf_mt[iface,i,j,idim,:]
                            )
                            idx += 1
            f.close()
        
        # synchronize
        comm.Barrier()
                        
def coupling_cart_stacey(param:dict):
    """
    stacey coupling in cartesian system
    
    :param param: Description
    :type param: dict
    """

    # find how many procs used in specfem
    filenames = os.listdir(param['SPECFEM_DB'])
    ntasks = 0
    for f in filenames:
        if '_normal.txt' in f:
            ntasks += 1
    if ntasks == 0:
        if rank == 0:
            print(f'please check proc*_normal.txt in {param["SPECFEM_DB"]}!')
        comm.Abort(1)

    # mpi 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # allocate tasks
    startid,endid = allocate_task(ntasks,nprocs,rank)
    
    # time window 
    t1 = np.arange(param['nt']) * param['dt'] + param['t0']
    for i in range(startid,endid+1):
        args = (i,
                param['AXISEM_DIR'],
                param['SPECFEM_DB'],
                param['OUTPUT_DIR'],
                t1,
                param['UTM_ZONE'])
        get_field_proc_cart(args,intp_method=param['intp_method'])

def coupling_cube2sph(param:dict):
    """
    cube to spherical coupling
    
    :param param: Description
    :type param: dict
    """

    # find how many procs used in specfem
    filenames = os.listdir(param['SPECFEM_DB'])
    ntasks = 0
    for f in filenames:
        if 'wavefield_discontinuity_points' in f:
            ntasks += 1
    if ntasks == 0:
        if rank == 0:
            print(f'please check proc*_wavefield_discontinuity_points in {param["SPECFEM_DB"]}!')
        comm.Abort(1)

    # mpi 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # allocate tasks
    startid,endid = allocate_task(ntasks,nprocs,rank)
    
    # time window 
    t1 = np.arange(param['nt']) * param['dt'] + param['t0']
    for i in range(startid,endid+1):
        args = (i,
                param['AXISEM_DIR'],
                param['SPECFEM_DB'],
                param['OUTPUT_DIR'],
                t1,
                param['DOWN_SAMPLING'])

        get_wavefield_sph(args,intp_method=param['intp_method'])