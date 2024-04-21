from database import AxiBasicDB,rotation_matrix,rotate_tensor2
import numpy as np 
import os 
from scipy.io import FortranFile
from mpi4py import MPI
from scipy.interpolate import interp1d 
import sys 
from pyproj import Proj

def diff1(u,dt):
    u1 = u * 0. 
    nt = len(u)
    u1[1:nt-1] = (u[2:] - u[0:nt-2]) / (2 * dt)
    u1[0] = (u[1] - u[0]) / dt 
    u1[-1] = (u[nt-1] - u[nt-2]) / dt 

    return u1


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

def get_field_proc(args):
    # unpack input paramters
    iproc,basedir,coordir,outdir,tvec = args

    # read database
    db = AxiBasicDB()
    db.read_basic(basedir + "/MZZ/Data/axisem_output.nc4")
    db.set_iodata(basedir)

    # read boundary points
    xx,yy,zz,nnx,nny,nnz = read_boundary_points(coordir,iproc)
    npts = len(xx)

    # create dataset
    t0 = np.arange(db.nt) * db.dtsamp - db.shift
    if tvec is None:
        t1 = np.arange(20000) * 0.05 - db.shift 
    else:
        t1 = tvec
    dt1 = t1[1] - t1[0]
    nt1 = len(t1)

    # allocate space for veloc/traction
    veloc_axi = np.zeros((nt1,npts,3),'f4')
    trac_axi = np.zeros((nt1,npts,3),'f4')

    # open file
    outbin = "%s/proc%06d_sol_axisem"%(outdir,iproc)
    f = FortranFile(outbin,'w')
    if npts == 0:
        for i in range(nt1):
            f.write_record(veloc_axi[i,...])
            f.write_record(trac_axi[i,...])
        f.close()
        return 0

    # convert to spherical coordinates
    p = Proj(proj='utm',zone=10,ellps='WGS84')
    stlo,stla = p(xx,yy,inverse=True)
    r = zz + 6371000
    stel = -6371000 + r

    if iproc == 0: print("synthetic traction/velocity ...")
    for ir in range(npts):
        print(f"synthetic traction for point {ir+1} of {npts} in proc {iproc} ...")

        # get rotation matrix from (xyz) to (enz)
        R = rotation_matrix(np.deg2rad(90-stla[ir]),np.deg2rad(stlo[ir]))
        tmp = R[:,1] * 1.
        R[:,1] = -R[:,0] * 1. # \hat{n}_n is -\hat{\theta}
        R[:,0] = tmp * 1.
        R = R.T

        # get stress 
        sig_xyz = db.syn_stress(stla[ir],stlo[ir],stel[ir],basedir + '/CMTSOLUTION')
        sig_xyz = rotate_tensor2(sig_xyz,R)
        Tx = np.zeros((db.nt)); Ty = Tx *  1.; Tz = Tx * 1. 

        # synthetic displ in enz, note that enz is specfem3d's (xyz)
        ux,uy,uz = db.syn_seismo(stla[ir],stlo[ir],stel[ir],'enz',basedir + '/CMTSOLUTION')

        # get velocity
        ux = interp1d(t0,ux,bounds_error=False,fill_value=0.)(t1)
        uy = interp1d(t0,uy,bounds_error=False,fill_value=0.)(t1)
        uz = interp1d(t0,uz,bounds_error=False,fill_value=0.)(t1)
        veloc_axi[:,ir,0] = diff1(ux,dt1)
        veloc_axi[:,ir,1] = diff1(uy,dt1)
        veloc_axi[:,ir,2] = diff1(uz,dt1)

        # traction
        nx = nnx[ir]; ny = nny[ir]; nz = nnz[ir]
        Tx = sig_xyz[0,:] * nx + sig_xyz[5,:] * ny + sig_xyz[4,:] * nz 
        Ty = sig_xyz[5,:] * nx + sig_xyz[1,:] * ny + sig_xyz[3,:] * nz 
        Tz = sig_xyz[4,:] * nx + sig_xyz[3,:] * ny + sig_xyz[2,:] * nz 

        trac_axi[:,ir,0] = interp1d(t0,Tx,bounds_error=False,fill_value=0.)(t1)
        trac_axi[:,ir,1] = interp1d(t0,Ty,bounds_error=False,fill_value=0.)(t1)
        trac_axi[:,ir,2] = interp1d(t0,Tz,bounds_error=False,fill_value=0.)(t1)

    # write file
    for i in range(nt1):
        f.write_record(veloc_axi[i,...])
        f.write_record(trac_axi[i,...])
        
    f.close()


def main():
    from utils import allocate_task
    if len(sys.argv) !=7:
        print("Usage: ./this h5dir coor_dir t0 dt nt outdir")
        exit(1)
    basedir = sys.argv[1]
    coordir = sys.argv[2]
    t0,dt = map(lambda x: float(x),sys.argv[3:5])
    nt = int(sys.argv[5])
    outdir = sys.argv[6]
    os.system(f'mkdir -p {outdir}')

    # mpi 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # find how many procs used in specfem
    filenames = os.listdir(coordir)
    ntasks = 0
    for f in filenames:
        if '_normal.txt' in f:
            ntasks += 1
    if ntasks == 0:
        if rank == 0:
            print(f'please check proc*_normal.txt in {coordir}!')
        comm.Abort(1)

    # alloc task
    startid,endid = allocate_task(ntasks,nprocs,rank)

    # time window
    t1 = np.arange(nt) * dt + t0

    #basedir = '/home/l/liuqy/nqdu/scratch/axisem/SOLVER/ak135'
    for i in range(startid,endid+1):
        args = (i,basedir,coordir,outdir,t1)
        get_field_proc(args)
    
    MPI.Finalize()

main()
