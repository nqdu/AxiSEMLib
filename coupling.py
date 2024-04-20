from database import AxiBasicDB
import numpy as np 
import os 
import h5py 
from mpi4py import MPI
from scipy.interpolate import interp1d 
import sys 
from utils import cart2sph,diff1,allocate_task

def get_trac_proc(args):
    iproc,basedir,coordir,outdir,tvec = args
    datadir = coordir
    filename1 = datadir + "proc%06d_wavefield_discontinuity_faces"%iproc
    outbin = "%s/proc%06d_traction.bin"%(outdir,iproc)
    if os.path.getsize(filename1) == 0:
        f = open(outbin,"wb")
        f.close()
        return 0
    f = h5py.File("%s/traction_proc%06d.h5"%(outdir,iproc),"w")

    # read database
    db = AxiBasicDB()
    db.read_basic(basedir + "/MZZ/Data/axisem_output.nc4")
    db.set_iodata(basedir)

    data = np.loadtxt(filename1,ndmin=2)
    r,stla,stlo = cart2sph(data[:,0],data[:,1],data[:,2])
    stel = -6371000 + r

    # create dataset
    t0 = np.arange(db.nt) * db.dtsamp - db.shift
    if tvec is None:
        t1 = np.arange(20000) * 0.05 - db.shift 
    else:
        t1 = tvec
    nt1 = len(t1)
    dset = f.create_dataset("field",(nt1,len(r),3),dtype=np.float32,chunks=True)
    field = np.zeros((nt1,3),dtype=np.float32)

    if iproc == 0: print("synthetic traction ...")

    for ir in range(len(stla)):
        #print(f"synthetic traction for point {ir+1} in proc {iproc} ...")
        sig_xyz = db.syn_stress(stla[ir],stlo[ir],stel[ir],basedir + '/CMTSOLUTION')
        Tx = np.zeros((db.nt)); Ty = Tx *  1.; Tz = Tx * 1. 

        nx = data[ir,3]; ny = data[ir,4]; nz = data[ir,5]
        Tx = sig_xyz[0,:] * nx + sig_xyz[5,:] * ny + sig_xyz[4,:] * nz 
        Ty = sig_xyz[5,:] * nx + sig_xyz[1,:] * ny + sig_xyz[3,:] * nz 
        Tz = sig_xyz[4,:] * nx + sig_xyz[3,:] * ny + sig_xyz[2,:] * nz 

        field[:,0] = interp1d(t0,Tx,bounds_error=False,fill_value=0.)(t1)
        field[:,1] = interp1d(t0,Ty,bounds_error=False,fill_value=0.)(t1)
        field[:,2] = interp1d(t0,Tz,bounds_error=False,fill_value=0.)(t1)

        # save to hdf5 file
        dset[:,ir,:] = field
        
    f.close()

    # create binary 
    os.system("h5dump -d 'field' -b LE -o %s %s/traction_proc%06d.h5" %(outbin,outdir,iproc))
    os.remove("%s/traction_proc%06d.h5"%(outdir,iproc))

def get_displ_proc(args):
    iproc,basedir,coordir,outdir,tvec = args 
    datadir = coordir
    
    filename1 = datadir + "proc%06d_wavefield_discontinuity_points"%iproc
    
    outbin = "%s/proc%06d_displ.bin"%(outdir,iproc)
    if os.path.getsize(filename1) == 0:
        os.system(f":> {outbin}")
        return 0
    f = h5py.File("%s/displ_proc%06d.h5"%(outdir,iproc),"w")
    
    # read database
    db = AxiBasicDB()
    db.read_basic(basedir + "/MZZ/Data/axisem_output.nc4")
    db.set_iodata(basedir)
    
    data = np.loadtxt(filename1,ndmin=2)
    r,stla,stlo = cart2sph(data[:,0],data[:,1],data[:,2])
    stel = -6371000 + r

    # create dataset
    t0 = np.arange(db.nt) * db.dtsamp - db.shift
    if tvec is None:
        t1 = np.arange(20000) * 0.05 - db.shift 
    else:
        t1 = tvec
    nt1 = len(t1)
    dt1 = t1[1] - t1[0]
    dset = f.create_dataset("field",(nt1,len(r),6),dtype=np.float32,chunks=True)

    # allocate space
    field = np.zeros((nt1,6))
    
    if iproc == 0: print("synthetic displ/accel ...")

    for ir in range(len(stla)):
        #print(f"synthetic displ/accel for point {ir+1} in proc {iproc} ...")
        ux1,uy1,uz1 = db.syn_seismo(stla[ir],stlo[ir],stel[ir],'xyz',basedir + '/CMTSOLUTION')

        field[:,0] = interp1d(t0,ux1,bounds_error=False,fill_value=0.)(t1)
        field[:,1] = interp1d(t0,uy1,bounds_error=False,fill_value=0.)(t1)
        field[:,2] = interp1d(t0,uz1,bounds_error=False,fill_value=0.)(t1)
        field[:,3] = diff1(diff1(field[:,0],dt1),dt1)
        field[:,4] = diff1(diff1(field[:,1],dt1),dt1)
        field[:,5] = diff1(diff1(field[:,2],dt1),dt1)

        # save to hdf5 file
        dset[:,ir,:] = field
    
    f.close()
    os.system(f"h5dump -d 'field' -b LE -o {outbin} {outdir}/displ_proc%06d.h5"%iproc)
    os.remove(f"{outdir}/displ_proc%06d.h5"%(iproc))

def main():
    # check input 
    if len(sys.argv) !=7:
        print("Usage: ./this h5dir coor_dir t0 dt nt outdir")
        exit(1)
    basedir = sys.argv[1]
    coordir = sys.argv[2]
    t0,dt = map(lambda x: float(x),sys.argv[3:5])
    nt = int(sys.argv[5])
    outdir = sys.argv[6]
    os.system(f'mkdir -p {outdir}')

    # find how many procs used in specfem
    filenames = os.listdir(coordir)
    ntasks = 0
    for f in filenames:
        if 'wavefield_discontinuity_faces' in f:
            ntasks += 1

    # alloc task to all mpi procs
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    startid,endid = allocate_task(ntasks,nprocs,rank)

    # time window
    t1 = np.arange(nt) * dt + t0

    #basedir = '/home/l/liuqy/nqdu/scratch/axisem/SOLVER/ak135'
    for i in range(startid,endid+1):
        args = (i,basedir,coordir,outdir,t1)
        get_trac_proc(args)
        get_displ_proc(args)
    
    MPI.Finalize()

main()
