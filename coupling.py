from database import AxiBasicDB
import numpy as np 
import os 
import h5py 
from mpi4py import MPI
from scipy.interpolate import interp1d 
import sys 

# import matplotlib.pyplot as plt 

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi

    elev = np.rad2deg(elev)
    az = np.rad2deg(az)
    return r, elev, az

def diff1(u,dt):
    u1 = u * 0. 
    nt = len(u)
    u1[1:nt-1] = (u[2:] - u[0:nt-2]) / (2 * dt)
    u1[0] = (u[1] - u[0]) / dt 
    u1[-1] = (u[nt-1] - u[nt-2]) / dt 

    return u1

def get_trac_proc(args):
    iproc,basedir,coordir,outdir,tvec = args
    datadir = coordir
    filename1 = datadir + "proc%06d_wavefield_discontinuity_faces"%iproc
    outbin = "%s/proc%06d_traction.bin"%(outdir,iproc)
    if os.path.getsize(filename1) == 0:
        os.system(":> %s"%outbin)
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

def allocate_task(ntasks,nprocs,myrank):
    sub_n = ntasks // nprocs
    num_larger_procs = ntasks - nprocs * sub_n
    startid = 0
    if myrank < num_larger_procs:
        sub_n = sub_n + 1
        startid = 0 + myrank * sub_n
    elif sub_n > 0 : 
        startid = 0 + num_larger_procs + myrank * sub_n
    else : #// this process has only zero elements
        startid = -1
        sub_n = 0
    
    endid = startid + sub_n - 1

    return startid,endid

def get_time_window(basedir,phase,tb,te):
    from obspy.taup import TauPyModel
    model = TauPyModel("ak135")

    # read stations
    sta_dat = np.loadtxt(basedir + "/MZZ/STATIONS",usecols=[2,3])
    stla = np.mean(sta_dat[:,0])
    stlo = np.mean(sta_dat[:,1])

    # read events
    cmtfile = basedir + "/CMTSOLUTION"
    f = open(cmtfile,"r")
    line = f.readline()
    info = line.split()
    f.close()
    evla,evlo,evdp = map(lambda x :float(x),info[6:9])

    # compute travel time
    t0 = model.get_travel_times_geo(evdp,evla,evlo,stla,stlo,  \
                phase_list=list(phase))[0].time 

    # get target time window 
    dt1 = 0.05
    nt1 = int((te + tb) / dt1) 
    time_vec  = np.arange(nt1) * dt1 + t0 - tb 
        
    return time_vec 

def main():
    # check input 
    if len(sys.argv) !=7:
        print("Usage: ./this h5dir coor_dir phase tb te outdir")
        exit(1)
    basedir = sys.argv[1]
    coordir = sys.argv[2]
    phase = sys.argv[3]
    tb,te = map(lambda x: float(x),sys.argv[4:6])
    outdir = sys.argv[6]
    os.system(f'mkdir -p {outdir}')

    # alloc task
    ntasks = 120
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    startid,endid = allocate_task(ntasks,nprocs,rank)

    # time window
    t1 = get_time_window(basedir,phase,tb,te)
    #t1 = np.arange(72000) * 0.05

    #basedir = '/home/l/liuqy/nqdu/scratch/axisem/SOLVER/ak135'
    for i in range(startid,endid+1):
        args = (i,basedir,coordir,outdir,t1)
        get_trac_proc(args)
        get_displ_proc(args)
    
    MPI.Finalize()
# def main():
#     # read axisem database
#     basedir = '/home/l/liuqy/nqdu/scratch/axisem/SOLVER/ak135'
#     db = AxiBasicDB(basedir + "/MZZ/Data/axisem_output.nc4")
#     data = np.loadtxt("STATION",usecols = [2,3,4])
#     names = np.loadtxt("STATION",usecols = [1,0],dtype=str)
#     stla = data[:,0]; stlo = data[:,1]; stel = data[:,2]
#     nr = len(stla)

#     for ir in range(5):
#         ux,uy,uz = db.syn_seismo(basedir,stla[ir],stlo[ir],stel[ir],'enz','CMTSOLUTION')
#         print(ir)
#         t = np.arange(db.nt) * db.dtsamp - db.shift

#         name = "/home/l/liuqy/nqdu/scratch/axisem/SOLVER/ak135/Data_Postprocessing/SEISMOGRAMS/" +   \
#                 names[ir,1] + "_" + names[ir,0] + "_disp_post_mij_conv0000_"

#         data_z = np.loadtxt(name + "Z.dat")
#         data_e = np.loadtxt(name + "E.dat")
#         data_n = np.loadtxt(name + "N.dat")
#         t = np.arange(db.nt) * db.dtsamp - db.shift

#         plt.figure(1,figsize=(14,15))
#         plt.subplot(3,1,1)
#         plt.plot(data_z[:,0],data_z[:,1],color='red')
#         plt.plot(t,uz[:])

#         plt.subplot(3,1,2)
#         plt.plot(t,ux[:])
#         plt.plot(data_e[:,0],data_e[:,1],color='red')

#         plt.subplot(3,1,3)
#         plt.plot(data_n[:,0],data_n[:,1],color='red')
#         plt.plot(t,uy[:])

#         outname = names[ir,0] + "_" + names[ir,1] + ".jpg"
#         plt.savefig(outname)

#         plt.clf()
    

main()
