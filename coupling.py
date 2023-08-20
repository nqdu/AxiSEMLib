from database import AxiBasicDB,rotation_matrix
import numpy as np 
import os 
import h5py 
from multiprocessing import Pool
#import matplotlib.pyplot as plt 

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

def get_displ_proc(args):
    iproc,basedir = args 

    datadir = '../DATABASES_MPI/'
    filename1 = datadir + "proc%06d_wavefield_discontinuity_points"%iproc

    # create h5 file 
    f = h5py.File("output/displ_proc%06d.h5"%(iproc),"w")
    if os.path.getsize(filename1) == 0:
        f.close()
        return 0
    
    # read database
    db = AxiBasicDB(basedir + "/MZZ/Data/axisem_output.nc4")
    
    data = np.loadtxt(filename1,ndmin=2)
    r,stla,stlo = cart2sph(data[:,0],data[:,1],data[:,2])
    stel = -6371000 + r

    # create dataset
    dset = f.create_dataset("field",(db.nt,len(r),6))

    for ir in range(len(r)):
        print(f"synthetic for point {ir+1} in proc {iproc} ...")
        ux,uy,uz = db.syn_seismo(basedir,stla[ir],stlo[ir],stel[ir],'xyz','CMTSOLUTION')
        ax = diff1(diff1(ux,db.dtsamp),db.dtsamp); 
        ay = diff1(diff1(uy,db.dtsamp),db.dtsamp); 
        az = diff1(diff1(uz,db.dtsamp),db.dtsamp); 

        # save to hdf5 file
        dset[:,ir,0] = ux 
        dset[:,ir,1] = uy 
        dset[:,ir,2] = uz 
        dset[:,ir,3] = ax 
        dset[:,ir,4] = ay 
        dset[:,ir,5] = az 
    
    f.close()

    # create h5 file 
    filename1 = datadir + "proc%06d_wavefield_discontinuity_faces"%iproc
    f = h5py.File("output/traction_proc%06d.h5"%(iproc),"w")
    if os.path.getsize(filename1) == 0:
        f.close()
        return 0

    data = np.loadtxt(filename1,ndmin=2)
    r,stla,stlo = cart2sph(data[:,0],data[:,1],data[:,2])
    stel = -6371000 + r

    # create dataset
    dset = f.create_dataset("field",(db.nt,len(r),3))

    for ir in range(len(r)):
        print(f"synthetic for point {ir+1} in proc {iproc} ...")
        sig_xyz = db.syn_stress(basedir,stla[ir],stlo[ir],stel[ir],'xyz','CMTSOLUTION')

        # save to hdf5 file
        dset[:,ir,0] = ux 
        dset[:,ir,1] = uy 
        dset[:,ir,2] = uz 
        dset[:,ir,3] = ax 
        dset[:,ir,4] = ay 
        dset[:,ir,5] = az 
    
    f.close()


def main():
    nprocs = 120
    basedir = '/home/l/liuqy/nqdu/scratch/axisem/SOLVER/ak135'
    args = []
    for i in range(nprocs):
        args.append((i,basedir))
        if i == 15: break 
    
    # map 
    pool = Pool(40)
    pool.map(get_displ_proc,args)
    pool.close()
    pool.join()


# def main():
#     nproc = 120

#     # read axisem database
#     basedir = '/home/l/liuqy/nqdu/scratch/axisem/SOLVER/ak135'
#     db = AxiBasicDB(basedir + "/MZZ/Data/axisem_output.nc4")

#     for iproc in range(nproc):
#         datadir = '../DATABASES_MPI/'
#         filename1 = datadir + "proc%06d_wavefield_discontinuity_points"%iproc

#         # create h5 file 
#         f = h5py.File("output/displ_proc%06d.h5"%(iproc),"w")
#         if os.path.getsize(filename1) == 0:
#             f.close()
#             continue
#         data = np.loadtxt(filename1,ndmin=2)
#         r,stla,stlo = cart2sph(data[:,0],data[:,1],data[:,2])
#         r[0],stla[0],stlo[0] = 6371000,42.966350,131.010620
#         stel = -6371000 + r

#         # create dataset
#         dset = f.create_dataset("field",(db.nt,len(r),6))

#         for ir in range(len(r)):
#             print(f"synthetic for point {ir+1} in proc {iproc} ...")
#             ux,uy,uz = db.syn_seismo(basedir,stla[ir],stlo[ir],stel[ir],'enz','CMTSOLUTION')
            
#             t = np.arange(db.nt) * db.dtsamp - db.shift

#             name = "/home/l/liuqy/nqdu/scratch/axisem/SOLVER/ak135/Data_Postprocessing/SEISMOGRAMS/DB_EN01" + "_disp_post_mij_conv0000_"

#             data_z = np.loadtxt(name + "Z.dat")
#             data_e = np.loadtxt(name + "E.dat")
#             data_n = np.loadtxt(name + "N.dat")
#             t = np.arange(db.nt) * db.dtsamp - db.shift

#             plt.figure(1,figsize=(14,15))
#             plt.subplot(3,1,1)
#             plt.plot(t,uz[:],'k')
#             plt.plot(data_z[:,0],data_z[:,1],ls='--')

#             plt.subplot(3,1,2)
#             plt.plot(t,ux[:],'k')
#             plt.plot(data_e[:,0],data_e[:,1],ls='--')

#             plt.subplot(3,1,3)
#             plt.plot(t,uy[:],'k')
#             plt.plot(data_n[:,0],data_n[:,1],ls='--')

#             outname = 'AA' + "_" + str(ir+1) + ".jpg"
#             plt.savefig(outname)

#             exit(1)

#         f.close()
#         exit(1)
    

main()