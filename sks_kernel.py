from database import AxiBasicDB,rotation_matrix,rotate_tensor2
import numpy as np 
import matplotlib.pyplot as plt 
import os 
#import cProfile

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi

    elev = np.rad2deg(elev)
    az = np.rad2deg(az)
    return az,elev,r

def sph2cart(lon,lat,r):
    elevation = np.deg2rad(lat)
    azimuth = np.deg2rad(lon)
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z

def get_sgt(db:AxiBasicDB,lat,lon,depth,idx=None):
    """
    compute strain green's tensor at a given location

    Parameters:
    ======================================
    lat,lon,depth: float
        in deg/m 
    db: AxiBasicDB
        axisem database 
    
    Returns:
    =======================================
    strain: np.ndarray
        shape(3,6,nt), E_{ijn}^{rs} strain green's tensor, 
        the coordinate system is ij(size 6) -> (xr,yr,zr), n (size 3) -> (xs,ys,zs)
    """
    # rotate mat from (xr,yr,zr) to xyz
    Rr = rotation_matrix(np.deg2rad(90-lat),np.deg2rad(lon))
    
    # comptue rotated theta and phi
    theta,phi = db.compute_tp_recv(lat,lon)
    R1 = np.eye(3) # rotate from (s,phi,z) to (xs,ys,zs)
    R1[0,:2] = [np.cos(phi),-np.sin(phi)]
    R1[1,:2] = [np.sin(phi),np.cos(phi)]

    # locate point 
    r = 6371000 - depth
    sr = r * np.sin(theta)
    zr = r * np.cos(theta)
    elemid,xi,eta = db._locate_elem(sr,zr)

    # get basic strain
    datadir = db.ncfile_dir
    strain_z = db._get_strain(elemid,xi,eta,datadir + "/PZ/Data/axisem_fields.h5")
    strain_x = db._get_strain(elemid,xi,eta,datadir + "/PX/Data/axisem_fields.h5")

    # alloc space
    nt = strain_z.shape[-1]
    eps_out = np.zeros((3,6,nt))

    # loop every source type
    for i,stype in enumerate(['PX','PY','PZ']):
        #print("synthetic strain tensor for  ... %s" %(stype))
        # get basic waveform

        # parameters
        a = 0.; b = 0.
        eps0 = None
        if stype == "PZ":
            a = 1. 
            b = 0.  
            eps0 = strain_z
        else:
            eps0 = strain_x

            # cos/terms
            cosphi = np.cos(phi); sinphi = np.sin(phi)
            fx = 0.; fy = 0.
            if stype == 'PX':
                fx = 1.; fy = 0.
            else:
                fx = 0.; fy = 1.
            a = fx * cosphi + fy * sinphi 
            b = -fx * sinphi + fy * cosphi
            
        # add contribution from each term
        eps_out[i,0:3,:] += eps0[0:3,:] * a
        eps_out[i,4,:] += eps0[4,:] * a
        eps_out[i,3,:] += eps0[3,:] * b
        eps_out[i,5,:] += eps0[5,:] * b
    
    # now the strain is in computation coordinates, we rotate it to local coordinates at this point
    R1 = Rr.T @ db.rot_mat @ R1
    for i in range(3):
        eps_out[i,...] = rotate_tensor2(eps_out[i,...],R1) # 6 -> (xr,yr,zr)
    
    return elemid,xi,eta,eps_out

def compute_sks_kl(db:AxiBasicDB,dbdiff:AxiBasicDB,db0:AxiBasicDB,
                   xs,xr,x,cmtfile='CMTSOLUTION'):
    """
    compute duT for a given source/receiver/point
    Parameters: 
    ============================================

    db: AxiBasicDB
        database, the source depth = x[3] / 1000
    dbdiff: AxiBasicDB
        database for FD calculation, the source depth = (x[3] + dz) / 1000
    db0: AxiBasicDB
        database, the source depth = 0.
    xs,xr,x: np.array(3)
        source/receiver/point coordinates, in lon,lat (deg) and depth(m)

    """
    # load libs 
    from sem_funcs import lagrange_interpol_2D_td

    # extract coordinates
    evlo,evla,evdp = xs 
    stlo,stla,_ = xr 
    lon,lat,depth = x 

    # check if the depth is the same as source depth in db 
    assert(abs(db.evdp * 1000 - depth) <= 0.1)
    
    # get moment tensor
    mzz,mxx,myy,mxz,myz,mxy = db.read_cmt(cmtfile)
    mij = np.array([mxx,myy,mzz,2 * myz,2 * mxz,2 * mxy])

    # compute rotated station backazimuth
    db0.set_source(stla,stlo)
    theta,baz = db0.compute_tp_recv(lat,lon)
    if baz < np.pi:
        baz = np.pi - baz
    print(np.rad2deg(theta))

    # get sgt from receiver to x
    elemid,xi,eta,sgt_xr = get_sgt(db0,lat,lon,depth) # 3 -> (xr,yr,zr) 6 -> (t,p,r)
    nt = sgt_xr.shape[-1]
    
    # only keep T component
    ETxr =  sgt_xr[0,...] * np.sin(baz) + sgt_xr[1,...] * np.cos(baz)

    # rotation matrix from s to x
    RxT = rotation_matrix(np.deg2rad(90-lat),np.deg2rad(lon)).T
    Rs = rotation_matrix(np.deg2rad(90-evla),np.deg2rad(evlo))
    Rs = RxT @ Rs 

    # synthetic phi field
    db.set_source(lat,lon)
    _,_,_,sgt_sx1 = get_sgt(db,evla,evlo,evdp)
    data1 = np.einsum('ijk,j',sgt_sx1,mij) # data1 in (t,p,r)

    # compute derivative at t,p,r
    dx = 0.01 # 0.01 degree
    Esx = np.zeros((6,nt),dtype=np.float32) # E_{k,l}(t)
    E = np.zeros((3,3,nt),dtype=np.float32) # u_{k,l}(t), without christoffel symbol
    x0 = np.array([lat,lon])
    for k in range(2): # only for lat/lon
        # synthetic seismograms for (x + dx/2)
        x1 = x0 * 1.
        x1[k] = x0[k] + dx
        lat1,lon1 = x1
        db.set_source(lat1,lon1)
        #Rx1 = rotation_matrix(np.deg2rad(90-lat1),np.deg2rad(lon1))
        _,_,_,sgt_sx2 = get_sgt(db,evla,evlo,evdp) # 3-> (x+dx local) 6-> (xs,ys,zs)
        data2 = np.einsum('ijk,j',sgt_sx2,mij)
        #data2 = Rx1 @ data2 # rotate to xyz coordinates
        E[k,:,:] = (data2 - data1) / dx * 180 / np.pi 
    E[0,...] = - E[0,...] # dE / dtheta
    
    # for depth
    dbdiff.set_source(lat,lon)
    _,_,_,sgt_sx2 = get_sgt(dbdiff,evla,evlo,evdp)
    data2 = np.einsum('ijk,j',sgt_sx2,mij)
    data2 = RxT.T @ data1 
    dx = (db.evdp - dbdiff.evdp) * 1000
    E[2,:,:] = -(data1 - data2) / dx # dr = -ddepth

    # covert displ_grad to strain
    # Dahlen and Tromp 1998, A139
    # E_{i,j} = du_j / dx_i
    # now Esx is in local coordinates of x 
    csctheta = 1. / np.cos(np.deg2rad(lat))
    cottheta = 1. / np.tan(np.deg2rad(90 - lat))
    rinv = 1. / (6371000 - depth)
    Esx[0,:] = rinv * (data1[2,:] + E[0,0,:])  # E_{t,t}
    Esx[1,:] = rinv * (csctheta * E[1,1,:] + data1[2,:] + data1[0,:] * cottheta)  # E_{p,p}
    Esx[2,:] = E[2,2,:] # u_{r,r}
    Esx[3,:] = E[2,1,:] + rinv * (E[1,2,:] * csctheta - data1[1,:]) # E_{r,p}
    Esx[4,:] = E[2,0,:] + rinv * (E[0,2,:] - data1[0,:]) # E_{r,t}
    Esx[5,:] = rinv * (E[0,1,:] + csctheta * E[1,0,:] - data1[1,:] * cottheta) # E_{t,p}
    Esx[3:,:] *= 0.5 

    # integrate to get heaviside response
    Esx = np.cumsum(Esx,axis=-1) * db.dtsamp

    # get elastic parameters
    ngll = db0.ngll
    xmu = np.zeros((1,db0.ngll,db0.ngll),dtype='f4',order='F')
    xrho = np.zeros((1,db0.ngll,db0.ngll),dtype='f4',order='F')
    for iz in range(ngll):
        for ix in range(ngll):
            iglob = db0.ibool[elemid,iz,ix]
            xmu[0,ix,iz] = db0.mu[iglob]
            xrho[0,ix,iz] = db0.rho[iglob]

    # gll/glj array
    sgll = db0.gll
    zgll = db0.gll 
    if db0.axis[elemid]:
        sgll = db0.glj 
    mu = lagrange_interpol_2D_td(sgll,zgll,xmu,xi,eta)[0]
    rho = lagrange_interpol_2D_td(sgll,zgll,xrho,xi,eta)[0]

    # compute dun, note that Esx(3,3,nt) ETxr(6,nt)
    # ETxr/Esx: tt,pp,rr,rp,rt,tp
    # [(hrog - hoo)* Hrrpg - 2(hrro * Hrpg - hrr* Hrpg)]
    k_gc = (ETxr[0,:] - ETxr[1,:]) * Esx[2,:] - 2. * (
            ETxr[4,:] * Esx[4,:] - ETxr[3,:] * Esx[3,:])
    k_gc = -k_gc * 2. * mu 
    k_gs = ETxr[5,:] * Esx[2,:] - ETxr[4,:] * Esx[3,:] - ETxr[3,:] * Esx[4,:]
    k_gs = k_gs * 4 * mu

    # integral
    #duT = np.cumsum(duT) * db.dtsamp
    return k_gc,k_gs,Esx,ETxr

def main():
    # read db of single force 
    basedir = '/home/nqdu/scratch/axisem/SOLVER/'
    datadir = basedir + '/prem_10s_0/'
    datadir0 = basedir + '/prem_10s_0/'
    db = AxiBasicDB(datadir + "/PZ/Data/axisem_output.nc4")
    db0 = AxiBasicDB(datadir0 + "/PZ/Data/axisem_output.nc4")

    # get all depth for source
    filenames = os.listdir(basedir)
    src_depth = []
    for f in filenames:
        if 'prem' not in f:continue
        d = f.split('/')[-1].split('_')[-1]
        src_depth.append(int(d))
    src_depth = np.sort(src_depth)

    # station and events
    #xs = np.array([178.48,-26.04,552000])
    xr = np.array([122.629,43.3034,0.])
    xs = np.array([82.629,43.3034,0.])

    # 35-55 115-135 0.5°
    z0 = 0
    #x = np.array([45.,125.,z0 * 1000])
    x =np.array([82.629,43.3034,z0 * 1000]) 
    idx = np.argsort(abs(z0-src_depth))
    datadir_diff = basedir + '/prem_10s_' + str(src_depth[idx[1]]) + "/"
    dbdiff = AxiBasicDB(datadir_diff + "/PZ/Data/axisem_output.nc4")

    # compute
    #with cProfile.Profile() as pr:
    #k_gc,k_gs,Esx,ETxr = compute_sks_kl(db,dbdiff,db0,xs,xr,x)
    db0.set_source(xr[1],xr[0])
    _,_,_,sgt_xr = get_sgt(db0,x[1],x[0],x[-1])
    #    pr.print_stats('cumtime')
    #t = np.arange(db.nt) * db.dtsamp - db.shift
    t = np.arange(db.nt) * db.dtsamp
    idx = np.logical_and(t > 300,t < 1500)

    # filter
    from scipy.signal import butter,sosfiltfilt
    freqmin = 0.002; freqmax = 0.08
    sos = butter(4,[freqmin,freqmax],btype='bandpass',output='sos',fs=1. / db.dtsamp)
    data = sosfiltfilt(sos,sgt_xr[2,2,:])

    plt.figure(1,figsize=(14,4))
    plt.plot(t[idx],data[idx])
    plt.savefig("test.jpg")


def main1():
    # read db of single force 
    datadir = '/mnt/d/prem_aniso_10_crust_0/'
    #datadir = '/mnt/c/Documents/workshop/sgt_sks/axisem/SOLVER/prem'
    db = AxiBasicDB(datadir + "/PZ/Data/axisem_output.nc4")
    db.set_source(0.,0)
    t = np.arange(db.nt) * db.dtsamp - db.shift
    print(np.rad2deg(db.compute_tp_recv(-35,128)))

    # given place
    lat0 = 37.91; lon0 = -77.93; depth0 = 12000.

    # read cmt solution
    cmtfile = "CMTSOLUTION"
    mzz,mxx,myy,mxz,myz,mxy = db.read_cmt(cmtfile)
    mij = np.array([mxx,myy,mzz,2*myz,2*mxz,2*mxy])
    
    # synthetic from sgt
    eps_out = get_sgt(db,lat0,lon0,depth0)
    nt = eps_out.shape[-1]

    # synthetic seismograms by using sgt
    data = np.zeros((3,nt))
    for i in range(3):
        for j in range(6):
            p = mij[j]
            if j >= 3:
                p = p * 1.
            data[i,:] += eps_out[i,j,:] * p
    data_z = data[2,:] *1. 
    data_n = -data[0,:]
    data_e = data[1,:]
    
    data1 = np.einsum('ijk,j',eps_out,mij)

    plt.figure(1,figsize=(14,16))
    plt.subplot(3,1,1)
    plt.plot(t,data1[2,:],linewidth=1.)
    plt.plot(t,data_z[:],ls='--',color='r',linewidth=1.)

    plt.subplot(3,1,2)
    plt.plot(t,data1[1,:],linewidth=1.)
    plt.plot(t,data_e[:],ls='--',color='r',linewidth=1.)

    plt.subplot(3,1,3)
    plt.plot(t,-data1[0,:],linewidth=1.)
    plt.plot(t,data_n[:],ls='--',color='r',linewidth=1.)
    plt.show()

main()
