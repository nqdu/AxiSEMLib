from database import AxiBasicDB,rotation_matrix,rotate_tensor2
import numpy as np 
import matplotlib.pyplot as plt 
import os 

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

def get_sgt(db:AxiBasicDB,lat,lon,depth):
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
        eps_out[i,...] = rotate_tensor2(eps_out[i,...],R1)
    
    return eps_out

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
    from obspy.geodetics import gps2dist_azimuth

    # extract coordinates
    evlo,evla,evdp = xs 
    stlo,stla,_ = xr 
    lon,lat,depth = x 

    # check if the depth is the same as source depth in db 
    assert(abs(db.evdp * 1000 - depth) > 0.1)
    
    # get moment tensor
    mzz,mxx,myy,mxz,myz,mxy = db.read_cmt(cmtfile)
    mij = np.array([mxx,myy,mzz,2 * myz,2 * mxz,2 * mxy])

    # get sgt from r to x
    db0.set_source(stla,stlo)
    sgt_xr = get_sgt(db0,lat,lon,depth)
    nt = sgt_xr.shape[-1]

    # rotation matrix from s to x
    RxT = rotation_matrix(np.deg2rad(90-lat),np.deg2rad(lon)).T
    Rs = rotation_matrix(np.deg2rad(90-evla),np.deg2rad(evlo))
    Rs = RxT @ Rs 

    # compute disp_grad from x to s
    dx = 500
    Esx = np.zeros((3,3,nt)) # u_{k,l}(t)
    x0 = np.array(sph2cart(x[0],x[1],6371000 - x[1])) # to cartesian
    for k in range(2): # only for lon/lat
        # synthetic seismograms for (x + dx/2)
        x1 = x0 * 1.
        x1[k] = x0[k] + dx
        xtemp = np.array(cart2sph(x1[0],x1[1],x1[2]))
        lon1,lat1,_ = xtemp 
        db.set_source(lat1,lon1)
        Rx1 = rotation_matrix(np.deg2rad(90-lat1),np.deg2rad(lon1))
        sgt_sx1 = get_sgt(db,evla,evlo,evdp)
        data1 = np.einsum('ijk,j',sgt_sx1,mij)
        data1 = Rx1 @ data1 # rotate to xyz coordinates

        # synthetic seismograms for (x - dx/2)
        x1[k] = x0[k] - dx
        xtemp = np.array(cart2sph(x1[0],x1[1],x1[2]))
        lon1,lat1,_ = xtemp 
        db.set_source(lat1,lon1)
        Rx1 = rotation_matrix(np.deg2rad(90-lat1),np.deg2rad(lon1))
        sgt_sx1 = get_sgt(db,evla,evlo,evdp)
        data2 = np.einsum('ijk,j',sgt_sx1,mij)
        data2 = Rx1 @ data2 # rotate to xyz coordinates
        
        Esx[k,:,:] = 0.5 * (data1 - data2) / dx
    
    # for depth
    db.set_source(lat,lon)
    sgt_sx1 = get_sgt(db,evla,evlo,evdp)
    data1 = np.einsum('ijk,j',sgt_sx1,mij)
    data1 = RxT.T @ data1 
    dbdiff.set_source(lat,lon)
    sgt_sx1 = get_sgt(dbdiff,evla,evlo,evdp)
    data2 = np.einsum('ijk,j',sgt_sx1,mij)
    data2 = RxT.T @ data1 
    dx = (db.evdp - dbdiff.evdp) * 1000
    Esx[2,:,:] = (data1 - data2) / dx

    # covert displ_grad to strain
    Esx = 0.5 * (Esx + np.transpose(Esx,(1,0,2)))
    print('haha')

    # now Esx is in global cartesian (xyz) coordinates
    # we rotate it to local coordinates of x 
    Esx = RxT @ Esx
    Esx = np.einsum('ik,jkl',RxT,Esx)

    # get elastic parameters
    # compute rotated station phi,theta
    theta,_ = db0.compute_tp_recv(lat,lon)
    sr,zr = db0.compute_local(theta,-depth)
    elemid,xi,eta = db0._locate_elem(sr,zr)
    ngll = db0.ngll
    xmu = np.zeros((1,db0.ngll,db0.ngll),dtype=float,order='F')
    xlam = np.zeros((1,db0.ngll,db0.ngll),dtype=float,order='F')
    for iz in range(ngll):
        for ix in range(ngll):
            iglob = db0.ibool[elemid,iz,ix]
            xmu[0,ix,iz] = db0.mu[iglob]
            xlam[0,ix,iz] = db0.lamda[iglob]
    # gll/glj array
    sgll = db0.gll
    zgll = db0.gll 
    if db0.axis[elemid]:
        sgll = db0.glj 
    mu = lagrange_interpol_2D_td(sgll,zgll,xmu,xi,eta)[0]
    lamb = lagrange_interpol_2D_td(sgll,zgll,xlam,xi,eta)[0]

    # compute dun, note that Esx(3,3,nt) sgt_xr(3,6,nt)
    k1 = 2. *(Esx[0,0,:] + Esx[1,1,:] + Esx[2,2,:]) * sgt_xr[:,0,:] # sgt shape(3,6,nt)
    k2 = 2. * np.sum(sgt_xr[:,:3,:],axis=1) * Esx[0,0,:]
    k3 = 4. * (sgt_xr[:,0,:] * Esx[0,0,:] + sgt_xr[:,5,:] * Esx[1,0,:] + 
               sgt_xr[:,4,:] * Esx[2,0,:])
    dun = 2 * mu * (k1 + k2 - k3) # shape(3,nt)

    # compute T component du_T 
    phi = gps2dist_azimuth(evla,evlo,stla,stlo,6371000.,0.0)[1]
    phi = np.deg2rad(phi)
    duT = dun[0,:] * np.sin(phi) + dun[1,:] * np.cos(phi)

    # integral
    #duT = np.cumsum(duT) * db.dtsamp
    return duT

def main():
    # read db of single force 
    datadir = '/mnt/d/prem_aniso_10_crust_100/'
    datadir0 = '/mnt/d/prem_aniso_10_crust_0/'
    db = AxiBasicDB(datadir + "/PZ/Data/axisem_output.nc4")
    db0 = AxiBasicDB(datadir0 + "/PZ/Data/axisem_output.nc4")

    # get all depth for source
    filenames = os.listdir('/mnt/d/')
    src_depth = []
    for f in filenames:
        if 'prem' not in f:continue
        d = f.split('/')[-1].split('_')[-1]
        src_depth.append(int(d))
    src_depth = np.sort(src_depth)

    # station and events
    xs = np.array([178.48,-26.04,552000])
    xr = np.array([122.629,43.3034,0.])

    # 35-55 115-135 0.5°
    z0 = 100
    x = np.array([45.,125.,z0 * 1000])
    idx = np.argsort(abs(z0-src_depth))
    datadir_diff = '/mnt/d/prem_aniso_10_crust_' + str(src_depth[idx[1]]) + "/"
    dbdiff = AxiBasicDB(datadir_diff + "/PZ/Data/axisem_output.nc4")
    print(db.evdp,dbdiff.evdp)

    # compute
    duT = compute_sks_kl(db,dbdiff,db0,xs,xr,x)
    t = np.arange(db.nt) * db.dtsamp - db.shift

    plt.plot(t,duT)
    plt.show()


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
