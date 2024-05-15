from database import AxiBasicDB,rotation_matrix
import numpy as np 
import matplotlib.pyplot as plt 

def rotate_mij_src_to_recv(mt,R):
    """
    rotate Mij tensor from (xs,ys,zs) coordinate to (s,p,phi) in receiver centered system

    Input: 
        mt: np.array, shape(6), 
            momtent tensor in voigt notation, mxx,myy,mzz,myz,mxz,mzy
        R: np.ndarray, shape(3,3)
            rotation matrix  (Rr R0)^T Rs  
    
    Returns:
        mtnew: np.array, shape(6)
            rotated mt tensor
    """
    A = np.array(
        [
            [mt[0], mt[5], mt[4]],  # NOQA
            [mt[5], mt[1], mt[3]],
            [mt[4], mt[3], mt[2]],
        ]
    )

    R = np.require(R, dtype=np.float128)  # NOQA
    A = np.require(A, dtype=np.float128)  # NOQA

    B = (R @ A) @ R.T 

    mtnew = np.require(
        np.array([B[0, 0], B[1, 1], B[2, 2], B[1, 2], B[0, 2], B[0, 1]]),
        dtype=np.float64,
    )    

    return mtnew 

def reciprocity(db:AxiBasicDB,cmtfile):
    # read moment tensor
    mzz,mxx,myy,mxz,myz,mxy = db.read_cmt(cmtfile)
    mij = np.array([mxx,myy,mzz,myz,mxz,mxy])

    # get source coordinates
    evla = 37.91; evlo = -77.93; evdp = 12000.
    #evla = 42.65961867; evlo = 74.48293762; 
    Rs = rotation_matrix(np.deg2rad(90-evla),np.deg2rad(evlo))

    # comptue rotated theta and phi
    theta,phi = db.compute_tp_recv(evla,evlo)

    # get strain  # note that z/x are receiver centered coordinates
    # locate point 
    r = 6371000 - evdp
    sr = r * np.sin(theta)
    zr = r * np.cos(theta)

    # locate element
    elemid,xi,eta = db._locate_elem(sr,zr)
    strain_z = db._get_strain(elemid,xi,eta,"../PZ/Data/axisem_fields.h5")
    strain_1 = db._get_strain(elemid,xi,eta,"../PX_PY/Data/axisem_fields.h5")

    # cos sin factors
    fac_1_map = {"X": np.cos, "Y": np.sin}
    fac_2_map = {"X": lambda x: -np.sin(x), "Y": np.cos}

    # rotate mij to receiver centered coordinates (s,p,z)
    R1 = np.eye(3) # rotate from (s,phi,z) to (xr,yr,zr)
    R1[0,:2] = [np.cos(phi),-np.sin(phi)]
    R1[1,:2] = [np.sin(phi),np.cos(phi)]
    R2 = (db.rot_mat @ R1).T @ Rs
    mij = rotate_mij_src_to_recv(mij,R2)

    # compute synthetic seismograms
    nt = strain_1.shape[-1]
    u = np.zeros((3,nt))
    
    # z 
    for i in range(3):
        u[2,:] += strain_z[i,:] * mij[i]
    u[2,:] += 2.0 * mij[4] *strain_z[4,:]

    # x,y
    for i,comp in enumerate(['X','Y']):
        fac1 = fac_1_map[comp](phi)
        fac2 = fac_2_map[comp](phi)
        for j in range(6):
            p = mij[j]
            if j < 3:
                p = p * fac1
            elif j == 3 or j == 5:
                p = p * fac2 *2.
            else:
                p = p * fac1 *2. 

            u[i,:] += strain_1[j,:] * p
    
    # note that vertforce and Thetaforce are in south/east direction
    uz = u[2,:] *1. 
    un = -u[0,:]
    ue = u[1,:]

    # compare 
    net = 'AAK'
    sta = 'II'
    name = "../prem2//Data_Postprocessing/SEISMOGRAMS/" + net + "_" +   \
            sta + "_disp_post_mij_conv0000_"
    data_z = np.loadtxt(name + "Z.dat")
    data_e = np.loadtxt(name + "E.dat")
    data_n = np.loadtxt(name + "N.dat")
    t = np.arange(db.nt) * db.dtsamp + db.t0

    plt.figure(1,figsize=(14,16))
    plt.subplot(3,1,1)
    plt.plot(t,uz[:])
    plt.plot(data_z[:,0],data_z[:,1],ls='--')
    plt.title("Z")

    plt.subplot(3,1,2)
    plt.plot(t,ue[:])
    plt.plot(data_e[:,0],data_e[:,1],ls='--')
    plt.title("E")

    plt.subplot(3,1,3)
    plt.plot(t,un[:])
    plt.plot(data_z[:,0],data_n[:,1],ls='--')
    plt.title("N")

    outname = net + "_" + sta + ".jpg"
    plt.savefig(outname)


# read db of single force 
db = AxiBasicDB("../PZ/Data/axisem_output.nc4")

reciprocity(db,"../CMTSOLUTION")

