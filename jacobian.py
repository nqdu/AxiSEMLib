import numpy as np 
from numba import jit

@jit(nopython=True)
def lagrange_poly(xi,xctrl):
    nctrl = len(xctrl)
    hprime = np.array([0.0 for i in range(nctrl)])
    h = hprime * 1.0

    #! note: this routine is hit pretty hard by the mesher, optimizing the loops here will be beneficial
    for dgr in range(nctrl):
        prod1 = 1.; prod2 = 1.

        #// lagrangian interpolants
        x0 = xctrl[dgr]
        for i in range(nctrl):
            if i != dgr:
                x = xctrl[i]
                prod1 = prod1*(xi-x)
                prod2 = prod2*(x0-x)

        #//! takes inverse to avoid additional divisions
        #//! (multiplications are cheaper than divisions)
        prod2_inv = 1. / prod2
        h[dgr] = prod1 * prod2_inv

        #// first derivatives
        s = 0.0
        for i in range(nctrl):
            if i != dgr :
                prod3 = 1.0
                for j in range(nctrl):
                    if j != dgr and j != i:
                        prod3 = prod3*(xi-xctrl[j])
                s = s + prod3
        hprime[dgr] = s * prod2_inv
    

    return h,hprime

@jit(nopython=True)
def _cal_jaco_surf(cords:np.ndarray):
    # cords: (NGLL,NGLL,3) array of (x,y,z) in m 

    # gll points in [-1,1]
    xgll = np.array([-1.0, -0.6546536707079771, 0.0, 0.6546536707079771, 1.0],dtype=np.float64)

    # check which line is constant, and get x,z
    NGLL = 5
    x = np.ascontiguousarray(cords[0:NGLL:2,0:NGLL:2,0])
    y = np.ascontiguousarray(cords[0:NGLL:2,0:NGLL:2,1])
    z = np.ascontiguousarray(cords[0:NGLL:2,0:NGLL:2,2])
    
    # allocate space for jacobian
    jaco = np.zeros((NGLL,NGLL),dtype=np.float64)
    dxi_dr = np.zeros((NGLL,NGLL,2,3),dtype=np.float64)  # dxi/dr and dzi/dr

    # 3 point lagrange interpolation 
    xctrl = np.array([-1.0, 0.0, 1.0],dtype=np.float64)

    # loop over all GLL points to compute jacobian
    for iz in range(NGLL):
        for ix in range(NGLL):
            xi = xgll[ix]
            zi = xgll[iz]

            # lagrange polynomials and derivatives
            hx, hpx = lagrange_poly(xi,xctrl)
            hz, hpz = lagrange_poly(zi,xctrl)

            # compute jacobian
            dx_dxi0 = 0.0
            dy_dxi0 = 0.0
            dz_dxi0 = 0.0
            dx_dzi0 = 0.0
            dy_dzi0 = 0.0
            dz_dzi0 = 0.0

            for m in range(3):
                for n in range(3):
                    d1 = hpx[m] * hz[n]
                    d2 = hx[m] * hpz[n]
                    dx_dxi0 += x[m,n] * d1
                    dy_dxi0 += y[m,n] * d1
                    dz_dxi0 += z[m,n] * d1
                    dx_dzi0 += x[m,n] * d2
                    dy_dzi0 += y[m,n] * d2
                    dz_dzi0 += z[m,n] * d2
            
            # compute metric tensor 
            # gij = dr/dxi_i . dr/dxi_j
            g11 = dx_dxi0 * dx_dxi0 + dy_dxi0 * dy_dxi0 + dz_dxi0 * dz_dxi0
            g22 = dx_dzi0 * dx_dzi0 + dy_dzi0 * dy_dzi0 + dz_dzi0 * dz_dzi0
            g12 = dx_dxi0 * dx_dzi0 + dy_dxi0 * dy_dzi0 + dz_dxi0 * dz_dzi0

            # inverse metric tensor
            ginv = np.zeros((2,2),dtype=np.float64)
            ginv[0,0] = g22
            ginv[1,1] = g11
            ginv[0,1] = -g12
            ginv[1,0] = -g12
            ginv *= 1.0 / (g11 * g22 - g12 * g12)

            # compute contravariant basis vectors
            a = np.array([
                [dx_dxi0, dy_dxi0, dz_dxi0],
                [dx_dzi0, dy_dzi0, dz_dzi0]
            ],dtype=np.float64)
            
            # jaco 2D = |dr/dxi x dr/dzi|
            nx = dy_dxi0 * dz_dzi0 - dy_dzi0 * dz_dxi0
            ny = dz_dxi0 * dx_dzi0 - dz_dzi0 * dx_dxi0
            nz = dx_dxi0 * dy_dzi0 - dx_dzi0 * dy_dxi0
            jacobian = np.sqrt(nx*nx + ny*ny + nz*nz)

            # store
            jaco[iz,ix] = jacobian
            dxi_dr[iz,ix,:,:] = ginv @ a # (2,3) array

    return jaco, dxi_dr
        
@jit(nopython=True)
def compute_jacobian_surface(cord_faces:np.ndarray):
    """
    compute jacobian on surface elements 

    Parameters
    -------------------
    cord_faces: np.ndarray
        (nfaces, NGLL, NGLL, 3) array of (x,y,z) in m 

    Returns
    -------------------
    jaco_faces: np.ndarray
        (nfaces, NGLL, NGLL) array of jacobian values 
    dxi_dr_faces: np.ndarray
        (nfaces, NGLL, NGLL, 2,3) array of dr/dxi and dr/dzi values
    """
    nfaces = cord_faces.shape[0]
    NGLL = cord_faces.shape[1]

    jaco_faces = np.zeros((nfaces,NGLL,NGLL),dtype=np.float64)
    dxi_dr_faces = np.zeros((nfaces,NGLL,NGLL,2,3),dtype=np.float64)

    for ifa in range(nfaces):
        cords = cord_faces[ifa,:,:,:]
        jaco, dxi_dr = _cal_jaco_surf(cords)
        jaco_faces[ifa,:,:] = jaco * 1.0
        dxi_dr_faces[ifa,:,:,:,:] = dxi_dr * 1.0

    return jaco_faces, dxi_dr_faces

@jit(nopython=True)
def moment_to_force(mt:np.ndarray, jaco:np.ndarray,dxi_dr:np.ndarray):
    """
    convert moment tensor to equivalent force 

    Parameters
    -------------------
    mt: np.ndarray
        (nfaces, NGLL, NGLL, 6,nt) moment tensor in voigt notation 
    jaco: np.ndarray
        (nfaces, NGLL, NGLL) jacobian on the surface element 
    dxi_dr: np.ndarray
        (nfaces, NGLL, NGLL, 2,3) dr/dxi and dr/dzi on the surface element

    Returns
    -------------------
    f_eq: np.ndarray
        (nfaces,NGLL, NGLL, 3,nt) equivalent force components on the surface element 
    """
    NGLL = 5
    nfaces = mt.shape[0]
    nt = mt.shape[4]

    # allocate space for equivalent force
    f_eq = np.zeros((nfaces,NGLL,NGLL,3,nt),dtype=np.float64)

    # gll weights and locs for NGLL=5
    wgll = np.array([0.1, 0.5444444444444444, 0.7111111111111111, 0.5444444444444444, 0.1],dtype=np.float64)
    hprime_wgllT =  np.array([
    [-4.9999999999999994e-01, -6.7565024887242375e-01,  2.6666666666666661e-01, -1.4101641779424262e-01,  5.0000000000000003e-02],
    [ 6.7565024887242409e-01, -3.7022853918402794e-16, -9.5046014413898916e-01,  4.1582631306080775e-01, -1.4101641779424273e-01],
    [-2.6666666666666677e-01,  9.5046014413898927e-01,  1.8421478334521120e-16, -9.5046014413898927e-01,  2.6666666666677e-01],
    [ 1.4101641779424268e-01, -4.1582631306080770e-01,  9.5046014413898883e-01,  4.9363805224537042e-16, -6.7565024887242398e-01],
    [-5.0000000000000003e-02,  1.4101641779424262e-01, -2.6666666666666661e-01,  6.7565024887242375e-01,  5.0000000000000000e-01]
    ], dtype=float)

    # voigt to tensor mapping
    voigt = np.array([[0,5,4],[5,1,3],[4,3,2]],dtype=np.int64)

    # now loop over all faces and GLL points to compute equivalent force
    for ifa in range(nfaces):
        for iz in range(NGLL):
            for ix in range(NGLL):
                jac = jaco[ifa,iz,ix]

                for p in range(3):
                    for q in range(3):
                        idx = voigt[p,q]

                        # contribution to equivalent force
                        c1 = np.zeros(nt,dtype=np.float64)
                        c2 = c1 * 0.
                        for a in range(NGLL):
                            c1 += mt[ifa,a,ix,idx,:] * hprime_wgllT[iz,a] *  \
                                jaco[ifa,a,ix] * dxi_dr[ifa,a,ix,0,q]
                            c2 += mt[ifa,iz,a,idx,:] * hprime_wgllT[ix,a] *  \
                                jaco[ifa,iz,a] * dxi_dr[ifa,iz,a,1,q]
                
                    f_eq[ifa,iz,ix,p,:] = c1 / (wgll[iz] * jac) + c2 / (wgll[ix] * jac)
    
    return f_eq 