# import matplotlib.pyplot as plt 
import numpy as np
from numba import jit 

def rotation_matrix(colat,lon):
    """
    get rotation matrix, Tarje 2007 (2.14), from (xyz) to (tpr): v^{sph} = R.T @ v^{cart}

    Parameters: 
    ========================
    colat: float 
        co-lattitude, in rad
    lon : float
        longitude, in rad

    Returns:
    =======================
    R: np.ndarray
        3x3 rotation matrix
    """
    cost = np.cos(colat); sint = np.sin(colat)
    cosp = np.cos(lon); sinp = np.sin(lon)

    R = np.zeros((3,3))
    R[0,:] = [cost * cosp, -sinp, sint * cosp]
    R[1,:] = [cost * sinp, cosp, sint * sinp]
    R[2,:] = [-sint, 0, cost]

    return R

@jit(nopython=True)
def rotate_tensor2(eps,R):
    """
    eps1_{pq} = R_{pi} eps_{ij} R_{qj}

    eps: np.ndarray
        symmetric 3x3 tensor, in voigt form
    R: np.ndarray
        3x3 rotation matrix
    
    Returns:
    ===================
    eps1: np.ndarray
        eps1_{pq} = R_{pi} eps_{ij} R_{qj}
    """
    eps_xyz = np.zeros(eps.shape,dtype=np.float64)

    # voigt notation 
    vid = np.array([[0,5,4],[5,1,3],[4,3,2]])

    for p in range(3):
        for q in range(3):
            id = vid[p,q]
            for m in range(3):
                for n in range(3):
                    id1 = vid[m,n]
                    eps_xyz[id,...] += R[p,m] * eps[id1,...] * R[q,n]
    
    # divide by 2 for last 3 index
    eps_xyz[3:,...] *= 0.5

    return eps_xyz

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

def prefilt_interp(t, u, t_new,
                method='savgol',  # 'savgol' or 'linear'
                fmax=0.25,        # Hz: max reliable freq of target solver
                sg_alpha=0.6,     # Savitzky–Golay window ~ fraction of a cycle
                sg_poly=5,
                deriv=0):       # Savitzky–Golay poly order (>=3)
    """
    Interpolate f(t) -> f, f', f'' on t_new.
    Outside original domain: zeros.
    Steps:
      1. IIR low-pass filter (zero-phase) (for sg filter, or directly return linear interp)
      2. Interpolate to uniform t_new.
      3. Savitzky-Golay differentiation for f', f''.
    """

    from scipy.signal import sosfiltfilt,butter,detrend, savgol_filter
    from scipy.interpolate import interp1d

    # santity check
    if deriv not in [0, 1, 2]:
        print("Error: deriv should be 0, 1, or 2")
        return None, None

    if method != 'savgol':
        f = interp1d(t, u, kind='linear',
                     bounds_error=False, fill_value=0.0)
        f0 = f(t_new)

        if deriv == 2:
            dt1 = t_new[1] - t_new[0]
            f2 = diff1(diff1(f0, dt1), dt1)
        elif deriv == 1:
            dt1 = t_new[1] - t_new[0]
            f2 = diff1(f0, dt1)
        else:
            f2 = None

        return f0, f2

    t = np.asarray(t); f = np.asarray(u)
    dt_src = t[1] - t[0]
    fs_src = 1.0 / dt_src

    # detrend/demean
    f = f - np.mean(f)
    f = detrend(f, type='linear')

    # taper
    taper_frac = 0.05
    m = int(np.floor(taper_frac * len(f)))  # number of points tapered at each end
    window = np.ones(len(f))
    if m > 0:
        hann = np.hanning(2*m)
        window[:m] = hann[:m]
        window[-m:] = hann[m:]
    f = f * window 

    # --- 1. IIR low-pass on source grid ---
    sos = butter(4, fmax, btype='low', fs=fs_src, output='sos')
    f_lp = sosfiltfilt(sos, f)

    # taper again
    taper_frac = 0.03
    window = window * 0 + 1. 
    m = int(np.floor(taper_frac * len(f_lp)))  # number of points tapered at each end
    if m > 0:
        hann = np.hanning(2*m)
        window[:m] = hann[:m]
        window[-m:] = hann[m:]
    f_lp = f_lp * window

    # --- 2. Interpolate to new grid ---
    interp_fun = interp1d(t, f_lp, kind='linear',
                          bounds_error=False, fill_value=0.0)
    f_res = interp_fun(t_new)

    # --- 3. Savitzky–Golay differentiation on final grid ---
    dx = t_new[1] - t_new[0]
    fc = fmax
    win = int(np.round(sg_alpha * (1.0 / (fc * dx))))  # ~0.5–1.0 cycles of fc
    if win % 2 == 0: win += 1
    win = max(win, sg_poly + 2 + (sg_poly % 2 == 0))   # valid odd length

    f0 = savgol_filter(f_res, window_length=win, polyorder=sg_poly,
                       deriv=0, delta=dx, mode='interp')

    if deriv == 0:
        f2 = None
    else:
        f2 = savgol_filter(f_res, winow_length=win, polyorder=sg_poly,
                           deriv=deriv, delta=dx, mode='interp')

    return f0, f2

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

def read_cmtsolution(cmtfile):
    M = {}
    with open(cmtfile) as f:
        for line in f:
            info = line.strip().split(":")
            if len(info) == 2:
                key,value = info 

                if 'M' in key:
                    M[key] = float(value) * 1.0e-7
    
    # get moment tensor
    mzz = M['Mrr']; mxx = M['Mtt']
    myy = M['Mpp']; mxz = M['Mrt']
    myz = M['Mrp']; mxy = M['Mtp']

    return mzz,mxx,myy,mxz,myz,mxy

def c_ijkl_ani(lambda_, mu, xi_ani, phi_ani, eta_ani, theta_fa, phi_fa, i1, j1, k1, l1):
    """
    Returns the stiffness tensor as defined in Nolet (2008), Eq. (16.2).
    Indices i, j, k, and l should be in [0,2] (converted from Fortran's [1,3]).
    """
    # Constants
    one = 1.0
    two = 2.0

    # index start from 0 
    i,j,k,l = map(lambda x: x-1,[i1,j1,k1,l1])
    
    # Initialize delta function (Kronecker delta)
    deltaf = np.zeros((3, 3))
    np.fill_diagonal(deltaf, one)

    # get shape
    nx,nz = lambda_.shape
    
    # Compute s vector
    s = np.zeros((3,nx,nz))
    s[0,...] = np.cos(phi_fa) * np.sin(theta_fa)
    s[1,...] = np.sin(phi_fa) * np.sin(theta_fa)
    s[2,...] = np.cos(theta_fa)
    
    # Initialize c_ijkl
    c_ijkl = np.zeros(lambda_.shape)
    
    # Isotropic part
    c_ijkl += lambda_ * deltaf[i, j] * deltaf[k, l]
    c_ijkl += mu * (deltaf[i, k] * deltaf[j, l] + deltaf[i, l] * deltaf[j, k])
    
    # Anisotropic part
    c_ijkl += ((eta_ani - one) * lambda_ + two * eta_ani * mu * (one - one / xi_ani)) * \
                  (deltaf[i, j] * s[k] * s[l] + deltaf[k, l] * s[i] * s[j])
    
    c_ijkl += mu * (one / xi_ani - one) * \
                  (deltaf[i, k] * s[j] * s[l] + deltaf[i, l] * s[j] * s[k] +
                   deltaf[j, k] * s[i] * s[l] + deltaf[j, l] * s[i] * s[k])
    
    c_ijkl += ((one - two * eta_ani + phi_ani) * (lambda_ + two * mu) +
                   (4.0 * eta_ani - 4.0) * mu / xi_ani) * \
                  (s[i] * s[j] * s[k] * s[l])
    
    return c_ijkl
