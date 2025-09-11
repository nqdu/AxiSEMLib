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

def _taper_hann_edges(f,taper_fac=0.05):
    """
    taper the signal with a Hann window
    taper_fac: fraction of the signal length to be tapered at each end
    """

    n = len(f)
    m = int(np.floor(taper_fac * n))  # number of points tapered at each end
    window = np.ones(n)
    if m > 0:
        hann_win = np.hanning(2*m)
        window[:m] = hann_win[:m]
        window[-m:] = hann_win[m:]
    f_tapered = f * window 

    return f_tapered


def prefilt_interp(
    t, u, t_new,
    method='savgol',     # 'savgol' or 'linear'
    fmax=0.25,           # Hz: desired max passband (e.g., solver's reliable max freq). None = auto.
    sg_alpha=0.6,        # Savitzky–Golay window ~ fraction of a cycle at fmax
    sg_poly=5,           # Savitzky–Golay poly order (>= 2)
    deriv=0              # 0, 1, or 2: derivative order to return
):
    """
    Interpolate u(t) onto t_new and optionally compute its derivative(s) at the new rate.
    Returns (f, df) where df is None if deriv==0; if deriv==2, df is the 2nd derivative.
    Pipeline:
      1) Optional zero-phase IIR low-pass on source grid (to prevent aliasing / limit band).
      2) Linear interpolation onto t_new.
      3) If method='savgol': Savitzky-Golay smoothing/derivative on the new grid.

    Notes:
      - If downsampling, set fmax <= 0.5*fs_new (with margin).
      - Outside the original domain, we hold edge values (safer than zeros).
    """
    from scipy.signal import detrend, savgol_filter, butter, sosfiltfilt
    from scipy.interpolate import interp1d
    if deriv not in (0, 1, 2):
        raise ValueError("deriv must be 0, 1, or 2")

    t = np.asarray(t)
    u = np.asarray(u)
    t_new = np.asarray(t_new)

    # --- basic checks ---
    if t.ndim != 1 or u.ndim != 1 or t.size != u.size:
        raise ValueError("t and u must be 1D arrays of equal length")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    # uniformity check (tolerant)
    dt_src = np.median(np.diff(t))
    if not np.allclose(np.diff(t), dt_src, rtol=1e-6, atol=1e-9):
        raise ValueError("IIR filtering requires uniform t")

    fs_src = 1.0 / dt_src
    dt_new = np.median(np.diff(t_new))
    fs_new = 1.0 / dt_new

    # --- detrend & gentle taper to reduce filtfilt edge transients ---
    x = u - np.mean(u)
    x = detrend(x, type='linear')
    x = _taper_hann_edges(x, 0.05)

    # --- 1) zero-phase LPF if requested/needed ---
    # Auto-pick fmax if not provided: keep content safe for new Nyquist
    fmax_eff = float(fmax)
    # clamp to valid digital cutoff
    fmax_eff = min(fmax_eff, 0.49*fs_src)
    # Only filter if it actually does something
    do_filter = fmax_eff < 0.49*fs_src
    if do_filter:
        sos = butter(4, fmax_eff, btype='low', fs=fs_src, output='sos')
        x = sosfiltfilt(sos, x)
        x = _taper_hann_edges(x, 0.03)

    # --- 2) interpolate to new grid (edge hold to avoid zeros/sharp edges) ---
    f_interp = interp1d(t, x, kind='linear',
                        bounds_error=False,
                        fill_value=(x[0], x[-1]))
    f_new = f_interp(t_new)

    if method == 'linear':
        if deriv == 0:
            return f_new, None
        elif deriv == 1:
            df = np.gradient(f_new, dt_new, edge_order=2)
            return f_new, df
        else:  # deriv == 2
            d1 = np.gradient(f_new, dt_new, edge_order=2)
            d2 = np.gradient(d1, dt_new, edge_order=2)
            return f_new, d2

    # --- 3) Savitzky–Golay on the new grid ---
    # Choose window ~ sg_alpha * one cycle at fmax_eff
    # cyclesamples = (1/fc) / dt_new; window ≈ sg_alpha * cyclesamples
    fc = max(1e-12, fmax_eff)          # avoid divide-by-zero
    win = int(round(sg_alpha * (1.0 / (fc * dt_new))))
    win = max(win, sg_poly + 2)        # must be > polyorder
    if win % 2 == 0:
        win += 1
    # cap by data length
    max_win = len(f_new) - (1 - len(f_new) % 2)   # largest odd <= N
    win = min(win, max_win)
    # fallbacks for tiny signals
    win = max(win, sg_poly + 1 + ((sg_poly + 1) % 2), 5)

    f0 = savgol_filter(f_new, window_length=win, polyorder=sg_poly,
                       deriv=0, delta=dt_new, mode='interp')

    if deriv == 0:
        return f0, None
    else:
        df = savgol_filter(f_new, window_length=win, polyorder=sg_poly,
                           deriv=deriv, delta=dt_new, mode='interp')
        return f0, df

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
