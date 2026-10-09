import numpy as np
from tqdm import tqdm
from .sem_mesh import set_sem_model

# import gpu device if required
try:
    import cupy as cp
except:
    import numpy as cp

def smooth_pde(NGLL,sigma,model_list = ('vp','vs','rho'),model1d = 'ak135',element_size_km=1.0):
    # get gll arrays
    from .libgll import get_gll
    _,wxgll,hprime = get_gll(NGLL)

    # get model
    model,xx,jaco,xix,ibool,core_string = set_sem_model(NGLL,model_list,model1d,element_size_km)
    nglob = np.max(ibool) + 1
    nspec = xx.shape[0]
    nm = model.shape[-1]

    # prepare C tensor
    dmin = np.min(np.diff(xx, axis=1))
    dt = 1.0 / 12 * dmin * dmin
    nt = int(sigma**2 / ( 2 * dt)) + 1
    c = dt
    print(f"SEM smoothing, nspec = {nspec} NGLL = {NGLL} nstep = {nt}")

    # get unique coordinates
    xstore = np.zeros((nglob,1))
    for ispec in range(nspec):
        for igll in range(NGLL):
            iglob = ibool[ispec,igll]
            xstore[iglob] = xx[ispec,igll]

    # alloc mem on gpu
    d_jaco = cp.asarray(jaco)
    d_xix = cp.asarray(xix)
    d_ibool = cp.asarray(ibool)
    d_wgll = cp.asarray(wxgll)
    d_hprime = cp.asarray(hprime)
    d_hprimeT = cp.transpose(cp.asarray(d_hprime))

    # inverse mass matrix
    rmassinv = d_wgll * d_jaco
    temp = cp.bincount(d_ibool.flatten(),rmassinv.flatten())
    rmassinv = 1. / temp[d_ibool]

    # get average model
    counter_1d = cp.bincount(d_ibool.flatten(),cp.ones(ibool.size))
    counter = counter_1d[d_ibool]
    d_displ = cp.zeros((nm,nspec,NGLL))
    for im in range(nm):
        temp = cp.asanyarray(model[:,:,im]).flatten()
        temp1d = cp.bincount(d_ibool.flatten(),temp.flatten())
        temp = temp1d[d_ibool]
        d_displ[im,:,:] = temp / counter

    # smooth begin
    print('smoothing begin ...')
    # smoothing
    for it in tqdm(range(nt)):
        accel = cp.zeros((nm,nspec,NGLL))

        for im in range(nm):
            #s = np.einsum('ik,nk',d_hprime,d_displ[im,:,:] * c) * d_xix
            s = cp.dot(c * d_displ[im,:,:],d_hprimeT) * d_xix
            f = cp.dot(s * d_jaco * d_xix * d_wgll,d_hprime)
            accel[im,...] -= f

            # bin count
            accel1d = cp.bincount(d_ibool.flatten(),accel[im,...].flatten())
            accel[im,...] = accel1d[d_ibool] * rmassinv

        d_displ += accel

    # convert function from cupy to numpy
    if cp == np:
        func = np.asarray
    else:
        func = cp.asnumpy
    # back to gpu
    displ = np.zeros((nglob,nm))
    for im in range(nm):
        temp = func(d_displ[im,:,:])
        for ispec in range(nspec):
            for igll in range(NGLL):
                iglob = ibool[ispec,igll]
                displ[iglob,im] = temp[ispec,igll]

    return model,xx,xstore,displ,core_string

def smooth_pde_slow(NGLL,sigma,model_list = ('vp','vs','rho'),model1d = 'ak135',element_size_km=1.0):
    # get gll arrays
    from .libgll import get_gll
    xgll,wxgll,hprime = get_gll(NGLL)

    # get model
    model,xx,jaco,xix,ibool,core_string = set_sem_model(NGLL,model_list,model1d,element_size_km)
    nglob = np.max(ibool) + 1
    nspec = xx.shape[0]
    nm = model.shape[-1]

    # rmassinv
    rmassinv = np.zeros((nglob,1))
    for ispec in range(nspec):
        mass = wxgll * jaco[ispec,:]
        for igll in range(NGLL):
            iglob = ibool[ispec,igll]
            rmassinv[iglob] += mass[igll]
    rmassinv = 1. / rmassinv

    # prepare C tensor
    dmin = np.min(np.diff(xx, axis=1))
    dt = 1.0 / 12 * dmin * dmin
    nt = int(sigma**2 / ( 2 * dt)) + 1
    c = dt

    # get average model
    counter = np.zeros((nglob,1),dtype=float)
    displ = np.zeros((nglob,nm))
    xstore = counter * 1.
    for ispec in range(nspec):
        for igll in range(NGLL):
            iglob = ibool[ispec,igll]
            for im in range(nm):
                displ[iglob,im] += model[ispec,igll,im]
            counter[iglob] += 1.
            xstore[iglob] = xx[ispec,igll]
    displ /= counter

    # smooth begin
    print('smoothing begin ...')
    # smoothing
    for it in tqdm(range(nt)):
        accel = np.zeros((nglob,nm))

        for ispec in range(nspec):
            idx = ibool[ispec,:]
            for im in range(nm):
                u = displ[idx,im]
                s = c * np.dot(hprime,u) * xix[ispec,:]
                f = np.dot(s*jaco[ispec,:] * xix[ispec,:] * wxgll,hprime)
                accel[idx,im] -= f

        # apply mass matrix
        accel *= rmassinv
        displ += accel

    return model,xx,xstore,displ,core_string
