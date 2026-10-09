import numpy as np

def compute_jacobian(skel:np.ndarray,xi_loc):
    from .libgll import lagrange_poly
    control_loc = np.array([-1,1.])

    poly_termx,poly_derix = lagrange_poly(xi_loc,control_loc)

    # loop every point
    xloc_sum = np.sum(skel * poly_termx)
    dx_dxi = np.sum(skel * poly_derix)
    jaco = dx_dxi
    xix = 1. / dx_dxi

    return xloc_sum,jaco,xix

def _get_mesh(loc_discon,nspec_discon):
    # mesh
    nspec = np.sum(nspec_discon)
    skel = np.zeros((nspec + 1))
    n = 0
    for id in range(len(nspec_discon)):
        xmin = loc_discon[id]
        xmax = loc_discon[id+1]
        dx = (xmax - xmin) / nspec_discon[id]
        for i in range(nspec_discon[id]):
            skel[n] = xmin + i * dx
            n = n + 1
    skel[n] = loc_discon[-1]

    return skel

def _get_gll_model(skel,NGLL):
    from .libgll import get_gll

    # gll
    xgll,_,_ = get_gll(NGLL)

    # set jacobian
    nspec = len(skel) - 1
    jaco = np.zeros((nspec,NGLL))
    xix = jaco * 1.
    xx = np.zeros((nspec,NGLL))
    for ispec in range(nspec):
        for igll in range(NGLL):
            xx[ispec,igll],jaco[ispec,igll],xix[ispec,igll] = compute_jacobian(skel[ispec:ispec+2],xgll[igll])

    # ibool
    ibool = np.zeros((nspec,NGLL),dtype=int)
    num = 0
    for i in range(nspec):
        for igll in range(NGLL):
            ibool[i,igll] = num
            if igll < NGLL-1:
                num += 1

    return xx,jaco,xix,ibool

def set_sem_model(NGLL,model_list,model1d,element_size_km=1.0):
    # check input
    assert(model1d in ['ak135','prem'])

    # options
    if model1d == 'ak135':
        from .model import ak135,ak135_core_string

        # discon_loc and nspecs for each layer
        # discon_loc is up to CMB
        model_discon = np.array([0,20,35,210,410.,660.,2740.,2889]) # dicon location
        elem_size_in_km = element_size_km
        nspec_discon = np.int32(np.diff(model_discon) / elem_size_in_km) + 1
        #nspec_discon = np.array([20,20,60,80,60,100,30])     # no of elements in each domain
        param_func =  ak135

        # core string
        core_string = ak135_core_string()

    elif model1d == "prem":
        from .model import prem,prem_core_string
        # discon_loc and nspecs for each layer
        # discon_loc is up to CMB
        model_discon = np.array([ 0., 15., 24.4, 80., 220., 400., 600., 670.,771.,2741.,2891]) # dicon location
        elem_size_in_km = element_size_km
        nspec_discon = np.int32(np.diff(model_discon) / elem_size_in_km) + 1
        #nspec_discon = np.array([20,20,60,80,60,100,30])     # no of elements in each domain
        param_func =  prem

        # core string
        core_string = prem_core_string()
        pass

    # mesh skeleton
    skel = _get_mesh(model_discon,nspec_discon)
    nspec = len(skel) - 1

    # gll points
    xx,jaco,xix,ibool = _get_gll_model(skel,NGLL)

    # param model
    nm = len(model_list)
    model = np.zeros((nspec,NGLL,nm))
    n = 0
    for id in range(len(nspec_discon)):
        for i in range(nspec_discon[id]):
            for igll in range(NGLL):
                r = 1000 * (6371 - xx[n,igll])
                for im in range(nm):
                    model[n,igll,im] = param_func(r,model_list[im],id+1)
            n = n + 1

    # return
    return model,xx,jaco,xix,ibool,core_string
