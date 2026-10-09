import numpy as np

def ak135(r0, param, idom):
    """
    ak135 model function

    Parameters:
    =================================
    r0: float
        current radius, in m
    param: str
        vp/vs/rho/vpv/vph/vph/vsh/eta/Qmu/Qkappa
    idom: int
        domain id

    Returns:
    =================================
    out: float
        param at current radius, in m/s
    """

    r = r0 / 1000.
    x_ak = r / 6371. #! normalized


    if idom==1 :
        ro_ak = 2.72
        vp_ak = 5.8
        vs_ak = 3.46
        Qmu = 600.0
        Qkappa = 57827.0
    elif idom==2:
        ro_ak = 2.92
        vp_ak = 6.5
        vs_ak = 3.85
        Qmu = 600.0
        Qkappa = 57827.0
    elif idom==3:
        #! moho -> 210
        ro_ak =  7.1576 - 3.859  * x_ak
        vp_ak = 17.4734 - 9.5332 * x_ak
        vs_ak =  5.8556 - 1.3825 * x_ak
        Qmu = 600.0
        Qkappa = 57827.0
    elif idom==4:
        #! 210 -> 410
        ro_ak =  7.1594 -  3.8608 * x_ak
        vp_ak = 30.7877 - 23.2542 * x_ak
        vs_ak = 15.2181 - 11.0601 * x_ak
        Qmu = 143.0
        Qkappa = 57827.0
    elif idom==5:
        #! 410 -> 660
        ro_ak = 11.1204 -  7.8713 * x_ak
        vp_ak = 29.389  - 21.4066 * x_ak
        vs_ak = 17.7173 - 13.5065 * x_ak
        Qmu = 143.0
        Qkappa = 57827.0
    elif idom==6:
        #! 660 -> D''
        ro_ak =  6.8294 - 1.7227  * x_ak -  1.1064 * x_ak**2 -  0.034409 * x_ak**3
        vp_ak = 26.8598 - 48.9644 * x_ak + 63.7326 * x_ak**2 - 32.4155   * x_ak**3
        vs_ak = 18.0019 - 43.6346 * x_ak + 60.4205 * x_ak**2 - 29.689    * x_ak**3
        Qmu = 312.0
        Qkappa = 57827.0
    elif idom==7:
        #! D'' -> CMB
        ro_ak = -65.8145 + 386.221  * x_ak - 691.6551 *x_ak**2 + 409.6742 * x_ak**3
        vp_ak =   3.4872 + 55.1872  * x_ak -  99.0089 *x_ak**2 +  58.7141 * x_ak**3
        vs_ak = -22.9553 + 164.0287 * x_ak - 294.2766 *x_ak**2 + 174.5113 * x_ak**3
        Qmu = 312.0
        Qkappa = 57827.0
    elif idom==8:
        #! CMB -> ICB
        ro_ak = 12.592  - 1.778  * x_ak - 1.6964 * x_ak**2 -  7.3524 * x_ak**3
        vp_ak = 10.7738 - 2.4831 * x_ak + 3.2584 * x_ak**2 - 14.9171 * x_ak**3
        vs_ak = 0
        Qmu = 0.0
        Qkappa = 57827.0
    elif idom==9:
        #! inner core
        ro_ak = 13.0122 - 0.0011863 *x_ak - 8.4449 * x_ak**2
        vp_ak = 11.2641 - 0.090247  *x_ak - 5.7431 * x_ak**2
        vs_ak =  3.6677 + 0.0049932 *x_ak - 4.4808 * x_ak**2
        Qmu = 84.6
        Qkappa = 1327.7

    if param=='rho':
        out = ro_ak * 1000.
    elif param=='vp':
        out = vp_ak * 1000.
    elif param=='vs':
        out = vs_ak * 1000.
    elif param=='vpv':
        out = vp_ak * 1000.
    elif param=='vsv':
        out = vs_ak * 1000.
    elif param=='vph':
        out = vp_ak * 1000.
    elif param=='vsh':
        out = vs_ak * 1000.
    elif param=='eta':
        out = 1.
    elif param=='Qmu':
        out = Qmu
    elif param=='Qka':
        out = Qkappa
    else:
        out =None

    return out

def prem_ani(r0, param, idom):
    """
    PREM anisotropic model function

    Parameters:
    =================================
    r0: float
        current radius, in m
    param: str
        vp/vs/rho/vpv/vph/vph/vsh/eta/Qmu/Qkappa
    idom: int
        domain id

    Returns:
    =================================
    out: float
        param at current radius, in m/s
    """
    r = r0 / 1000.

    x_prem = r / 6371.     #! Radius (normalized to x(surface)=1 )
    eta_aniso = 1.

    if idom == 1:     #! upper crustal layer
        ro_prem  = 2.6
        vpv_prem = 5.8
        vsv_prem = 3.2
        vph_prem = vpv_prem
        vsh_prem = vsv_prem
        Qmu = 600.0
        Qkappa = 57827.0
    elif idom == 2:  #! lower crustal layer
        ro_prem  = 2.9
        vpv_prem = 6.8
        vsv_prem = 3.9
        vph_prem = vpv_prem
        vsh_prem = vsv_prem
        Qmu = 600.0
        Qkappa = 57827.0
    elif idom == 3:  #! upper mantle
        ro_prem   =  2.6910 + 0.6924 * x_prem
        vpv_prem  =  0.8317 + 7.2180 * x_prem
        vph_prem  =  3.5908 + 4.6172 * x_prem
        vsv_prem  =  5.8582 - 1.4678 * x_prem
        vsh_prem  = -1.0839 + 5.7176 * x_prem
        eta_aniso =  3.3687 - 2.4778 * x_prem
        Qmu = 600.0
        Qkappa = 57827.0
    elif idom == 4:  #! upper mantle
        ro_prem   =  2.6910 + 0.6924 * x_prem
        vpv_prem  =  0.8317 + 7.2180 * x_prem
        vph_prem  =  3.5908 + 4.6172 * x_prem
        vsv_prem  =  5.8582 - 1.4678 * x_prem
        vsh_prem  = -1.0839 + 5.7176 * x_prem
        eta_aniso =  3.3687 - 2.4778 * x_prem
        Qmu = 80.0
        Qkappa = 57827.0
    elif idom == 5:
        ro_prem  =  7.1089 -  3.8045 * x_prem
        vpv_prem = 20.3926 - 12.2569 * x_prem
        vsv_prem =  8.9496 -  4.4597 * x_prem
        vph_prem = vpv_prem
        vsh_prem = vsv_prem
        Qmu = 143.0
        Qkappa = 57827.0
    elif idom == 6:
        ro_prem  = 11.2494 -  8.0298 * x_prem
        vpv_prem = 39.7027 - 32.6166 * x_prem
        vsv_prem = 22.3512 - 18.5856 * x_prem
        vph_prem = vpv_prem
        vsh_prem = vsv_prem
        Qmu = 143.0
        Qkappa = 57827.0
    elif idom == 7:
        ro_prem  =  5.3197 - 1.4836 * x_prem
        vpv_prem = 19.0957 - 9.8672 * x_prem
        vsv_prem =  9.9839 - 4.9324 * x_prem
        vph_prem = vpv_prem
        vsh_prem = vsv_prem
        Qmu = 143.0
        Qkappa = 57827.0
    elif idom == 8: #!lower mantle
        ro_prem  =  7.9565 - 6.4761 * x_prem + 5.5283 * x_prem**2 - 3.0807 * x_prem**3
        vpv_prem = 29.2766 -23.6027 * x_prem + 5.5242 * x_prem**2 - 2.5514 * x_prem**3
        vsv_prem = 22.3459 -17.2473 * x_prem - 2.0834 * x_prem**2 + 0.9783 * x_prem**3
        vph_prem = vpv_prem
        vsh_prem = vsv_prem
        Qmu = 312.0
        Qkappa = 57827.0
    elif idom ==9:
        ro_prem  =  7.9565 -  6.4761 * x_prem +  5.5283 * x_prem**2 -  3.0807 * x_prem**3
        vpv_prem = 24.9520 - 40.4673 * x_prem + 51.4832 * x_prem**2 - 26.6419 * x_prem**3
        vsv_prem = 11.1671 - 13.7818 * x_prem + 17.4575 * x_prem**2 -  9.2777 * x_prem**3
        vph_prem = vpv_prem
        vsh_prem = vsv_prem
        Qmu = 312.0
        Qkappa = 57827.0
    elif idom == 10:
        ro_prem  =  7.9565 - 6.4761 * x_prem + 5.5283 * x_prem**2 - 3.0807 * x_prem**3
        vpv_prem = 15.3891 - 5.3181 * x_prem + 5.5242 * x_prem**2 - 2.5514 * x_prem**3
        vsv_prem =  6.9254 + 1.4672 * x_prem - 2.0834 * x_prem**2 + 0.9783 * x_prem**3
        vph_prem = vpv_prem
        vsh_prem = vsv_prem
        Qmu = 312.0
        Qkappa = 57827.0
    elif idom == 11: #! outer core
        ro_prem  = 12.5815 - 1.2638 * x_prem - 3.6426 * x_prem**2 -  5.5281 * x_prem**3
        vpv_prem = 11.0487 - 4.0362 * x_prem + 4.8023 * x_prem**2 - 13.5732 * x_prem**3
        vsv_prem =  0.0
        vph_prem = vpv_prem
        vsh_prem = vsv_prem
        Qmu = 0.0
        Qkappa = 57827.0
    elif idom == 12: #! inner core
        ro_prem  = 13.0885 - 8.8381 * x_prem**2
        vpv_prem = 11.2622 - 6.3640 * x_prem**2
        vsv_prem =  3.6678 - 4.4475 * x_prem**2
        vph_prem = vpv_prem
        vsh_prem = vsv_prem
        Qmu = 84.6
        Qkappa = 1327.7

    if param=='rho':
        out = ro_prem * 1000.
    elif param=='vpv':
        out = vpv_prem * 1000.
    elif param=='vsv':
        out = vsv_prem * 1000.
    elif param=='vph':
        out = vph_prem * 1000.
    elif param=='vsh':
        out = vsh_prem * 1000.
    elif param=='eta':
        out = 1.
    elif param=='Qmu':
        out = Qmu
    elif param=='Qka':
        out = Qkappa
    else:
        out =None

    return out

def prem(r0, param, idom):
    r = r0 / 1000.0
    x_prem = r / 6371.0  # normalized radius

    ro_prem = vp_prem = vs_prem = Qmu = Qkappa = 0.0

    if idom == 1:
        ro_prem, vp_prem, vs_prem = 2.6, 5.8, 3.2
        Qmu, Qkappa = 600.0, 57827.0
    elif idom == 2:
        ro_prem, vp_prem, vs_prem = 2.9, 6.8, 3.9
        Qmu, Qkappa = 600.0, 57827.0
    elif idom == 3 or idom == 4:
        ro_prem = 2.691 + 0.6924 * x_prem
        vp_prem = 4.1875 + 3.9382 * x_prem
        vs_prem = 2.1519 + 2.3481 * x_prem
        Qmu = 600.0 if idom == 3 else 80.0
        Qkappa = 57827.0
    elif idom == 5:
        ro_prem = 7.1089 - 3.8045 * x_prem
        vp_prem = 20.3926 - 12.2569 * x_prem
        vs_prem = 8.9496 - 4.4597 * x_prem
        Qmu, Qkappa = 143.0, 57827.0
    elif idom == 6:
        ro_prem = 11.2494 - 8.0298 * x_prem
        vp_prem = 39.7027 - 32.6166 * x_prem
        vs_prem = 22.3512 - 18.5856 * x_prem
        Qmu, Qkappa = 143.0, 57827.0
    elif idom == 7:
        ro_prem = 5.3197 - 1.4836 * x_prem
        vp_prem = 19.0957 - 9.8672 * x_prem
        vs_prem = 9.9839 - 4.9324 * x_prem
        Qmu, Qkappa = 143.0, 57827.0
    elif idom == 8:
        ro_prem = 7.9565 - 6.4761 * x_prem + 5.5283 * x_prem**2 - 3.0807 * x_prem**3
        vp_prem = 29.2766 - 23.6027 * x_prem + 5.5242 * x_prem**2 - 2.5514 * x_prem**3
        vs_prem = 22.3459 - 17.2473 * x_prem - 2.0834 * x_prem**2 + 0.9783 * x_prem**3
        Qmu, Qkappa = 312.0, 57827.0
    elif idom == 9:
        ro_prem = 7.9565 - 6.4761 * x_prem + 5.5283 * x_prem**2 - 3.0807 * x_prem**3
        vp_prem = 24.9520 - 40.4673 * x_prem + 51.4832 * x_prem**2 - 26.6419 * x_prem**3
        vs_prem = 11.1671 - 13.7818 * x_prem + 17.4575 * x_prem**2 - 9.2777 * x_prem**3
        Qmu, Qkappa = 312.0, 57827.0
    elif idom == 10:
        ro_prem = 7.9565 - 6.4761 * x_prem + 5.5283 * x_prem**2 - 3.0807 * x_prem**3
        vp_prem = 15.3891 - 5.3181 * x_prem + 5.5242 * x_prem**2 - 2.5514 * x_prem**3
        vs_prem = 6.9254 + 1.4672 * x_prem - 2.0834 * x_prem**2 + 0.9783 * x_prem**3
        Qmu, Qkappa = 312.0, 57827.0
    elif idom == 11:
        ro_prem = 12.5815 - 1.2638 * x_prem - 3.6426 * x_prem**2 - 5.5281 * x_prem**3
        vp_prem = 11.0487 - 4.0362 * x_prem + 4.8023 * x_prem**2 - 13.5732 * x_prem**3
        vs_prem = 0.0
        Qmu, Qkappa = 0.0, 57827.0
    elif idom == 12:
        ro_prem = 13.0885 - 8.8381 * x_prem**2
        vp_prem = 11.2622 - 6.3640 * x_prem**2
        vs_prem = 3.6678 - 4.4475 * x_prem**2
        Qmu, Qkappa = 84.6, 1327.7

    param = param.lower()
    if param == 'rho':
        return ro_prem * 1000.0
    elif param in ('vp', 'vpv', 'vph'):
        return vp_prem * 1000.0
    elif param in ('vs', 'vsv', 'vsh'):
        return vs_prem * 1000.0
    elif param == 'eta':
        return 1.0
    elif param == 'qmu':
        return Qmu
    elif param == 'qka':
        return Qkappa
    else:
        raise ValueError(f"ERROR IN PREM_SUB FUNCTION: '{param}' NOT AN OPTION")


def ak135_core_string():
    s = "#          Discontinuity   1, depth:    2889.00 km\n"
    n = 150

    # CMB -> ICB
    z = np.linspace(3482,1217,n)
    for i in range(len(z)):
        z0 = z[i]
        r = 1000 * z0
        vp0 = ak135(r,'vp',8)
        vs0= ak135(r,'vs',8)
        rho0 = ak135(r,'rho',8)
        s += "\t %f %f %f %f %f %f %f %f %f\n"%(r,rho0,vp0,vs0,
                                                9999.,9999.,vp0,vs0,1.)
    # ICB -> CORE
    s += "#          Discontinuity   2, depth:    5154.00 km\n"
    z = np.linspace(1217,0,n)
    for i in range(len(z)):
        z0 = z[i]
        r = 1000 * z0
        vp0 = ak135(r,'vp',9)
        vs0= ak135(r,'vs',9)
        rho0 = ak135(r,'rho',9)
        s += "\t %f %f %f %f %f %f %f %f %f\n"%(r,rho0,vp0,vs0,
                                                9999.,9999.,vp0,vs0,1.)

    return s

def prem_aniso_core_string():
    s = "#          Discontinuity   1, depth:    2891.00 km\n"
    n = 150

    # CMB -> ICB
    z = np.linspace(3480,1221.5,n)
    for i in range(len(z)):
        z0 = z[i]
        r = 1000 * z0
        vpv0 = prem_ani(r,'vpv',11)
        vph0 = prem_ani(r,'vph',11)
        vsv0 = prem_ani(r,'vsv',11)
        vsh0 = prem_ani(r,'vsh',11)
        rho0 = prem_ani(r,'rho',11)

        s += "\t %f %f %f %f %f %f %f %f %f\n"%(r,rho0,vpv0,vsv0,
                                                9999.,9999.,vph0,vsh0,1.)
    # ICB -> CORE
    s += "#          Discontinuity   2, depth:    5149.50 km\n"
    z = np.linspace(1221.5,0,n)
    for i in range(len(z)):
        z0 = z[i]
        r = 1000 * z0
        vpv0 = prem_ani(r,'vpv',11)
        vph0 = prem_ani(r,'vph',11)
        vsv0 = prem_ani(r,'vsv',11)
        vsh0 = prem_ani(r,'vsh',11)
        rho0 = prem_ani(r,'rho',11)
        s += "\t %f %f %f %f %f %f %f %f %f\n"%(r,rho0,vpv0,vsv0,
                                                9999.,9999.,vph0,vsh0,1.)

    return s

def prem_core_string():
    s = "#          Discontinuity   1, depth:    2891.00 km\n"
    n = 150

    # CMB -> ICB
    z = np.linspace(3480,1221.5,n)
    for i in range(len(z)):
        z0 = z[i]
        r = 1000 * z0
        vpv0 = prem(r,'vpv',11)
        vph0 = prem(r,'vph',11)
        vsv0 = prem(r,'vsv',11)
        vsh0 = prem(r,'vsh',11)
        rho0 = prem(r,'rho',11)

        s += "\t %f %f %f %f %f %f %f %f %f\n"%(r,rho0,vpv0,vsv0,
                                                9999.,9999.,vph0,vsh0,1.)
    # ICB -> CORE
    s += "#          Discontinuity   2, depth:    5149.50 km\n"
    z = np.linspace(1221.5,0,n)
    for i in range(len(z)):
        z0 = z[i]
        r = 1000 * z0
        vpv0 = prem(r,'vpv',11)
        vph0 = prem(r,'vph',11)
        vsv0 = prem(r,'vsv',11)
        vsh0 = prem(r,'vsh',11)
        rho0 = prem(r,'rho',11)
        s += "\t %f %f %f %f %f %f %f %f %f\n"%(r,rho0,vpv0,vsv0,
                                                9999.,9999.,vph0,vsh0,1.)

    return s
