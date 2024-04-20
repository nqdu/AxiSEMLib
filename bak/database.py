import numpy as np 
import h5py 
from numba import jit 

def rotation_matrix(colat,lon):
    """
    get rotation matrix, Tarje 2007 (2.14)
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

class AxiBasicDB:

    def __init__(self) -> None:
        pass

    def read_basic(self,ncfile:str) -> None:
        """
        ncfile : str
            input netcdf file 
        """
        # load library
        from scipy.spatial import KDTree

        fio:h5py.File = h5py.File(ncfile,"r")
        self.nspec = len(fio['Mesh/elements'])
        self.nctrl = len(fio['Mesh/control_points'])
        self.ngll = len(fio['Mesh/npol'])

        # read attributes
        self.dtsamp = fio.attrs['strain dump sampling rate in sec'][0]
        self.shift = fio.attrs['source shift factor in sec'][0]
        self.nt = len(fio['snapshots'])
        self.nglob = len(fio['gllpoints_all'])
    
        # read source parameters
        self.evdp = fio.attrs['source depth in km'][0] * 1.
        evcola = fio.attrs['Source colatitude'][0]
        evlo = fio.attrs['Source longitude'][0]
        self.evla:float = 90 - np.rad2deg(evcola) 
        self.evlo:float = np.rad2deg(evlo)
        self.mag = fio.attrs['scalar source magnitude'][0] * 1.

        # rotation matrix
        self.rot_mat = rotation_matrix(evcola,evlo)

        # read mesh 
        self.s = fio['Mesh/mesh_S'][:]
        self.z = fio['Mesh/mesh_Z'][:]

        # create kdtree
        md_pts = np.zeros((self.nspec,2))
        md_pts[:,0] = fio['Mesh/mp_mesh_S'][:]
        md_pts[:,1] = fio['Mesh/mp_mesh_Z'][:]
        self.kdtree = KDTree(data=md_pts)

        # elemtype
        self.eltype = fio['Mesh/eltype'][:]
        self.axis = fio['Mesh/axis'][:]

        # skeleton
        self.skelid = fio['Mesh/fem_mesh'][:]

        # connectivity[
        self.ibool = fio['Mesh/sem_mesh'][:]

        # other useful arrays
        self.G0 = fio['Mesh/G0'][:]
        self.G1 = fio['Mesh/G1'][:].T 
        self.G2 = fio['Mesh/G2'][:].T
        self.G1T = np.require(self.G1.T,requirements=['F_CONTIGUOUS'])
        self.G2T = np.require(self.G2.T,requirements=['F_CONTIGUOUS'])

        self.gll = fio['Mesh/gll'][:]
        self.glj = fio['Mesh/glj'][:]

        # elastic moduli
        self.mu = fio['Mesh/mesh_mu'][:]
        self.lamda = fio['Mesh/mesh_lambda'][:]

        # close file
        fio.close()

        # data file dict
        self.iodict = {}

    def set_iodata(self,ncfile_dir:str):
        """
        set absolute path to top simulation dir
        Example:
            set_iodata('/path/to/axisem/solver/simudir')
        )
        """
        import os 
        for stype in ['MZZ',"MXX_P_MYY","MXZ_MYZ","MXY_MXX_M_MYY","PZ","PX","PY"]:
            dirname = ncfile_dir + '/' + stype
            if os.path.exists(dirname):
                self.iodict[stype] = h5py.File(dirname + '/Data/axisem_fields.h5',"r")

        # check if iodit is empty
        if len(self.iodict) == 0 :
            print(f"no data has been accessed, please check {ncfile_dir}!")

    def read_cmt(self,cmtfile):
        """
        read cmt file, and return 6 moment tensor components
        
        """
        from obspy import read_events
        cat = read_events(cmtfile)[0]
        tensor = cat.focal_mechanisms[0].moment_tensor.tensor
        mzz = tensor.m_rr; mxx = tensor.m_tt; myy = tensor.m_pp
        mxz = tensor.m_rt; myz = tensor.m_rp; mxy = tensor.m_tp
        mzz,mxx,myy,mxz,myz,mxy = map(lambda x: x / self.mag,[mzz,mxx,myy,mxz,myz,mxy])

        return mzz,mxx,myy,mxz,myz,mxy
    
    def _locate_elem(self,s,z):
        from sem_funcs import inside_element
        id_elem = None 

        # get nearest 10 points 
        points = self.kdtree.query([s,z],k=10)[1]
        for tol in [1e-3, 1e-2, 5e-2, 8e-2]:
            for idx in points:
                skel = np.zeros((4,2))
                ctrl_id = self.skelid[idx,:4]
                eltype = self.eltype[idx]
                for i in range(4):
                    skel[i,0] = self.s[ctrl_id[i]]
                    skel[i,1] = self.z[ctrl_id[i]]

                isin,xi,eta = inside_element(s,z,skel,eltype,tolerance=tol)
                if isin:
                    id_elem = idx 
                    break; 

            if id_elem is not None:
                break 
        
        return id_elem,xi,eta
    
    def _get_displ(self,elemid,xi,eta,stype):
        """
        Get displacement for one station

        stel: float 
            elevation, in m
        theta: float 
            epicenter distance, in rad
        ncfile: str
            ncfile which the displ is stored in 

        Returns:
        
        us,up,uz: np.ndarray
            s,p,z components 
            
        """
        from sem_funcs import lagrange_interpol_2D_td
        nt = self.nt 
        ngll = self.ngll

        # allocate space
        us = np.zeros((nt)); up = us * 0; uz = us * 1.

        # cache element
        utemp = np.zeros((nt,ngll,ngll,3),dtype=float,order='F')

        # connectivity 
        ibool = self.ibool[elemid,:,:]
        
        # dataset
        fio:h5py.File = self.iodict[stype]
        disp_s = fio['disp_s']
        disp_z = fio['disp_z']
        disp_p = {}
        flag = 'disp_p' in fio.keys()
        if flag:
            disp_p = fio['disp_p']
       
        for iz in range(ngll):
            for ix in range(ngll):
                iglob = ibool[iz,ix]
                #print(iglob,iz,ix)
                utemp[:,ix,iz,0] = disp_s[iglob,:]
                utemp[:,ix,iz,2] = disp_z[iglob,:]
                if flag :
                    utemp[:,ix,iz,1] = disp_p[iglob,:]
                    #disp_p.read_direct(utemp[:,ix,iz,1],idx,np.s_[0:nt])
        
        # interpolate
        sgll = self.gll
        zgll = self.gll 
        if self.axis[elemid]:
            sgll = self.glj 
        us = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,0],xi,eta)
        up = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,1],xi,eta)
        uz = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,2],xi,eta)
        
        return us,up,uz

    def _get_displ_t(self,t,elemid,xi,eta,stype):
        """
        Get displacement for one station

        stel: float 
            elevation, in m
        theta: float 
            epicenter distance, in rad
        ncfile: str
            ncfile which the displ is stored in 

        Returns:
        
        us,up,uz: np.ndarray
            s,p,z components 
            
        """
        from sem_funcs import lagrange_interpol_2D_td
        ngll = self.ngll

        # allocate space
        us = np.zeros((1)); up = us * 0; uz = us * 1.

        # cache element
        utemp = np.zeros((1,ngll,ngll,3),dtype=float,order='F')
        utemp1 = np.zeros((1,ngll,ngll,3),dtype=float,order='F')

        # connectivity 
        ibool = self.ibool[elemid,:,:]

        # get time step
        it = int((t + self.shift) / self.dtsamp)
        if it < 0 or it >= self.nt-1:
            return 0.0,0.0,0.0
        
        # dataset
        fio:h5py.File = self.iodict[stype]
        disp_s = fio['Snapshots/disp_s']
        disp_z = fio['Snapshots/disp_z']
        disp_p = {}
        flag = 'Snapshots/disp_p' in fio.keys()
        if flag:
            disp_p = fio['Snapshots/disp_p']
       
        for iz in range(ngll):
            for ix in range(ngll):
                iglob = ibool[iz,ix]
                #print(iglob,iz,ix)
                utemp[0,ix,iz,0] = disp_s[it,iglob]
                utemp[0,ix,iz,2] = disp_z[it,iglob]
                utemp1[0,ix,iz,0] = disp_s[it+1,iglob]
                utemp1[0,ix,iz,2] = disp_z[it+1,iglob]
                if flag :
                    utemp[:,ix,iz,1] = disp_p[it,iglob]
                    utemp1[0,ix,iz,1] = disp_p[it+1,iglob]
        fio.close()
        
        # interpolate
        sgll = self.gll
        zgll = self.gll 
        if self.axis[elemid]:
            sgll = self.glj 
        us1 = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,0],xi,eta)
        up1 = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,1],xi,eta)
        uz1 = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,2],xi,eta)

        us2 = lagrange_interpol_2D_td(sgll,zgll,utemp1[:,:,:,0],xi,eta)
        up2 = lagrange_interpol_2D_td(sgll,zgll,utemp1[:,:,:,1],xi,eta)
        uz2 = lagrange_interpol_2D_td(sgll,zgll,utemp1[:,:,:,2],xi,eta)
        
        fac = (t - it * self.dtsamp - self.shift) / self.dtsamp
        us = us1 + fac * (us2 - us1) 
        up = up1 + fac * (up2 - up1) 
        uz = uz1 + fac * (uz2 - uz1) 

        return us,up,uz

    def _get_excitation_type(self,stype:str) -> str :
        if stype in ['MZZ',"PZ",'MXX_P_MYY']:
            return 'monopole'
        elif stype in ['MXZ_MYZ',"PX","PY"]:
            return 'dipole'
        else:         
            return 'quadpole'

    def _get_strain(self,elemid,xi,eta,stype):
        """
        get strain field from file

        Returns:
        strain : np.ndarray
                shape(6,nt), ess,epp,ezz,epz,esz,esp
        """
        from sem_funcs import lagrange_interpol_2D_td,strain_td
        nt = self.nt 
        nglob = self.nglob 
        ngll = self.ngll

        # allocate space
        eps = np.zeros((6,nt))

        # get source type
        etype = self._get_excitation_type(stype)

        # cache element
        utemp = np.zeros((nt,ngll,ngll,3),dtype=float,order='F')

        # connectivity 
        ibool = self.ibool[elemid,:,:]
        
        # dataset
        fio:h5py.File = self.iodict[stype]
        disp_s = fio['disp_s']
        disp_z = fio['disp_z']
        disp_p = {}
        flag = 'disp_p' in fio.keys()
        if flag:
            disp_p = fio['disp_p']
        
        # read utemp 
        
        for iz in range(ngll):
            for ix in range(ngll):
                iglob = ibool[iz,ix]
                #print(iglob,iz,ix)
                utemp[:,ix,iz,0] = disp_s[iglob,:]
                utemp[:,ix,iz,2] = disp_z[iglob,:]
                if flag :
                    utemp[:,ix,iz,1] = disp_p[iglob,:]
                    #disp_p.read_direct(utemp[:,ix,iz,1],idx,np.s_[0:nt])
        # gll/glj array
        sgll = self.gll
        zgll = self.gll 
        if self.axis[elemid]:
            sgll = self.glj 

        # control points
        skel = np.zeros((4,2))
        ctrl_id = self.skelid[elemid,:4]
        eltype = self.eltype[elemid]
        for i in range(4):
            skel[i,0] = self.s[ctrl_id[i]]
            skel[i,1] = self.z[ctrl_id[i]]

        if self.axis[elemid]:
            G = self.G2 
            GT = self.G1T 
        else:
            G = self.G2 
            GT = self.G2T 

        # compute strain shape(nt,npol+1,npol+1,6)
        strain = strain_td(utemp,G,GT,sgll,zgll,ngll-1,self.nt,
                            skel,eltype,self.axis[elemid]==1,etype)

        # interpolate 
        # es shape(6,nt)
        for j in range(6):
            eps[j,:] = lagrange_interpol_2D_td(sgll,zgll,strain[:,:,:,j],xi,eta)
        
        return eps

    def _get_stress(self,elemid,xi,eta,stype):
        """
        get stress field from file

        Returns:
        stress : np.ndarray
                shape(6,nt), ess,epp,ezz,epz,esz,esp
        """
        from sem_funcs import lagrange_interpol_2D_td,strain_td
        nt = self.nt 
        ngll = self.ngll

        # get source type
        etype = self._get_excitation_type(stype)
    
        # cache element
        utemp = np.zeros((nt,ngll,ngll,3),dtype=float,order='F')

        # connectivity 
        ibool = self.ibool[elemid,:,:]
        
        # dataset
        fio:h5py.File = self.iodict[stype]
        disp_s = fio['disp_s']
        disp_z = fio['disp_z']
        disp_p = {}
        flag = 'disp_p' in fio.keys()
        if flag:
            disp_p = fio['disp_p']
        
        # read utemp 
        for iz in range(ngll):
            for ix in range(ngll):
                iglob = ibool[iz,ix]
                #print(iglob,iz,ix)
                utemp[:,ix,iz,0] = disp_s[iglob,:]
                utemp[:,ix,iz,2] = disp_z[iglob,:]
                if flag :
                    utemp[:,ix,iz,1] = disp_p[iglob,:]
                    #disp_p.read_direct(utemp[:,ix,iz,1],idx,np.s_[0:nt])

        # alloc arrays for mu and lambda
        xmu = np.zeros((ngll,ngll),dtype=float,order='F')
        xlam = np.zeros((ngll,ngll),dtype=float,order='F')
        for iz in range(ngll):
            for ix in range(ngll):
                iglob = ibool[iz,ix]
                xmu[ix,iz] = self.mu[iglob]
                xlam[ix,iz] = self.lamda[iglob]

        # gll/glj array
        sgll = self.gll
        zgll = self.gll 
        if self.axis[elemid]:
            sgll = self.glj 

        # control points
        skel = np.zeros((4,2))
        ctrl_id = self.skelid[elemid,:4]
        eltype = self.eltype[elemid]
        for i in range(4):
            skel[i,0] = self.s[ctrl_id[i]]
            skel[i,1] = self.z[ctrl_id[i]]

        if self.axis[elemid]:
            G = self.G2 
            GT = self.G1T 
        else:
            G = self.G2 
            GT = self.G2T 

        # compute strain shape(nt,npol+1,npol+1,6)
        strain = strain_td(utemp,G,GT,sgll,zgll,ngll-1,self.nt,
                            skel,eltype,self.axis[elemid]==1,etype)

        # compute stress
        stress = strain * 0.
        stress[...,0] = (xlam + 2 * xmu) * strain[...,0] + xlam * (strain[...,1] + strain[...,2])
        stress[...,1] = (xlam + 2 * xmu) * strain[...,1] + xlam * (strain[...,0] + strain[...,2])
        stress[...,2] = (xlam + 2 * xmu) * strain[...,2] + xlam * (strain[...,0] + strain[...,1])
        stress[...,3] = 2. * xmu * strain[...,3]
        stress[...,4] = 2. * xmu * strain[...,4]
        stress[...,5] = 2. * xmu * strain[...,5]

        # interpolate 
        # es shape(6,nt)
        sigma = np.zeros((6,nt))
        for j in range(6):
            sigma[j,:] = lagrange_interpol_2D_td(sgll,zgll,stress[:,:,:,j],xi,eta)
        
        return sigma

    def _get_stress_t(self,t,elemid,xi,eta,ncfile):
        """
        get stress field from file

        Returns:
        stress : np.ndarray
                shape(6), ess,epp,ezz,epz,esz,esp
        """
        from sem_funcs import lagrange_interpol_2D_td,strain_td 
        ngll = self.ngll

        # get time step
        it = int((t + self.shift) / self.dtsamp)
        if it < 0 or it >= self.nt-1:
            return np.zeros((6),dtype=float)
    
        # cache element
        utemp = np.zeros((1,ngll,ngll,3),dtype=float,order='F')
        utemp1 = np.zeros((1,ngll,ngll,3),dtype=float,order='F')

        # get source type
        ncfile_org = ncfile
        fio = h5py.File(ncfile_org,"r")
        stype = fio.attrs['excitation type'].decode("utf-8")
        fio.close()

        # connectivity 
        ibool = self.ibool[elemid,:,:]
        
        # dataset
        fio = h5py.File(ncfile,"r")
        disp_s = fio['Snapshots/disp_s']
        disp_z = fio['Snapshots/disp_z']
        disp_p = {}
        flag = 'Snapshots/disp_p' in fio.keys()
        if flag:
            disp_p = fio['Snapshots/disp_p']
        
        for iz in range(ngll):
            for ix in range(ngll):
                iglob = ibool[iz,ix]
                #print(iglob,iz,ix)
                utemp[0,ix,iz,0] = disp_s[it,iglob]
                utemp[0,ix,iz,2] = disp_z[it,iglob]
                utemp1[0,ix,iz,0] = disp_s[it+1,iglob]
                utemp1[0,ix,iz,2] = disp_z[it+1,iglob]
                if flag :
                    utemp[0,ix,iz,1] = disp_p[it,iglob]
                    utemp1[0,ix,iz,1] = disp_p[it+1,iglob]
        fio.close()

        # alloc arrays for mu and lambda
        xmu = np.zeros((ngll,ngll),dtype=float,order='F')
        xlam = np.zeros((ngll,ngll),dtype=float,order='F')
        for iz in range(ngll):
            for ix in range(ngll):
                iglob = ibool[iz,ix]
                xmu[ix,iz] = self.mu[iglob]
                xlam[ix,iz] = self.lamda[iglob]

        # gll/glj array
        sgll = self.gll
        zgll = self.gll 
        if self.axis[elemid]:
            sgll = self.glj 

        # control points
        skel = np.zeros((4,2))
        ctrl_id = self.skelid[elemid,:4]
        eltype = self.eltype[elemid]
        for i in range(4):
            skel[i,0] = self.s[ctrl_id[i]]
            skel[i,1] = self.z[ctrl_id[i]]

        if self.axis[elemid]:
            G = self.G2 
            GT = self.G1T 
        else:
            G = self.G2 
            GT = self.G2T 

        # compute strain shape(nt,npol+1,npol+1,6)
        strain = strain_td(utemp,G,GT,sgll,zgll,ngll-1,1,
                            skel,eltype,self.axis[elemid]==1,stype)
        strain1 = strain_td(utemp1,G,GT,sgll,zgll,ngll-1,1,
                            skel,eltype,self.axis[elemid]==1,stype)

        # compute stress
        stress = strain * 0.; stress1 = strain1 * 0.
        stress[...,0] = (xlam + 2 * xmu) * strain[...,0] + xlam * (strain[...,1] + strain[...,2])
        stress[...,1] = (xlam + 2 * xmu) * strain[...,1] + xlam * (strain[...,0] + strain[...,2])
        stress[...,2] = (xlam + 2 * xmu) * strain[...,2] + xlam * (strain[...,0] + strain[...,1])
        stress[...,3] = 2. * xmu * strain[...,3]
        stress[...,4] = 2. * xmu * strain[...,4]
        stress[...,5] = 2. * xmu * strain[...,5]
        stress1[...,0] = (xlam + 2 * xmu) * strain1[...,0] + xlam * (strain1[...,1] + strain1[...,2])
        stress1[...,1] = (xlam + 2 * xmu) * strain1[...,1] + xlam * (strain1[...,0] + strain1[...,2])
        stress1[...,2] = (xlam + 2 * xmu) * strain1[...,2] + xlam * (strain1[...,0] + strain1[...,1])
        stress1[...,3] = 2. * xmu * strain1[...,3]
        stress1[...,4] = 2. * xmu * strain1[...,4]
        stress1[...,5] = 2. * xmu * strain1[...,5]

        fac = (t - it * self.dtsamp - self.shift) / self.dtsamp
        stress = stress + fac * (stress1 - stress)

        # interpolate 
        # es shape(6,nt)
        sigma = np.zeros((6))
        for j in range(6):
            sigma[j] = lagrange_interpol_2D_td(sgll,zgll,stress[:,:,:,j],xi,eta)
        
        return sigma

    def compute_tp_recv(self,stla,stlo):
        """
        compute theta and phi for source centered coordinates
        stla: float
            latitude of station, in deg
        stlo: float
            longitude of station, in deg 
        """
        x = np.cos(stla * np.pi/180) * np.cos(stlo * np.pi/180)
        y = np.cos(stla * np.pi/180) * np.sin(stlo * np.pi/180)
        z = np.sin(stla * np.pi/ 180)

        # rotate xyz to source centered system
        x1,y1,z1 = np.dot(self.rot_mat.T,np.array([x,y,z]))

        # to phi and theta
        r = np.sqrt(x1**2 + y1**2 + z1**2)
        x1 /=r; y1 /= r; z1 /= r
        #r = np.sqrt(x1**2 + y1**2)
        theta = np.arccos(z1/r)
        phi = np.arctan2(y1,x1)

        return theta,phi
    
    def compute_local(self,theta,stel):
        """
        compute local coordinates in axisem system
        theta float
            epicenter distance, in rad
        stel: float
            elevation of the station, in m 
        """
        # locate point 
        r = 6371000 + stel
        sr = r * np.sin(theta)
        zr = r * np.cos(theta)

        return sr,zr

    def syn_seismo(self,stla,stlo,stel,comp:str='enz',cmtfile=None,forcevec=None):
        """
        comp: Specify the orientation of the synthetic seismograms as a list
                one of [enz,xyz,spz]
        """
        # check components
        comp = comp.lower()
        assert(comp in ['enz','xyz','spz'])

        # read source type
        assert((cmtfile is not None) or (forcevec is not None))
        mzz,mxx,myy,mxz,myz,mxy = [0. for i in range(6)]
        fx,fy,fz = [0.,0.,0.]
        srctypes = []

        if cmtfile is not None:
            mzz,mxx,myy,mxz,myz,mxy = self.read_cmt(cmtfile)
            srctypes = ['MZZ',"MXX_P_MYY","MXZ_MYZ","MXY_MXX_M_MYY"]
        else:
            fx,fy,fz = forcevec
            srctypes = ["PZ","PX","PY"]


        # alloc space for seismograms
        nt = self.nt 
        us = np.zeros((nt))
        uz = us.copy(); up = us.copy()

        # compute rotated station phi,theta
        theta,phi = self.compute_tp_recv(stla,stlo)
        sr,zr = self.compute_local(theta,stel)

        # locate element
        elemid,xi,eta = self._locate_elem(sr,zr)

        # loop every source type
        for stype in srctypes:
            #print("synthetic seismograms for  ... %s" %(stype))
            # get basic waveform
            us1,up1,uz1 = self._get_displ(elemid,xi,eta,stype)

            # parameters
            a = 0.; b = 0.
            if stype == 'MZZ':  # mono
                a = mzz 
                b = 0.
            elif stype == "PZ":
                a = fz 
                b = 0.
            elif stype == 'MXX_P_MYY': # mono
                a = mxx + myy
                b = 0            
            
            # interpolate
            cosphi = np.cos(phi); sinphi = np.sin(phi)
            cos2phi = np.cos(2 * phi); sin2phi = np.sin(2 * phi)
            if stype  == 'MXZ_MYZ':
                a = mxz * cosphi + myz * sinphi
                b = myz * cosphi - mxz * sinphi
            elif stype == "MXY_MXX_M_MYY":
                a = (mxx - myy) * cos2phi + 2 * mxy * sin2phi 
                b = -(mxx - myy) * sin2phi + 2 * mxy * cos2phi 
            
            elif stype == 'PX' or stype == "PY":
                a = fx * cosphi + fy * sinphi 
                b = -fx * sinphi + fy * cosphi

            # normalize
            us1[:] *= a; up1[:] *= b; uz1[:] *= a
            
            # add contribution from each term
            us += us1 
            up += up1 
            uz += uz1 

        # rotate to specified coordinates
        u1 = uz * 0.; u2 = up * 0.; u3 = up * 0.   
        # rotation matrix to enz
        R1 = np.eye(3) # rotate from (s,phi,z) to (xs,ys,z)
        R1[0,:2] = [np.cos(phi),-np.sin(phi)]
        R1[1,:2] = [np.sin(phi),np.cos(phi)]
        Rr = rotation_matrix(np.deg2rad(90-stla),np.deg2rad(stlo))

        if comp == 'enz':
            Rr = Rr.T @ self.rot_mat @ R1
        elif comp == 'spz':
            Rr = np.eye(3)
        else:
            Rr = self.rot_mat @ R1 

        u1 = Rr[0,0] * us + Rr[0,1] * up + Rr[0,2] * uz 
        u2 = Rr[1,0] * us + Rr[1,1] * up + Rr[1,2] * uz 
        u3 = Rr[2,0] * us + Rr[2,1] * up + Rr[2,2] * uz 
        
        if comp == 'enz':
            u1 = -u1
            temp = u1.copy()
            u1 = u2.copy()
            u2 = temp.copy()

        return u1,u2,u3


    def syn_strain(self,simu_path,stla,stlo,stel,cmtfile=None,forcevec=None):
        # read source type
        assert((cmtfile is not None) or (forcevec is not None))
        mzz,mxx,myy,mxz,myz,mxy = [0. for i in range(6)]
        fx,fy,fz = [0.,0.,0.]
        srctypes = []

        if cmtfile is not None:
            mzz,mxx,myy,mxz,myz,mxy = self.read_cmt(cmtfile)
            srctypes = ['MZZ',"MXX_P_MYY","MXZ_MYZ","MXY_MXX_M_MYY"]
        else:
            fx,fy,fz = forcevec
            srctypes = ["PZ","PX_PY"]

        # alloc space for seismograms
        nt = self.nt 
        eps = np.zeros((6,nt))

        # compute rotated station phi,theta
        theta,phi = self.compute_tp_recv(stla,stlo)
        sr,zr = self.compute_local(theta,stel)
        
        # locate element
        elemid,xi,eta = self._locate_elem(sr,zr)

        # loop every source type
        for stype in srctypes:
            #print("synthetic strain tensor for  ... %s" %(stype))
            # get basic waveform
            #filename = simu_path + "/" + stype + "/Data/axisem_fields.h5"
            eps0 = self._get_strain(elemid,xi,eta,stype)

            # parameters
            a = 0.; b = 0.
            if stype == 'MZZ':  # mono
                a = mzz 
                b = 0.
            elif stype == "PZ":
                a = fz 
                b = 0.
            elif stype == 'MXX_P_MYY': # mono
                a = mxx + myy
                b = 0            
            
            # interpolate
            cosphi = np.cos(phi); sinphi = np.sin(phi)
            cos2phi = np.cos(2 * phi); sin2phi = np.sin(2 * phi)
            if stype  == 'MXZ_MYZ':
                a = mxz * cosphi + myz * sinphi
                b = myz * cosphi - mxz * sinphi
            elif stype == "MXY_MXX_M_MYY":
                a = (mxx - myy) * cos2phi + 2 * mxy * sin2phi 
                b = -(mxx - myy) * sin2phi + 2 * mxy * cos2phi 
            
            elif stype == "PX_PY":
                a = fx * cosphi + fy * sinphi 
                b = -fx * sinphi + fy * cosphi

            # normalize
            # eps: ss pp zz pz sz sp 
            eps0[0:3,:] *= a; eps0[4,:] *= a
            eps0[3,:] *= b; eps0[5:,:] *= b
            
            # add contribution from each term
            eps += eps0

        # rotate to xyz 
        R1 = np.eye(3) # rotate from (s,phi,z) to (xs,ys,z)
        R1[0,:2] = [np.cos(phi),-np.sin(phi)]
        R1[1,:2] = [np.sin(phi),np.cos(phi)]
        R = self.rot_mat @ R1
        eps_xyz = rotate_tensor2(eps,R)

        return eps_xyz


    def syn_stress(self,stla,stlo,stel,cmtfile=None,forcevec=None):
        # read source type
        assert((cmtfile is not None) or (forcevec is not None))
        mzz,mxx,myy,mxz,myz,mxy = [0. for i in range(6)]
        fx,fy,fz = [0.,0.,0.]
        srctypes = []

        if cmtfile is not None:
            mzz,mxx,myy,mxz,myz,mxy = self.read_cmt(cmtfile)
            srctypes = ['MZZ',"MXX_P_MYY","MXZ_MYZ","MXY_MXX_M_MYY"]
        else:
            fx,fy,fz = forcevec
            srctypes = ["PZ","PX_PY"]

        # alloc space for seismograms
        nt = self.nt 
        sigma = np.zeros((6,nt))

        # compute rotated station phi,theta
        theta,phi = self.compute_tp_recv(stla,stlo)
        sr,zr = self.compute_local(theta,stel)
        
        # locate element
        elemid,xi,eta = self._locate_elem(sr,zr)

        # loop every source type
        for stype in srctypes:
            #print("synthetic strain tensor for  ... %s" %(stype))
            # get basic waveform
            #filename = simu_path + "/" + stype + "/Data/axisem_fields.h5"
            eps0 = self._get_stress(elemid,xi,eta,stype)

            # parameters
            a = 0.; b = 0.
            if stype == 'MZZ':  # mono
                a = mzz 
                b = 0.
            elif stype == "PZ":
                a = fz 
                b = 0.
            elif stype == 'MXX_P_MYY': # mono
                a = mxx + myy
                b = 0            
            
            # interpolate
            cosphi = np.cos(phi); sinphi = np.sin(phi)
            cos2phi = np.cos(2 * phi); sin2phi = np.sin(2 * phi)
            if stype  == 'MXZ_MYZ':
                a = mxz * cosphi + myz * sinphi
                b = myz * cosphi - mxz * sinphi
            elif stype == "MXY_MXX_M_MYY":
                a = (mxx - myy) * cos2phi + 2 * mxy * sin2phi 
                b = -(mxx - myy) * sin2phi + 2 * mxy * cos2phi 
            
            elif stype == "PX_PY":
                a = fx * cosphi + fy * sinphi 
                b = -fx * sinphi + fy * cosphi

            # normalize
            eps0[0:3,:] *= a; eps0[4,:] *= a
            eps0[3,:] *= b; eps0[5:,:] *= b
            
            # add contribution from each term
            sigma += eps0

        # rotate to xyz 
        R1 = np.eye(3) # rotate from (s,phi,z) to (xs,ys,z)
        R1[0,:2] = [np.cos(phi),-np.sin(phi)]
        R1[1,:2] = [np.sin(phi),np.cos(phi)]
        R = self.rot_mat @ R1
        sigma_xyz = rotate_tensor2(sigma,R)

        return sigma_xyz