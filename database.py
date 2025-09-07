import numpy as np 
import h5py 
from utils import rotation_matrix,rotate_tensor2

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
        self.t0 = fio.attrs['dump_t0']
        self.dominant_T0 = fio.attrs['dominant source period'][0]

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
        self.mesh_s = fio['Mesh/mesh_S'][:]
        self.mesh_z = fio['Mesh/mesh_Z'][:]

        # elastic moduli
        self.xmu = fio['Mesh/mesh_mu'][:]
        self.xlamda = fio['Mesh/mesh_lambda'][:]
        self.xxi = fio['Mesh/mesh_xi'][:]
        self.xphi = fio['Mesh/mesh_phi'][:]
        self.xeta = fio['Mesh/mesh_eta'][:]

        # check if the media contains acoustic elements
        self.is_elastic = np.zeros((self.nspec),dtype=bool)
        self.is_elastic[:] = True
        self.nspec_el = 0
        self.nspec_ac = 0
        for ispec in range(self.xmu.shape[0]):
            if np.mean(self.xmu[ispec,:,:]) < 1.0e-5:
                self.nspec_ac += 1
                self.is_elastic[ispec] = False
            else:
                self.nspec_el += 1


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

        # close file
        fio.close()

        # data file dict
        self.iodict = {}
    
    def __copy__(self):
        """
        shallow copy of necessary basic variables
        """
        db = AxiBasicDB()

        # shallow copy of necessary variables
        db.nspec = self.nspec
        db.nctrl = self.nctrl
        db.ngll  = self.ngll

        # attributes
        db.dtsamp = self.dtsamp
        db.shift = self.shift
        db.nt = self.nt
        db.nglob = self.nglob
        db.t0 = self.t0
    
        # source parameters
        db.evdp = self.evdp
        db.evla = self.evla
        db.evlo = self.evlo
        db.mag = self.mag

        # rotation matrix
        db.rot_mat = self.rot_mat * 1.

        # read mesh 
        db.mesh_s = self.mesh_s
        db.mesh_z = self.mesh_z

        # create kdtree
        db.kdtree = self.kdtree

        # elemtype
        db.eltype = self.eltype
        db.axis = self.axis

        # skeleton
        db.skelid = self.skelid

        # connectivity[
        db.ibool = self.ibool

        # deep copy useful arrays
        db.G0 = self.G0.copy()
        db.G1 = self.G1.copy()
        db.G2 = self.G2.copy()
        db.G1T = self.G1T.copy()
        db.G2T = self.G2T.copy()
        db.gll = self.gll.copy()
        db.glj = self.glj.copy()

        # elastic moduli
        db.mu = self.mu
        db.lamda = self.lamda

        # data file dict
        db.iodict = {}

        return db

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
    
    def copy(self):
        return self.__copy__()
    
    def close(self):
        for _,val in self.iodict.items():
            val.close()
        self.iodict = {}
    
    def set_source(self,evla:float,evlo:float):
        """
        update source info in the database
        evla: float
            latitude of station, in deg
        evlo: float
            longitude of station, in deg 
        """
        self.evla = evla
        self.evlo = evlo 
        self.rot_mat = rotation_matrix(np.pi/2-np.deg2rad(evla),np.deg2rad(evlo))
        pass

    def read_cmt(self,cmtfile:str):
        from utils import read_cmtsolution
        mzz,mxx,myy,mxz,myz,mxy = read_cmtsolution(cmtfile)
        mzz,mxx,myy,mxz,myz,mxy = map(lambda x: x / self.mag,[mzz,mxx,myy,mxz,myz,mxy])

        return mzz,mxx,myy,mxz,myz,mxy
    
    def _locate_elem(self,s,z,is_el_point = True):
        from sem_funcs import inside_element
        id_elem = None 

        # get nearest 10 points 
        points = self.kdtree.query([s,z],k=10)[1]
        for tol in [1e-3, 1e-2, 5e-2, 8e-2]:
            for idx in points:
                skel = np.zeros((self.nctrl,2))
                ctrl_id = self.skelid[idx,:]
                eltype = self.eltype[idx]
                for i in range(self.nctrl):
                    skel[i,0] = self.mesh_s[ctrl_id[i]]
                    skel[i,1] = self.mesh_z[ctrl_id[i]]

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

        # allocate space
        us = np.zeros((nt)); up = us * 0; uz = us * 1.
        
        # dataset info
        fio:h5py.File = self.iodict[stype]
        ngll = fio['disp_s'].shape[1]
        utemp = np.zeros((3,ngll,ngll,nt),dtype=float)

        # read dataset
        utemp[0,...] = fio['disp_s'][elemid,...]
        utemp[2,...] = fio['disp_z'][elemid,...]
        if 'disp_p' in fio.keys():
            utemp[1,...] = fio['disp_p'][elemid,...]
        utemp = np.transpose(utemp,(3,2,1,0))

        sgll = self.gll
        zgll = self.gll
        flag = self.axis[elemid] == 1
        if flag:
            sgll = self.glj
        us = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,0],xi,eta)
        up = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,1],xi,eta)
        uz = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,2],xi,eta)
        
        return us,up,uz
    
    def _get_excitation_type(self,stype:str) -> str :
        if stype in ['MZZ',"PZ",'MXX_P_MYY']:
            return 'monopole'
        elif stype in ['MXZ_MYZ',"PX","PY"]:
            return 'dipole'
        else:         
            return 'quadpole'
    
    def _get_strain(self,elemid:int,xi:float,eta:float,stype:str):
        """
        get strain field at a given point, for a given source type

        Parameters:
        ===================================================
        elemid: current 
        xi/eta: local coordinates 
        stype: source type

        Returns:
        ====================================================
        strain : np.ndarray
                shape(6,nt), ess,epp,ezz,epz,esz,esp
        """
        from sem_funcs import lagrange_interpol_2D_td,strain_td
        nt = self.nt
        fio:h5py.File = self.iodict[stype]
        ngll = fio['disp_s'].shape[1]

        # allocate space
        eps = np.zeros((6,nt))

        # cache element
        utemp = np.zeros((3,ngll,ngll,nt),dtype=float)
        
        # dataset
        # read dataset
        utemp[0,...] = fio['disp_s'][elemid,...]
        utemp[2,...] = fio['disp_z'][elemid,...]
        if 'disp_p' in fio.keys():
            utemp[1,...] = fio['disp_p'][elemid,...]
        utemp = np.transpose(utemp,(3,2,1,0))

        # gll/glj array
        sgll = self.gll
        zgll = self.gll
        is_axi = self.axis[elemid] == 1
        if is_axi:
            sgll = self.glj

        # control points
        skel = np.zeros((self.nctrl,2))
        ctrl_id = self.skelid[elemid,:]
        eltype = self.eltype[elemid]
        skel[:,0] = self.mesh_s[ctrl_id]
        skel[:,1] = self.mesh_z[ctrl_id]
        if is_axi:
            G = self.G2 
            GT = self.G1T 
        else:
            G = self.G2 
            GT = self.G2T 

        # compute strain shape(nt,npol+1,npol+1,6)
        etype = self._get_excitation_type(stype)
        strain = strain_td(utemp,G,GT,sgll,zgll,ngll-1,nt,
                            skel,eltype,is_axi,etype)

        # interpolate 
        # es shape(6,nt)
        for j in range(6):
            eps[j,:] = lagrange_interpol_2D_td(sgll,zgll,strain[:,:,:,j],xi,eta)
        
        return eps

    def _get_stress(self,elemid,xi,eta,stype):
        """
        get stress field for a given point from file

        Returns:
        stress : np.ndarray
                shape(6,nt), ess,epp,ezz,epz,esz,esp
        """
        from sem_funcs import lagrange_interpol_2D_td,strain_td,find_theta
        from utils import c_ijkl_ani
        nt = self.nt 
        fio:h5py.File = self.iodict[stype]
        ngll = fio['disp_s'].shape[1]
    
        # cache element
        utemp = np.zeros((3,ngll,ngll,nt),dtype=float)
        
        # dataset
        utemp[0,...] = fio['disp_s'][elemid,...]
        utemp[2,...] = fio['disp_z'][elemid,...]
        if 'disp_p' in fio.keys():
            utemp[1,...] = fio['disp_p'][elemid,...]
        utemp = np.transpose(utemp,(3,2,1,0))

        # alloc arrays for elastic tensor
        xmu = np.zeros((ngll,ngll),dtype=float)
        xlam = np.zeros((ngll,ngll),dtype=float)
        xxi = xlam * 1. 
        xphi = xlam * 1. 
        xeta = xlam * 1.
        xmu[:,:] = self.xmu[elemid,:,:]; xlam[:,:] = self.xlamda[elemid,:,:]
        xxi[:,:] = self.xxi[elemid,:,:]; xphi[:,:] = self.xphi[elemid,:,:]
        xeta[:,:] = self.xeta[elemid,:,:]
        xmu = np.transpose(xmu); xlam = np.transpose(xlam)
        xxi = np.transpose(xxi); xphi = np.transpose(xphi)
        xeta = np.transpose(xeta)

        # gll/glj array
        sgll = self.gll
        zgll = self.gll
        is_axi = self.axis[elemid] == 1
        if is_axi:
            sgll = self.glj

        # control points
        skel = np.zeros((self.nctrl,2))
        ctrl_id = self.skelid[elemid,:]
        eltype = self.eltype[elemid]
        skel[:,0] = self.mesh_s[ctrl_id]
        skel[:,1] = self.mesh_z[ctrl_id]

        if self.axis[elemid]:
            G = self.G2 
            GT = self.G1T 
        else:
            G = self.G2 
            GT = self.G2T 

        # compute strain shape(nt,npol+1,npol+1,6)
        etype = self._get_excitation_type(stype)
        e = strain_td(utemp,G,GT,sgll,zgll,ngll-1,self.nt,
                            skel,eltype,self.axis[elemid]==1,etype)
        
        # find theta
        theta = find_theta(sgll,zgll,skel,eltype)

        # get elastic tensor c21
        c11 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 1, 1)
        c12 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 2, 2)
        c13 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 3, 3)
        c15 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 3, 1)
        c22 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 2, 2, 2)
        c23 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 2, 3, 3)
        c25 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 2, 3, 1)
        c33 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 3, 3, 3, 3)
        c35 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 3, 3, 3, 1)
        c44 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 3, 2, 3)
        c46 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 3, 1, 2)
        c55 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 3, 1, 3, 1)
        c66 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 2, 1, 2)
        c14 = 0.; c26 = 0.; c36 = 0.; c24 = 0.
        c16 = 0.; c45 = 0.; c56 = 0.; c34 = 0.

        # compute stress
        stress = e * 0.
        e[..., 3:6] = 2. * e[...,3:6]

        # Compute stress components explicitly using Voigt notation
        stress[..., 0] = c11 * e[..., 0] + c16 * e[..., 5] + c12 * e[..., 1] + c15 * e[..., 4] + c14 * e[..., 3] + c13 * e[..., 2]  # sxx → s[..., 0]
        stress[..., 1] = c12 * e[..., 0] + c26 * e[..., 5] + c22 * e[..., 1] + c25 * e[..., 4] + c24 * e[..., 3] + c23 * e[..., 2]  # syy → s[..., 1]
        stress[..., 2] = c13 * e[..., 0] + c36 * e[..., 5] + c23 * e[..., 1] + c35 * e[..., 4] + c34 * e[..., 3] + c33 * e[..., 2]  # szz → s[..., 2]
        stress[..., 3] = c14 * e[..., 0] + c46 * e[..., 5] + c24 * e[..., 1] + c45 * e[..., 4] + c44 * e[..., 3] + c34 * e[..., 2]  # syz → s[..., 3]
        stress[..., 4] = c15 * e[..., 0] + c56 * e[..., 5] + c25 * e[..., 1] + c55 * e[..., 4] + c45 * e[..., 3] + c35 * e[..., 2]  # sxz → s[..., 4]
        stress[..., 5] = c16 * e[..., 0] + c66 * e[..., 5] + c26 * e[..., 1] + c56 * e[..., 4] + c46 * e[..., 3] + c36 * e[..., 2]  # sxy → s[..., 5]

        # interpolate 
        # es shape(6,nt)
        sigma = np.zeros((6,nt))
        for j in range(6):
            sigma[j,:] = lagrange_interpol_2D_td(sgll,zgll,stress[:,:,:,j],xi,eta)
        
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
        # if phi < 0:
        #     phi = np.pi - phi

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

    def syn_strain(self,stla,stlo,stel,cmtfile=None,forcevec=None):
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