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
    
    return eps_xyz

class AxiBasicDB:
    def __init__(self,ncfile) -> None:
        """
        ncfile : str
            input netcdf file 
        """
        # load library
        from scipy.spatial import KDTree

        print("reading database %s ..."%ncfile)
        self.ncfile_basic = ncfile 

        f = h5py.File(ncfile,"r")
        self.nspec = len(f['Mesh/elements'])
        self.nctrl = len(f['Mesh/control_points'])
        self.ngll = len(f['Mesh/npol'])

        # read attributes
        self.dtsamp = f.attrs['strain dump sampling rate in sec'][0]
        self.shift = f.attrs['source shift factor in sec'][0]
        self.nt = len(f['snapshots'])
        self.nglob = len(f['gllpoints_all'])
    
        # read source parameters
        self.evdp = f.attrs['source depth in km'][0]
        evcola = f.attrs['Source colatitude'][0]
        evlo = f.attrs['Source longitude'][0]
        self.evla = 90 - np.rad2deg(evcola) 
        self.evlo = np.rad2deg(evlo)
        self.mag = f.attrs['scalar source magnitude'][0]

        # rotation matrix
        self.rot_mat = rotation_matrix(evcola,evlo)

        # read mesh 
        self.s = f['Mesh/mesh_S']
        self.z = f['Mesh/mesh_Z']

        # create kdtree
        md_pts = np.zeros((self.nspec,2))
        md_pts[:,0] = f['Mesh/mp_mesh_S'][:]
        md_pts[:,1] = f['Mesh/mp_mesh_Z'][:]
        self.kdtree = KDTree(data=md_pts)

        # elemtype
        self.eltype = f['Mesh/eltype']
        self.axis = f['Mesh/axis']

        # skeleton
        self.skelid = f['Mesh/fem_mesh']

        # connectivity
        self.ibool = f['Mesh/sem_mesh']

        # other useful arrays
        self.G0 = f['Mesh/G0'][:]
        self.G1 = f['Mesh/G1'][:].T
        self.G2 = f['Mesh/G2'][:].T 
        self.G1T = np.require(self.G1.T,requirements=['F_CONTIGUOUS'])
        self.G2T = np.require(self.G2.T,requirements=['F_CONTIGUOUS'])

        self.gll = f['Mesh/gll'][:]
        self.glj = f['Mesh/glj'][:]

        # elastic moduli
        self.mu = f['Mesh/mesh_mu']
        self.lamda = f['Mesh/mesh_lambda']

    def read_cmt(self,cmtfile):
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
    
    def _get_displ(self,elemid,xi,eta,ncfile):
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
        fio = h5py.File(ncfile,"r")
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
        fio.close()
        
        # interpolate
        sgll = self.gll
        zgll = self.gll 
        if self.axis[elemid]:
            sgll = self.glj 
        us = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,0],xi,eta)
        up = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,1],xi,eta)
        uz = lagrange_interpol_2D_td(sgll,zgll,utemp[:,:,:,2],xi,eta)
        
        return us,up,uz

    def _get_strain(self,elemid,xi,eta,ncfile):
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
        ncfile_org = ' '.join(ncfile.split("axisem_fields.h5")[:-1]) + "/axisem_output.nc4"
        fio = h5py.File(ncfile_org,"r")
        stype = fio.attrs['excitation type'].decode("utf-8")
        fio.close()

        # cache element
        utemp = np.zeros((nt,ngll,ngll,3),dtype=float,order='F')

        # connectivity 
        ibool = self.ibool[elemid,:,:]
        
        # dataset
        fio = h5py.File(ncfile,"r")
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
        fio.close()

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
                            skel,eltype,self.axis[elemid]==1,stype)

        # interpolate 
        # es shape(6,nt)
        for j in range(6):
            eps[j,:] = lagrange_interpol_2D_td(sgll,zgll,strain[:,:,:,j],xi,eta)
        
        return eps

    def _get_stress(self,elemid,xi,eta,ncfile):
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
        ncfile_org = ' '.join(ncfile.split("axisem_fields.h5")[:-1]) + "/axisem_output.nc4"
        fio = h5py.File(ncfile_org,"r")
        stype = fio.attrs['excitation type'].decode("utf-8")
        fio.close()
    
        # cache element
        utemp = np.zeros((nt,ngll,ngll,3),dtype=float,order='F')

        # connectivity 
        ibool = self.ibool[elemid,:,:]
        
        # dataset
        fio = h5py.File(ncfile,"r")
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
        strain = strain_td(utemp,G,GT,sgll,zgll,ngll-1,self.nt,
                            skel,eltype,self.axis[elemid]==1,stype)

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

    def syn_seismo(self,simu_path,stla,stlo,stel,comp:str='enz',cmtfile=None,forcevec=None):
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
            filename = simu_path + "/" + stype + "/Data/axisem_fields.h5"
            us1,up1,uz1 = self._get_displ(elemid,xi,eta,filename)

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
            filename = simu_path + "/" + stype + "/Data/axisem_fields.h5"
            eps0 = self._get_strain(elemid,xi,eta,filename)

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
            eps += eps0

        # rotate to xyz 
        R1 = np.eye(3) # rotate from (s,phi,z) to (xs,ys,z)
        R1[0,:2] = [np.cos(phi),-np.sin(phi)]
        R1[1,:2] = [np.sin(phi),np.cos(phi)]
        R = self.rot_mat @ R1
        eps_xyz = rotate_tensor2(eps,R)

        return eps_xyz


    def syn_stress(self,simu_path,stla,stlo,stel,cmtfile=None,forcevec=None):
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
            filename = simu_path + "/" + stype + "/Data/axisem_fields.h5"
            eps0 = self._get_stress(elemid,xi,eta,filename)

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