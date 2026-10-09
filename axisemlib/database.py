from pathlib import Path
import numpy as np 
import h5py 
from .utils import rotation_matrix,rotate_tensor2

class AxiBasicDB:
    def __init__(self) -> None:
        pass 

    def read_basic(self,ncfile:str) -> None:
        """
        ncfile : str
            input netcdf file 
        """
        boundary_file = Path(ncfile).with_name('boundary_wavefields.nc4')
        if boundary_file.is_file():
            self._read_boundary_basic(boundary_file)
            return

        self._boundary_mode = False

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
        self.rot_s = rotation_matrix(evcola,evlo)

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
        self.G1 = np.ascontiguousarray(fio['Mesh/G1'][:].T)
        self.G2 = np.ascontiguousarray(fio['Mesh/G2'][:].T)
        self.G1T = np.ascontiguousarray(self.G1.T)
        self.G2T = np.ascontiguousarray(self.G2.T)
        self.gll = fio['Mesh/gll'][:]
        self.glj = fio['Mesh/glj'][:]

        # close file
        fio.close()

        # data file dict
        self.iodict = {}

        # field cache
        self._field_cache:dict[tuple[str,int],np.ndarray] = {}

    def _read_boundary_basic(self,boundary_file:Path) -> None:
        """Read the selected boundary mesh and metadata from a solver run."""
        from scipy.spatial import KDTree

        # The boundary file owns the mesh; the standard output still owns source metadata.
        output_file = boundary_file.with_name('axisem_output.nc4')
        if not output_file.is_file():
            raise FileNotFoundError(f'Source metadata is missing: {output_file}')

        self._boundary_mode = True
        self.boundary_file = boundary_file
        with h5py.File(output_file,'r') as source, h5py.File(boundary_file,'r') as boundary:
            mesh = boundary['Mesh']

            # Set array sizes and sampling information for later interpolation.
            self.nspec = len(mesh['elements'])
            self.nctrl = len(mesh['control_points'])
            self.ngll = len(boundary['npol'])
            self.nt = len(boundary['sample_time'])
            self.sample_time = boundary['sample_time'][:]
            self.nglob = len(boundary['point'])
            if self.nt > 1:
                self.dtsamp = float(self.sample_time[1] - self.sample_time[0])
            else:
                self.dtsamp = float(source.attrs['strain dump sampling rate in sec'][0])
            self.t0 = float(self.sample_time[0])
            self.shift = float(source.attrs['source shift factor in sec'][0])
            self.dominant_T0 = float(source.attrs['dominant source period'][0])

            # Read the source position and rotation used to combine modal fields.
            self.evdp = float(source.attrs['source depth in km'][0])
            evcola = float(source.attrs['Source colatitude'][0])
            evlo = float(source.attrs['Source longitude'][0])
            self.evla = 90 - np.rad2deg(evcola)
            self.evlo = np.rad2deg(evlo)
            self.mag = float(source.attrs['scalar source magnitude'][0])
            self.rot_s = rotation_matrix(evcola, evlo)

            # Load geometry and material data needed by the existing SEM routines.
            self.mesh_s = mesh['mesh_S'][:]
            self.mesh_z = mesh['mesh_Z'][:]
            self.xmu = mesh['mesh_mu'][:]
            self.xlamda = mesh['mesh_lambda'][:]
            self.xxi = mesh['mesh_xi'][:]
            self.xphi = mesh['mesh_phi'][:]
            self.xeta = mesh['mesh_eta'][:]
            self.xrho = mesh['mesh_rho'][:]
            self.is_elastic = np.mean(self.xmu, axis=(1, 2)) >= 1.0e-5
            self.nspec_el = int(self.is_elastic.sum())
            self.nspec_ac = self.nspec - self.nspec_el

            self.eltype = mesh['eltype'][:]
            self.axis = mesh['axis'][:]
            self.skelid = mesh['fem_mesh'][:]
            self.ibool = mesh['sem_mesh'][:]
            self.G0 = mesh['G0'][:]
            self.G1 = np.ascontiguousarray(mesh['G1'][:].T)
            self.G2 = np.ascontiguousarray(mesh['G2'][:].T)
            self.G1T = np.ascontiguousarray(self.G1.T)
            self.G2T = np.ascontiguousarray(self.G2.T)
            self.gll = mesh['gll'][:]
            self.glj = mesh['glj'][:]

            # Index element centers to support the usual receiver location lookup.
            middle = self.ngll // 2
            midpoints = np.column_stack((self.mesh_s[:,middle,middle],
                                         self.mesh_z[:,middle,middle]))
            self.kdtree = KDTree(data=midpoints)

            # Keep each face's point order and its location in the selected mesh.
            self.face_phase = boundary['face_phase'][:]
            self.face_index = boundary['face_index'][:]
            self.point_in_face = boundary['point_in_face'][:]
            self.point_element_slot = boundary['point_element_slot'][:]
            self.point_mesh_index = boundary['point_mesh_index'][:]
            self.point_xi = boundary['xi'][:]
            self.point_eta = boundary['eta'][:]
            self.point_longitude = boundary['longitude'][:]
            self.point_latitude = boundary['latitude'][:]
            self.point_depth_km = boundary['depth_km'][:]
            self.point_source_phi = boundary['source_phi'][:]
            self.solid_element_to_mesh = (boundary['solid_element_to_mesh'][:]
                                          if 'solid_element_to_mesh' in boundary
                                          else np.empty(0,dtype=int))
            self.fluid_element_to_mesh = (boundary['fluid_element_to_mesh'][:]
                                          if 'fluid_element_to_mesh' in boundary
                                          else np.empty(0,dtype=int))

        # Map mesh indices to the compact solid and fluid wavefield arrays.
        self._solid_mesh_to_slot = {int(mesh):slot
                                    for slot,mesh in enumerate(self.solid_element_to_mesh)}
        self._fluid_mesh_to_slot = {int(mesh):slot
                                    for slot,mesh in enumerate(self.fluid_element_to_mesh)}
        self.iodict = {}
        self._nc_handles = {}
        self._field_cache = {}
        self._is_dof_file = False
    
    def __copy__(self):
        """
        shallow copy of necessary basic variables
        """
        if getattr(self, '_boundary_mode', False):
            # Share fixed mesh metadata, then let the copy open its own field files.
            db = AxiBasicDB()
            db.__dict__.update(self.__dict__)
            db.iodict = {}
            db._nc_handles = {}
            db._field_cache = {}
            return db

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
        db.rot_mat = self.rot_s * 1.

        # read mesh 
        db.mesh_s = self.mesh_s.copy()
        db.mesh_z = self.mesh_z.copy()

        # create kdtree
        db.kdtree = self.kdtree.copy()

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
        if getattr(self, '_boundary_mode', False):
            self._set_boundary_iodata(Path(ncfile_dir))
            return

        import os 
        for stype in ['MZZ',"MXX_P_MYY","MXZ_MYZ","MXY_MXX_M_MYY","PZ","PX","PY"]:
            dirname = ncfile_dir + '/' + stype

            # check if all memmap files exist
            for field in ['disp_s','disp_z','disp_p']:
                filepath = dirname + '/Data/' + field + '.bin'
                if not os.path.exists(filepath):
                    break

                # check size of file
                size_of_float32 = os.path.getsize(filepath) // 4
                if size_of_float32 != self.nspec * self.ngll * self.ngll * self.nt:
                    self._is_dof_file = True
                    shape = (self.nglob,self.nt)
                else:
                    shape = (self.nspec,self.ngll,self.ngll,self.nt)
                    self._is_dof_file = False
                self.iodict[stype + '/' + field] = np.memmap(filepath,dtype=np.float32,mode='r',shape=shape,order='C')

        # check if iodit is empty
        if len(self.iodict) == 0 :
            print(f"no data has been accessed, please check {ncfile_dir}!")

    def _set_boundary_iodata(self,run_dir:Path) -> None:
        """Open element-major binaries, or the NetCDF arrays before transposition."""
        fields = ('disp_s', 'disp_z', 'disp_p', 'chi')
        for stype in ('MZZ', 'MXX_P_MYY', 'MXZ_MYZ', 'MXY_MXX_M_MYY', 'PZ', 'PX', 'PY'):
            data_dir = run_dir / stype / 'Data'
            ncfile = data_dir / 'boundary_wavefields.nc4'
            if not ncfile.is_file():
                continue

            # All source components must describe the same ordered face points.
            handle = h5py.File(ncfile, 'r')
            if not (np.array_equal(handle['face_index'][:], self.face_index)
                    and np.array_equal(handle['face_phase'][:], self.face_phase)
                    and np.array_equal(handle['point_mesh_index'][:], self.point_mesh_index)
                    and np.array_equal(handle['point_element_slot'][:], self.point_element_slot)
                    and np.array_equal(handle['xi'][:], self.point_xi)
                    and np.array_equal(handle['eta'][:], self.point_eta)
                    and np.array_equal(handle['source_phi'][:], self.point_source_phi)
                    and np.array_equal(handle['sample_time'][:], self.sample_time)):
                handle.close()
                raise ValueError(f'Boundary point mapping differs in {ncfile}')

            # Prefer transposed binaries and use NetCDF arrays when binaries are absent.
            needs_handle = False
            for field in fields:
                if field == 'chi':
                    nelem = len(self.fluid_element_to_mesh)
                else:
                    nelem = len(self.solid_element_to_mesh)
                if nelem == 0:
                    continue

                shape = (nelem, self.ngll, self.ngll, self.nt)
                binary = data_dir / f'{field}.bin'
                key = f'{stype}/{field}'
                if binary.is_file():
                    expected = int(np.prod(shape)) * np.dtype('f4').itemsize
                    if binary.stat().st_size != expected:
                        handle.close()
                        raise ValueError(f'{binary} has {binary.stat().st_size} bytes; '
                                         f'expected {expected}')
                    self.iodict[key] = np.memmap(binary, dtype='f4', mode='r', shape=shape)
                elif field in handle:
                    self.iodict[key] = handle[field]
                    needs_handle = True

            # Retain a NetCDF handle only while its datasets are in use.
            if needs_handle:
                self._nc_handles[stype] = handle
            else:
                handle.close()

        if not self.iodict:
            raise FileNotFoundError(f'No boundary wavefields found in {run_dir}')
    
    def copy(self):
        return self.__copy__()
    
    def close(self):
        # Release both mapped binaries and any NetCDF datasets kept open.
        for val in self.iodict.values():
            if isinstance(val, np.memmap):
                val._mmap.close()
        for handle in getattr(self, '_nc_handles', {}).values():
            handle.close()
        self._nc_handles = {}
        self.iodict = {}
        self._field_cache = {}
    
    def set_source(self,evla:float,evlo:float):
        """
        update source info in the database

        Parameters
        ============================================================
        evla: float
            latitude of station, in deg
        evlo: float
            longitude of station, in deg 
        """
        self.evla = evla
        self.evlo = evlo 
        self.rot_s = rotation_matrix(np.pi/2-np.deg2rad(evla),np.deg2rad(evlo))
        pass

    def read_cmt(self,cmtfile:str):
        from .utils import read_cmtsolution
        mzz,mxx,myy,mxz,myz,mxy = read_cmtsolution(cmtfile)
        mzz,mxx,myy,mxz,myz,mxy = map(lambda x: x / self.mag,[mzz,mxx,myy,mxz,myz,mxy])

        return mzz,mxx,myy,mxz,myz,mxy
    
    def _locate_elem(self,s,z,is_el_point = True):
        """
        locate element for one station

        Parameters
        ============================================================
        s: float
            s coordinate
        z: float
            z coordinate
        is_el_point: bool, optional
            whether the point is an element point (default is True)

        Returns
        ============================================================
        id_elem: int or None
            element id if found, otherwise None
        xi: float
            local coordinate xi
        eta: float
            local coordinate eta
        """
        from .sem_funcs import inside_element
        id_elem = None 

        # get nearest 10 points 
        points = np.atleast_1d(self.kdtree.query([s,z],k=min(10,self.nspec))[1])
        for tol in [1e-3, 1e-2, 5e-2, 8e-2]:
            for idx in points:
                skel = self._element_skeleton(idx)
                eltype = self.eltype[idx]

                isin,xi,eta = inside_element(s,z,skel,eltype,tolerance=tol)
                if isin:
                    id_elem = idx 
                    break; 

            if id_elem is not None:
                break 
        
        return id_elem,xi,eta

    def _element_skeleton(self, elemid:int) -> np.ndarray:
        if getattr(self, '_boundary_mode', False):
            # Boundary meshes store nodal coordinates, so derive the four corners.
            s = self.mesh_s[elemid]
            z = self.mesh_z[elemid]
            corners = ((0,0), (0,-1), (-1,-1), (-1,0))
            return np.array([(s[i,j], z[i,j]) for i,j in corners], dtype=float)

        ctrl_id = self.skelid[elemid]
        return np.column_stack((self.mesh_s[ctrl_id], self.mesh_z[ctrl_id]))
    
    def _get_field_elem(self,stype:str,fieldkey:str,elemid:int) -> np.ndarray:
        """
        get field data for one element from file

        Parameters
        ============================================================
        stype: str
            source type
        fieldkey: str
            field key in the binary file
        elemid: int
            element id

        Returns
        ============================================================
        field_data: np.ndarray
            field data for the specified element
        """

        # check 
        key = stype + '/' + fieldkey
        if key not in self.iodict:
            if (getattr(self, '_boundary_mode', False)
                    and (fieldkey != 'disp_p'
                         or self._get_excitation_type(stype) != 'monopole')):
                raise FileNotFoundError(f'Missing boundary field {key}')
            return np.zeros((self.ngll,self.ngll,self.nt),dtype='f4')

        fio = self.iodict[key]
        if getattr(self, '_boundary_mode', False):
            # Translate a mesh element index into its phase-specific storage slot.
            slots = self._fluid_mesh_to_slot if fieldkey == 'chi' else self._solid_mesh_to_slot
            if elemid not in slots:
                raise ValueError(f'Element {elemid} has no {fieldkey} wavefield')
            slot = slots[elemid]
            if isinstance(fio, h5py.Dataset):
                return np.moveaxis(fio[:,slot,:,:], 0, -1).copy()
            return fio[slot].copy()

        if self._is_dof_file:
            # check if (elemid,fieldkey) is in cache
            cache_key = (key, elemid)
            if cache_key in self._field_cache:
                field_data = self._field_cache[cache_key].copy()
            else:
                # read dataset for dof file
                idx = self.ibool[elemid,:,:]
                var = np.zeros((self.ngll,self.ngll,self.nt),dtype='f4')
                for i in range(self.ngll):
                    for j in range(self.ngll):
                        gll_id = idx[i,j]
                        var[i,j,:] = fio[gll_id,:]
                field_data = var
                self._field_cache[cache_key] = field_data.copy()
        else:
            field_data = fio[elemid,...].copy()

        return field_data

    
    def _get_displ(self,elemid,xi,eta,stype):
        """
        Get displacement for one station

        Parameters
        ============================================================

        stel: float 
            elevation, in m
        theta: float 
            epicenter distance, in rad
        ncfile: str
            ncfile which the displ is stored in 

        Returns
        -----------------------------------------------------------
        us,up,uz: np.ndarray
            s,p,z components 
            
        """
        if getattr(self, '_boundary_mode', False) and not self.is_elastic[elemid]:
            return self._get_element_displ_fluid(elemid,xi,eta,stype)

        from .sem_funcs import lagrange_interpol_2D_td

        # Field arrays are stored as (eta, xi, time), with time contiguous.
        sgll = self.gll
        zgll = self.gll
        flag = self.axis[elemid] == 1
        if flag:
            sgll = self.glj
        us = lagrange_interpol_2D_td(
            sgll,zgll,self._get_field_elem(stype,'disp_s',elemid),xi,eta)
        up = lagrange_interpol_2D_td(
            sgll,zgll,self._get_field_elem(stype,'disp_p',elemid),xi,eta)
        uz = lagrange_interpol_2D_td(
            sgll,zgll,self._get_field_elem(stype,'disp_z',elemid),xi,eta)
        
        return us,up,uz

    def _get_chi(self,elem,xi,eta,stype):
        from .sem_funcs import lagrange_interpol_2D_td

        values = self._get_field_elem(stype,'chi',elem)
        nodes_xi = self.glj if self.axis[elem] == 1 else self.gll
        return lagrange_interpol_2D_td(nodes_xi,self.gll,values,xi,eta)

    def _get_element_displ_fluid(self,elem,xi,eta,stype):
        """Derive modal fluid displacement from chi and the selected mesh."""
        from .sem_funcs import strain_td,lagrange_interpol_2D_td

        # Arrange chi as the s component of the C-order strain input.
        values = self._get_field_elem(stype,'chi',elem)
        utemp = np.zeros((3,self.ngll,self.ngll,self.nt),dtype=float)
        utemp[0,...] = values

        is_axi = self.axis[elem] == 1
        sgll = self.glj if is_axi else self.gll
        GT = self.G1T if is_axi else self.G2T

        # With chi in the s slot, monopole strain contains dchi/ds,
        # chi/s, and dchi/dz in components 0, 1, and 4, respectively.
        strain = strain_td(utemp,self.G2,GT,sgll,self.gll,self.ngll-1,self.nt,
                             self._element_skeleton(elem),self.eltype[elem],is_axi,'monopole')

        # Interpolate the gradient and convert it to displacement using local density.
        rho = lagrange_interpol_2D_td(
            sgll,self.gll,self.xrho[elem,:,:,None],xi,eta)[0]
        if rho <= 0:
            raise ValueError(f'Invalid density in fluid element {elem}')

        us = lagrange_interpol_2D_td(sgll,self.gll,strain[0],xi,eta)/rho
        uz = 2*lagrange_interpol_2D_td(sgll,self.gll,strain[4],xi,eta)/rho
        order = {'monopole':0,'dipole':1,'quadpole':2}[self._get_excitation_type(stype)]

        # Only nonzero azimuthal orders contribute an azimuthal component.
        if order == 0:
            up = np.zeros_like(us)
        else:
            up = order*lagrange_interpol_2D_td(sgll,self.gll,strain[1],xi,eta)/rho

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

        Parameters
        ===================================================
        elemid: current 
        xi/eta: local coordinates 
        stype: source type

        Returns
        ====================================================
        strain : np.ndarray
                shape(6,nt), ess,epp,ezz,epz,esz,esp
        """
        from .sem_funcs import lagrange_interpol_2D_td,strain_td
        nt = self.nt
        ngll = self.ngll

        # allocate space
        eps = np.zeros((6,nt))

        # cache element
        utemp = np.zeros((3,ngll,ngll,nt),dtype=float)
        
        # dataset
        # read dataset
        utemp[0,...] = self._get_field_elem(stype,'disp_s',elemid)
        utemp[2,...] = self._get_field_elem(stype,'disp_z',elemid)
        utemp[1,...] = self._get_field_elem(stype,'disp_p',elemid)

        # gll/glj array
        sgll = self.gll
        zgll = self.gll
        is_axi = self.axis[elemid] == 1
        if is_axi:
            sgll = self.glj

        # control points
        skel = self._element_skeleton(elemid)
        eltype = self.eltype[elemid]
        if is_axi:
            G = self.G2 
            GT = self.G1T 
        else:
            G = self.G2 
            GT = self.G2T 

        # Compute strain as (component, eta, xi, time).
        etype = self._get_excitation_type(stype)
        strain = strain_td(utemp,G,GT,sgll,zgll,ngll-1,nt,
                              skel,eltype,is_axi,etype)

        # interpolate 
        # es shape(6,nt)
        for j in range(6):
            eps[j,:] = lagrange_interpol_2D_td(sgll,zgll,strain[j],xi,eta)
        
        return eps

    def _get_stress(self,elemid,xi,eta,stype):
        """
        get stress field for a given point from file

        Parameters
        ===================================================
        elemid: int
            current element id
        xi: float
            local coordinate xi
        eta: float
            local coordinate eta
        stype: str
            source type

        Returns
        ===================================================
        stress : np.ndarray
                shape(6,nt), ess,epp,ezz,epz,esz,esp
        """
        from .sem_funcs import lagrange_interpol_2D_td,strain_td,find_theta
        from .utils import c_ijkl_ani
        nt = self.nt 
        ngll = self.ngll
    
        # cache element
        utemp = np.zeros((3,ngll,ngll,nt),dtype=float)
        
        # dataset
        utemp[0,...] = self._get_field_elem(stype,'disp_s',elemid)
        utemp[2,...] = self._get_field_elem(stype,'disp_z',elemid)
        utemp[1,...] = self._get_field_elem(stype,'disp_p',elemid)

        # Material arrays already have the C-order (eta, xi) mesh layout.
        xmu = self.xmu[elemid]
        xlam = self.xlamda[elemid]
        xxi = self.xxi[elemid]
        xphi = self.xphi[elemid]
        xeta = self.xeta[elemid]

        # gll/glj array
        sgll = self.gll
        zgll = self.gll
        is_axi = self.axis[elemid] == 1
        if is_axi:
            sgll = self.glj

        # control points
        skel = self._element_skeleton(elemid)
        eltype = self.eltype[elemid]

        if self.axis[elemid]:
            G = self.G2 
            GT = self.G1T 
        else:
            G = self.G2 
            GT = self.G2T 

        # Compute strain as (component, eta, xi, time).
        etype = self._get_excitation_type(stype)
        e = strain_td(utemp,G,GT,sgll,zgll,ngll-1,self.nt,
                        skel,eltype,self.axis[elemid]==1,etype)
        
        theta = find_theta(sgll,zgll,skel,eltype)

        # get elastic tensor c21
        c11 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 1, 1)[:,:,None]
        c12 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 2, 2)[:,:,None]
        c13 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 3, 3)[:,:,None]
        c15 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 3, 1)[:,:,None]
        c22 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 2, 2, 2)[:,:,None]
        c23 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 2, 3, 3)[:,:,None]
        c25 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 2, 3, 1)[:,:,None]
        c33 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 3, 3, 3, 3)[:,:,None]
        c35 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 3, 3, 3, 1)[:,:,None]
        c44 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 3, 2, 3)[:,:,None]
        c46 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 3, 1, 2)[:,:,None]
        c55 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 3, 1, 3, 1)[:,:,None]
        c66 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 2, 1, 2)[:,:,None]
        c14 = 0.; c26 = 0.; c36 = 0.; c24 = 0.
        c16 = 0.; c45 = 0.; c56 = 0.; c34 = 0.

        # compute stress
        stress = e * 0.
        e[3:6] = 2. * e[3:6]

        # Compute stress components explicitly using Voigt notation
        stress[0] = c11 * e[0] + c16 * e[5] + c12 * e[1] + c15 * e[4] + c14 * e[3] + c13 * e[2]  # sxx → s[..., 0]
        stress[1] = c12 * e[0] + c26 * e[5] + c22 * e[1] + c25 * e[4] + c24 * e[3] + c23 * e[2]  # syy → s[..., 1]
        stress[2] = c13 * e[0] + c36 * e[5] + c23 * e[1] + c35 * e[4] + c34 * e[3] + c33 * e[2]  # szz → s[..., 2]
        stress[3] = c14 * e[0] + c46 * e[5] + c24 * e[1] + c45 * e[4] + c44 * e[3] + c34 * e[2]  # syz → s[..., 3]
        stress[4] = c15 * e[0] + c56 * e[5] + c25 * e[1] + c55 * e[4] + c45 * e[3] + c35 * e[2]  # sxz → s[..., 4]
        stress[5] = c16 * e[0] + c66 * e[5] + c26 * e[1] + c56 * e[4] + c46 * e[3] + c36 * e[2]  # sxy → s[..., 5]

        # interpolate 
        # es shape(6,nt)
        sigma = np.zeros((6,nt))
        for j in range(6):
            sigma[j,:] = lagrange_interpol_2D_td(sgll,zgll,stress[j],xi,eta)
        
        return sigma
    
    def _get_eq_moment(self,elemid,xi,eta,norm_spz,stype):
        """
        get equivalent moment at a given point from file

        Parameters
        ===================================================
        elemid: int
            current element id
        xi: float
            local coordinate xi
        eta: float
            local coordinate eta
        norm_spz: np.ndarray
            normal vector at the station, shape(3,)
        stype: str
            source type

        Returns
        ===================================================
        sigma : np.ndarray
                shape(6,nt), ess,epp,ezz,epz,esz,esp
        """
        # get stress
        from .sem_funcs import lagrange_interpol_2D_td,find_theta
        from .utils import c_ijkl_ani
        nt = self.nt 
        ngll = self.ngll
    
        # cache element
        utemp = np.zeros((3,ngll,ngll,nt),dtype=float)
        
        # dataset
        utemp[0,...] = self._get_field_elem(stype,'disp_s',elemid)
        utemp[2,...] = self._get_field_elem(stype,'disp_z',elemid)
        utemp[1,...] = self._get_field_elem(stype,'disp_p',elemid)

        # Construct the equivalent strain with time last in C storage.
        e = np.zeros((6,ngll,ngll,nt),dtype=float)
        e[0] = norm_spz[0] * utemp[0]
        e[1] = norm_spz[1] * utemp[1]
        e[2] = norm_spz[2] * utemp[2]
        e[3] = norm_spz[1] * utemp[2] + norm_spz[2] * utemp[1]
        e[4] = norm_spz[0] * utemp[2] + norm_spz[2] * utemp[0]
        e[5] = norm_spz[0] * utemp[1] + norm_spz[1] * utemp[0]

        xmu = self.xmu[elemid]
        xlam = self.xlamda[elemid]
        xxi = self.xxi[elemid]
        xphi = self.xphi[elemid]
        xeta = self.xeta[elemid]

        # gll/glj array
        sgll = self.gll
        zgll = self.gll
        is_axi = self.axis[elemid] == 1
        if is_axi:
            sgll = self.glj

        # control points
        skel = self._element_skeleton(elemid)
        eltype = self.eltype[elemid]

        # find theta
        theta = find_theta(sgll,zgll,skel,eltype)

        # get elastic tensor c21
        c11 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 1, 1)[:,:,None]
        c12 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 2, 2)[:,:,None]
        c13 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 3, 3)[:,:,None]
        c15 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 1, 3, 1)[:,:,None]
        c22 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 2, 2, 2)[:,:,None]
        c23 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 2, 3, 3)[:,:,None]
        c25 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 2, 3, 1)[:,:,None]
        c33 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 3, 3, 3, 3)[:,:,None]
        c35 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 3, 3, 3, 1)[:,:,None]
        c44 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 3, 2, 3)[:,:,None]
        c46 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 2, 3, 1, 2)[:,:,None]
        c55 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 3, 1, 3, 1)[:,:,None]
        c66 = c_ijkl_ani(xlam,xmu,xxi,xphi,xeta,theta, 0., 1, 2, 1, 2)[:,:,None]
        c14 = 0.; c26 = 0.; c36 = 0.; c24 = 0.
        c16 = 0.; c45 = 0.; c56 = 0.; c34 = 0.

        # Compute stress components explicitly using Voigt notation
        stress = e * 0.
        stress[0] = c11 * e[0] + c16 * e[5] + c12 * e[1] + c15 * e[4] + c14 * e[3] + c13 * e[2]  # sxx → s[..., 0]
        stress[1] = c12 * e[0] + c26 * e[5] + c22 * e[1] + c25 * e[4] + c24 * e[3] + c23 * e[2]  # syy → s[..., 1]
        stress[2] = c13 * e[0] + c36 * e[5] + c23 * e[1] + c35 * e[4] + c34 * e[3] + c33 * e[2]  # szz → s[..., 2]
        stress[3] = c14 * e[0] + c46 * e[5] + c24 * e[1] + c45 * e[4] + c44 * e[3] + c34 * e[2]  # syz → s[..., 3]
        stress[4] = c15 * e[0] + c56 * e[5] + c25 * e[1] + c55 * e[4] + c45 * e[3] + c35 * e[2]  # sxz → s[..., 4]
        stress[5] = c16 * e[0] + c66 * e[5] + c26 * e[1] + c56 * e[4] + c46 * e[3] + c36 * e[2]  # sxy → s[..., 5]

        # interpolate 
        # es shape(6,nt)
        sigma = np.zeros((6,nt))
        for j in range(6):
            sigma[j,:] = lagrange_interpol_2D_td(sgll,zgll,stress[j],xi,eta)
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
        x1,y1,z1 = np.dot(self.rot_s.T,np.array([x,y,z]))

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
            Rr = Rr.T @ self.rot_s @ R1
        elif comp == 'spz':
            Rr = np.eye(3)
        else:
            Rr = self.rot_s @ R1 

        u1 = Rr[0,0] * us + Rr[0,1] * up + Rr[0,2] * uz 
        u2 = Rr[1,0] * us + Rr[1,1] * up + Rr[1,2] * uz 
        u3 = Rr[2,0] * us + Rr[2,1] * up + Rr[2,2] * uz 
        
        if comp == 'enz':
            u1 = -u1
            temp = u1.copy()
            u1 = u2.copy()
            u2 = temp.copy()

        return u1,u2,u3

    def _surface_points(self, iface:int) -> tuple[int, np.ndarray]:
        # Select the requested zero-based face in its original 25-point order.
        if not getattr(self, '_boundary_mode', False):
            raise RuntimeError('read_basic must load a boundary_wavefields.nc4 file first')
        if not isinstance(iface, (int, np.integer)) or not 0 <= iface < len(self.face_phase):
            raise IndexError(f'iface must be between 0 and {len(self.face_phase)-1}')

        points = np.flatnonzero(self.face_index == iface)
        points = points[np.argsort(self.point_in_face[points])]
        if len(points) != 25 or not np.array_equal(self.point_in_face[points], np.arange(1,26)):
            raise ValueError(f'Face {iface} does not have 25 ordered points')

        # Check that compact storage slots agree with the saved mesh indices.
        phase = int(self.face_phase[iface])
        if phase not in (0, 1):
            raise ValueError(f'Unknown phase {phase} for face {iface}')
        mapping = self.solid_element_to_mesh if phase == 0 else self.fluid_element_to_mesh
        if (np.any(self.point_element_slot[points] < 0)
                or np.any(self.point_element_slot[points] >= len(mapping))):
            raise ValueError(f'Invalid element slot on face {iface}')
        if not np.array_equal(mapping[self.point_element_slot[points]],
                              self.point_mesh_index[points]):
            raise ValueError(f'Element slot and mesh index disagree on face {iface}')

        return phase, points

    def _surface_source_terms(self, source, is_moment:bool, phi:float):
        """Use the same modal source coefficients as syn_seismo/syn_stress."""
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)

        if is_moment:
            mzz,mxx,myy,mxz,myz,mxy = source
            cos2phi = np.cos(2*phi)
            sin2phi = np.sin(2*phi)
            return (
                ('MZZ',mzz,0.),
                ('MXX_P_MYY',mxx+myy,0.),
                ('MXZ_MYZ',mxz*cosphi+myz*sinphi,
                 myz*cosphi-mxz*sinphi),
                ('MXY_MXX_M_MYY',(mxx-myy)*cos2phi+2*mxy*sin2phi,
                 -(mxx-myy)*sin2phi+2*mxy*cos2phi),
            )

        fx,fy,fz = source
        return (
            ('PZ',fz,0.),
            ('PX',fx*cosphi+fy*sinphi,-fx*sinphi+fy*cosphi),
        )

    def syn_surface_wavefield(self, iface:int, comp:str='enz', cmtfile=None,
                              forcevec=None, print_output:bool=True):
        """Print and return a face's displacement (solid) or chi (fluid).

        Face indices are 0-based. Displacement has shape (25,3,nt) and chi
        has shape (25,nt). Components may be 'enz', 'xyz', or 'spz'. Set
        print_output=False when collecting many faces without console output.
        """
        comp = comp.lower()
        if comp not in ('enz','xyz','spz'):
            raise ValueError("comp must be 'enz', 'xyz', or 'spz'")

        phase, points = self._surface_points(iface)
        if (cmtfile is None) == (forcevec is None):
            raise ValueError('Specify exactly one of cmtfile or forcevec')

        is_moment = cmtfile is not None
        source = self.read_cmt(cmtfile) if is_moment else forcevec
        wavefield = np.zeros((25,self.nt) if phase == 1 else (25,3,self.nt), dtype=float)

        # Combine source modes at each point, then rotate solid displacement.
        for row,point in enumerate(points):
            elem = int(self.point_mesh_index[point])
            xi,eta = self.point_xi[point],self.point_eta[point]
            latitude,longitude = self.point_latitude[point],self.point_longitude[point]
            _,phi = self.compute_tp_recv(latitude,longitude)
            terms = self._surface_source_terms(source,is_moment,phi)

            # Fluid faces expose chi directly; solid faces expose displacement.
            if phase == 1:
                for stype,a,_ in terms:
                    wavefield[row] += a*self._get_chi(elem,xi,eta,stype)
                continue

            us = np.zeros(self.nt)
            up = np.zeros(self.nt)
            uz = np.zeros(self.nt)
            for stype,a,b in terms:
                us1,up1,uz1 = self._get_displ(elem,xi,eta,stype)
                us += a*us1
                up += b*up1
                uz += a*uz1

            R1 = np.array([[np.cos(phi),-np.sin(phi),0.],
                           [np.sin(phi), np.cos(phi),0.],
                           [0.,0.,1.]])

            if comp == 'spz':
                R = np.eye(3)
            elif comp == 'xyz':
                R = self.rot_s @ R1
            else:
                Rlocal = rotation_matrix(np.deg2rad(90-latitude),
                                         np.deg2rad(longitude))
                R = Rlocal.T @ self.rot_s @ R1

            result = R @ np.stack((us,up,uz))
            if comp == 'enz':
                wavefield[row] = np.stack((result[1],-result[0],result[2]))
            else:
                wavefield[row] = result

        if print_output:
            print(wavefield)
        return wavefield

    def syn_surface_derived_fields(self, iface:int, cmtfile=None, forcevec=None,
                                   print_output:bool=True):
        """Print and return XYZ stress (solid) or XYZ displacement (fluid)."""
        phase, points = self._surface_points(iface)
        if (cmtfile is None) == (forcevec is None):
            raise ValueError('Specify exactly one of cmtfile or forcevec')

        is_moment = cmtfile is not None
        source = self.read_cmt(cmtfile) if is_moment else forcevec
        derived = np.zeros((25,6,self.nt) if phase == 0 else (25,3,self.nt),dtype=float)

        # Derive stress in solids and displacement from chi in fluids.
        for row,point in enumerate(points):
            elem = int(self.point_mesh_index[point])
            xi,eta = self.point_xi[point],self.point_eta[point]
            latitude,longitude = self.point_latitude[point],self.point_longitude[point]
            _,phi = self.compute_tp_recv(latitude,longitude)
            terms = self._surface_source_terms(source,is_moment,phi)

            R1 = np.array([[np.cos(phi),-np.sin(phi),0.],
                           [np.sin(phi), np.cos(phi),0.],
                           [0.,0.,1.]])

            # Use the same modal weights as the ordinary stress and seismogram paths.
            if phase == 0:
                sigma = np.zeros((6,self.nt))
                for stype,a,b in terms:
                    stress = self._get_stress(elem,xi,eta,stype)
                    sigma[[0,1,2,4]] += a*stress[[0,1,2,4]]
                    sigma[[3,5]] += b*stress[[3,5]]
                derived[row] = rotate_tensor2(sigma,self.rot_s @ R1)
            else:
                us = np.zeros(self.nt)
                up = np.zeros(self.nt)
                uz = np.zeros(self.nt)
                for stype,a,b in terms:
                    us1,up1,uz1 = self._get_element_displ_fluid(elem,xi,eta,stype)
                    us += a*us1
                    up += b*up1
                    uz += a*uz1
                derived[row] = (self.rot_s @ R1) @ np.stack((us,up,uz))

        if print_output:
            print(derived)
        return derived

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
        R = self.rot_s @ R1
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
        R = self.rot_s @ R1
        sigma_xyz = rotate_tensor2(sigma,R)

        return sigma_xyz
    
    def syn_eq_moment(self,stla,stlo,stel,norm_xyz,cmtfile=None,forcevec=None):
        """
        compute equivalent moment at a given station

        Parameters
        ============================================================
        stla: float
            latitude of station, in deg
        stlo: float
            longitude of station, in deg
        stel: float
            elevation of station, in m
        norm_xyz: np.ndarray
            normal vector (along xyz) at the station, shape(3,)
        cmtfile: str
            cmt solution file
        forcevec: tuple or list
            force vector components (fx, fy, fz)
        
        Returns
        ============================================================
        m_eq: np.ndarray
            equivalent moment at the station, shape(3,3,nt)
        """
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

        # rotation matrix from spz to xyz
        R1 = np.eye(3) # rotate from (spz to xs,ys,zs)
        R1[0,:2] = [np.cos(phi),-np.sin(phi)]
        R1[1,:2] = [np.sin(phi),np.cos(phi)]
        R = self.rot_s @ R1
        norm_spz = R.T @ norm_xyz.reshape((3,1))
        
        # locate element
        elemid,xi,eta = self._locate_elem(sr,zr)

        # loop every source type
        for stype in srctypes:
            #print("synthetic strain tensor for  ... %s" %(stype))
            # get basic waveform
            eps0 = self._get_eq_moment(elemid,xi,eta,norm_spz,stype)

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

        # rotate to xyz from spz
        moment = rotate_tensor2(sigma,R)

        return moment
