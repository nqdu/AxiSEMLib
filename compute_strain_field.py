from database import AxiBasicDB
from mpi4py import MPI
import h5py 
import numpy as np
from tqdm import tqdm
from numba import jit 

@jit(nopython=True)
def lagrange_poly(xi,xctrl):
    nctrl = len(xctrl)
    h = np.array([0.0 for i in range(nctrl)])

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
    

    return h

@jit(nopython=True)
def compute_resample_coefs(xgll,zgll,x,nsamp):
    ngll = len(xgll)

    # allocate space
    cfdump = np.zeros((nsamp,nsamp,ngll,ngll),dtype=np.float32)
    cfload = np.zeros((ngll,ngll,nsamp,nsamp),dtype=np.float32)

    for iz in range(nsamp):
        polyz = lagrange_poly(x[iz],zgll)
        for ix in range(nsamp):
            polyx = lagrange_poly(x[ix],xgll)
            for iz1 in range(ngll):
                for ix1 in range(ngll):
                    cfdump[iz,ix,iz1,ix1] =  polyx[ix1] * polyz[iz1]

    for iz1 in range(ngll):
        polyz = lagrange_poly(zgll[iz1],x)
        for ix1 in range(ngll):
            polyx = lagrange_poly(xgll[ix1],x)
            for iz in range(nsamp):
                for ix in range(nsamp):
                    cfload[iz1,ix1,iz,ix] =  polyx[ix] * polyz[iz]
    
    return cfdump,cfload

def compute_strain(db:AxiBasicDB,ncfile,cf,cf_axi):
    """
    compute strain field from displacement

    Returns:
    """
    from sem_funcs import strain_td
    nt = db.nt 
    ngll = db.ngll

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # get source type
    stype = db.fstream.attrs['excitation type'].decode("utf-8")
    fio = h5py.File(ncfile,"r+",driver='mpio',comm=MPI.COMM_WORLD)
    nspec = db.nspec

    # create dataset
    if 'strain' in fio.keys():
        del fio['strain']
    nsamp = cf.shape[0]
    fio.create_dataset('strain',(nspec,6,nsamp,nsamp,nt),dtype=np.float32)

    # loop each element
    for elemid in tqdm(range(rank,nspec,nprocs)):
    #for elemid in range(1):
        # cache element
        utemp = np.zeros((nt,ngll,ngll,3),dtype=float,order='F')

        # connectivity 
        ibool = db.ibool[elemid,:,:]
        
        # dataset
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
        sgll = db.gll
        zgll = db.gll 
        if db.axis[elemid]:
            sgll = db.glj 

        # control points
        skel = np.zeros((4,2))
        ctrl_id = db.skelid[elemid,:4]
        eltype = db.eltype[elemid]
        for i in range(4):
            skel[i,0] = db.s[ctrl_id[i]]
            skel[i,1] = db.z[ctrl_id[i]]

        if db.axis[elemid]:
            G = db.G2 
            GT = db.G1T 
        else:
            G = db.G2 
            GT = db.G2T 

        # compute strain shape(nt,npol+1,npol+1,6)
        temp = strain_td(utemp,G,GT,sgll,zgll,ngll-1,db.nt,
                            skel,eltype,db.axis[elemid]==1,stype)
        temp = np.float32(np.reshape(temp.flatten(),(6,ngll,ngll,nt),order='C'))
        if db.axis[elemid]:
            strain = np.einsum('kijl,pqij->kpql',temp,cf_axi)
        else:
            strain = np.einsum('kijl,pqij->kpql',temp,cf)
        fio['strain'][elemid,:,:,:] = strain
    comm.Barrier()

    # delete useless dataset
    del fio['disp_s']
    del fio['disp_z']
    if 'disp_p' in fio.keys():
        del fio['disp_p']
    
    fio.close()


# read database
datadir0 ='/home/nqdu/scratch/axisem/SOLVER/./prem_10s_0/'
db = AxiBasicDB(datadir0 + "/PZ/Data/axisem_output.nc4")

# resample matrix
ngll = db.ngll
x = np.array([-1.,0.,1.])
nsamp = len(x)
xgll = db.gll 
zgll = db.gll

cfdump,cfload = compute_resample_coefs(xgll,zgll,x,nsamp)
cfdump_axi,cfload_axi = compute_resample_coefs(xgll,db.glj,x,nsamp)
compute_strain(db,datadir0 + "/PZ/Data/axisem_fields.h5",cfdump,cfdump_axi)
db.close()
MPI.COMM_WORLD.Barrier()

# save cfdump
if MPI.COMM_WORLD.Get_rank() == 0:
    f = h5py.File(datadir0 + "/PZ/Data/axisem_output.nc4",'r+')
    if 'cfdump' in f.keys(): del f['cfdump']
    f.create_dataset('cfdump',cfdump.shape,dtype=np.float32)
    f['cfdump'][:] = cfdump 

    if 'cfdump_axi' in f.keys(): del f['cfdump_axi']
    f.create_dataset('cfdump_axi',cfdump.shape,dtype=np.float32)
    f['cfdump_axi'][:] = cfdump_axi 

    if 'cfload' in f.keys(): del f['cfload']
    f.create_dataset('cfload',cfload.shape,dtype=np.float32)
    f['cfload'][:] = cfload 

    if 'cfload_axi' in f.keys(): del f['cfload_axi']
    f.create_dataset('cfload_axi',cfload.shape,dtype=np.float32)
    f['cfload_axi'][:] = cfload_axi 
    f.close() 
