import numpy as np 
import h5py 
from tqdm import tqdm 
import sys 
import os 

def write_trans_data_dof(file_r:h5py.File,dsetstr:str,out_dir:str):
    dset1 = file_r['Snapshots/' + dsetstr]
    nt,npts = dset1.shape

    # read ibool 
    ngll = len(file_r['Mesh/npol'])
    nspec = len(file_r['Mesh/elements'])
    ibool = file_r['Mesh/sem_mesh'][:]
    idx = np.arange(0,ngll,1)
    ngll_out = len(idx)

    # create an memmap file
    dset2 = np.memmap(out_dir + '/' + dsetstr + '.bin',dtype='f4',mode='w+',shape=(npts,nt))

    # allocate buffer to reduce io time
    sizeGB = 2
    npts_one = int((sizeGB * 1024**3) / (nt*4))
    for i in tqdm(range(0,npts,npts_one)):
        ntasks = 0
        if i + npts_one <=npts:
            ntasks = npts_one 
        else:
            ntasks = npts - i 

        istart = i
        iend = i + ntasks
        n = iend - istart
        if n<=0 : n = 0

        # alloc space
        mydata = np.zeros((nt,n),dtype='f4')

        # read data from dset1 
        mydata = dset1[:,istart:istart+n] * np.float32(1.)
        mydata = np.transpose(mydata)

        # write to dset2
        dset2[istart:istart+n,:] = mydata 

        # flush to disk
        dset2.flush()

    # close memmap


def write_trans_data_elem(file_r:h5py.File,dsetstr:str,out_dir:str):
    dset1 = file_r['Snapshots/' + dsetstr]
    nt,npts = dset1.shape

    # read ibool 
    ngll = len(file_r['Mesh/npol'])
    nspec = len(file_r['Mesh/elements'])
    ibool = file_r['Mesh/sem_mesh'][:]
    idx = np.arange(0,ngll,1)
    ngll_out = len(idx)

    # create an memmap file
    dset2 = np.memmap(out_dir + '/' + dsetstr + '.bin',dtype='f4',mode='w+',shape=(nspec,ngll_out,ngll_out,nt))

    # allocate buffer to reduce io time
    sizeGB = 2
    npts_one = int((sizeGB * 1024**3) / (nt*4))
    for i in tqdm(range(0,npts,npts_one)):
        ntasks = 0
        if i + npts_one <=npts:
            ntasks = npts_one 
        else:
            ntasks = npts - i 

        istart = i
        iend = i + ntasks
        n = iend - istart
        if n<=0 : n = 0

        # alloc space
        mydata = np.zeros((nt,n),dtype='f4')

        # read data from dset1 
        mydata = dset1[:,istart:istart+n] * np.float32(1.)
        mydata = np.transpose(mydata)

        # write to dset2
        mask = (ibool >= istart) & (ibool < istart + n) 
        dset2[mask] = mydata[ibool[mask] - istart]

        # flush to disk
        dset2.flush()

    # close memmap

# def write_transpose_data(file_r:h5py.File,file_w:h5py.File,
#                          dataname:str,dsetstr:str,stride=1):
#     dset1 = file_r[dataname]
#     nt,npts = dset1.shape

#     # read ibool 
#     ngll = len(file_r['Mesh/npol'])
#     nspec = len(file_r['Mesh/elements'])
#     ibool = file_r['Mesh/sem_mesh'][:]
#     idx = np.arange(0,ngll,stride)
#     ngll_out = len(idx)

#     # mpi rank/nprocs
#     comm = MPI.COMM_WORLD
#     nprocs = comm.Get_size()
#     myrank = comm.Get_rank()

#     # create temp dataset
#     dset3 = file_w.create_dataset(dsetstr + ".tmp",(npts,nt),dtype=np.float32)

#     # estimate max memory usage
#     sizeGB = 2.
    
#     npts_one = int((sizeGB * 1024**3) / (nt*4))
#     for i in tqdm(range(0,npts,npts_one)):
#         ntasks = 0
#         if i + npts_one <=npts:
#             ntasks = npts_one 
#         else:
#             ntasks = npts - i 

#         istart,iend = allocate_task(ntasks,nprocs,myrank)
#         n = iend - istart + 1
#         istart += i 
#         if n<=0 : n = 0

#         # alloc space
#         mydata = np.zeros((nt,n),dtype='f4')

#         # read data from dset1 
#         mydata = dset1[:,istart:istart+n] * np.float32(1.)
#         mydata = np.transpose(mydata)

#         # write to dset2
#         dset3[istart:istart+n,:] = mydata 

#     # barrier
#     comm.barrier()

#     # now we change (ngll_all,nt) to (nspec,ngll_out,ngll_out,nt)
#     dset2 = file_w.create_dataset(dsetstr,(nspec,ngll_out,ngll_out,nt),dtype=np.float32)

#     istart,iend = allocate_task(nspec,nprocs,myrank)
#     for ispec in tqdm(range(istart,iend+1)):
#         for iz in range(ngll_out):
#             for ix in range(ngll_out):
#                 iz1 = idx[iz]
#                 ix1 = idx[ix]
#                 iglob = ibool[ispec,iz1,ix1]
#                 dset2[ispec,iz,ix,:] = dset3[iglob,:] * np.float32(1.)

#     # close and delete
#     comm.barrier()
#     del file_w[dsetstr + ".tmp"]

def main():
    if len(sys.argv) < 4:
        print("Usage:./this basedir to_dof MZZ MXX_P_MYY ...")
        exit(1)
    #
    ndirec = len(sys.argv) - 3
    basedir = sys.argv[1]
    to_dof = int(sys.argv[2]) 

    for i in range(ndirec):
        direc = sys.argv[i + 3]
        infile = basedir + "/" + direc + "/Data/axisem_output.nc4"
        file_r = h5py.File(infile,"r+")
        for dsetstr in ["disp_s","disp_p","disp_z"]:
            dataname = "Snapshots/" + dsetstr
            if dataname not in file_r.keys():
                continue

            print(f"reading {dataname} from {direc}")
            #write_transpose_data(file_r,file_w,dataname,dsetstr)
            if to_dof ==1 :
                write_trans_data_dof(file_r,dsetstr,basedir + "/" + direc + "/Data")
            else:
                write_trans_data_elem(file_r,dsetstr,basedir + "/" + direc + "/Data")

            # delete dataset
            #del file_r[dataname]

        # close file
        file_r.close()


if __name__ == "__main__":
    main()

