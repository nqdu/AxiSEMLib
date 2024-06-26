import numpy as np 
import h5py 
from mpi4py import MPI 
from tqdm import tqdm 
import sys 
import os 

def allocate_task(ntasks,nprocs,myrank):

    sub_n = ntasks // nprocs
    num_larger_procs = ntasks - nprocs * sub_n
    starid = 0; endid = 0
    if myrank < num_larger_procs :
        sub_n = sub_n + 1
        startid = 0 + myrank * sub_n
    elif sub_n > 0:
        startid = 0 + num_larger_procs + myrank * sub_n
    else: # // this process has only zero elements
        startid = -1
        sub_n = 0
    
    endid = startid + sub_n - 1

    return startid,endid


def write_transpose_data(file_r:h5py.File,file_w:h5py.File,
                         dataname:str,dsetstr:str,stride=1):
    dset1 = file_r[dataname]
    nt,npts = dset1.shape

    # read ibool 
    ngll = len(file_r['Mesh/npol'])
    nspec = len(file_r['Mesh/elements'])
    ibool = file_r['Mesh/sem_mesh'][:]
    idx = np.arange(0,ngll,stride)
    ngll_out = len(idx)

    nprocs = MPI.COMM_WORLD.Get_size()
    myrank = MPI.COMM_WORLD.Get_rank()

    # create temp dataset
    dset3 = file_w.create_dataset(dsetstr,(npts,nt),dtype=np.float32)

    # estimate max memory usage
    sizeGB = 4.
    
    npts_one = int((sizeGB * 1024**3) / (nt*4))
    for i in tqdm(range(0,npts,npts_one)):
        ntasks = 0
        if i + npts_one <=npts:
            ntasks = npts_one 
        else:
            ntasks = npts - i 

        istart,iend = allocate_task(ntasks,nprocs,myrank)
        n = iend - istart + 1
        istart += i 
        if n<=0 : n = 0

        # alloc space
        mydata = np.zeros((nt,n),dtype='f4')

        # read data from dset1 
        mydata = dset1[:,istart:istart+n] * np.float32(1.)
        mydata = np.transpose(mydata)

        # write to dset2
        dset3[istart:istart+n,:] = mydata 

def main():
    if len(sys.argv) < 3:
        print("Usage:./this basedir MZZ MXX_P_MYY ...")
        exit(1)

    # get mpi info
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()

    #
    ndirec = len(sys.argv) - 2
    basedir = sys.argv[1]
    for i in range(ndirec):
        direc = sys.argv[i + 2]
        infile = basedir + "/" + direc + "/Data/axisem_output.nc4"
        outfile = basedir + "/" + direc + "/Data/axisem_fields.h5"
        file_r = h5py.File(infile,"r+",driver='mpio',comm=MPI.COMM_WORLD)
        file_w = h5py.File(outfile,"w",driver='mpio',comm=MPI.COMM_WORLD)

        for dsetstr in ["disp_s","disp_p","disp_z"]:
            dataname = "Snapshots/" + dsetstr
            if dataname not in file_r.keys():
                continue

            if myrank == 0: print(f"reading {dataname} from {direc}")
            write_transpose_data(file_r,file_w,dataname,dsetstr)

            # delete dataset
            #del file_r[dataname]

        # close file
        file_r.close(); file_w.close()

        # repack the h5file 
        if myrank == 0:
            newname = infile + '.bak'
            os.rename(infile,newname)
            os.system(f'h5repack {newname} {infile}')
            os.remove(newname)

            newname = outfile + '.bak'
            os.rename(outfile,newname)
            os.system(f'h5repack {newname} {outfile}')
            os.remove(newname)
            pass 
    comm.Barrier()
main()

