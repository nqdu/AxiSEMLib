import numpy as np 
import h5py 
from tqdm import tqdm 
import sys 
from mpi4py import MPI
import os
import subprocess

def write_trans_data_dof(infile:str,dsetstr:str,out_dir:str, sizeGB:float=5.0):
    """
    Transpose data from (nt,npts) to (npts,nt) and write to memmap file

    Parameters
    -------------------
    infile: str
        input file path
    dsetstr: str
        dataset name under Snapshots/
    out_dir: str
        output directory
    sizeGB: float
        buffer size in GB for each read/write operation
    --------------------------- 
    """
    file_r = h5py.File(infile,"r")
    dset1 = file_r['Snapshots/' + dsetstr]
    nt,npts = dset1.shape

    # create an memmap file
    dset2 = np.memmap(out_dir + '/' + dsetstr + '.bin',dtype='f4',mode='w+',shape=(npts,nt))

    # allocate buffer to reduce io time
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


def write_trans_data_elem(infile:str,dsetstr:str,out_dir:str, sizeGB:float=5.0):
    """
    Transpose data from (nt,npts) to (nspec,ngll,ngll,nt) and write to memmap file
    
    Parameters
    -------------------
    infile: str
        input file path
    dsetstr: str
        dataset name under Snapshots/
    out_dir: str
        output directory    
    sizeGB: float
        buffer size in GB for each read/write operation 
    ---------------------------
    """
    file_r = h5py.File(infile,"r")
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

def main():
    if len(sys.argv) < 4:
        print("Usage:./this to_dof sizeGB_per_rank direc1 [direc2]  ...")
        exit(1)
    #
    ndirec = len(sys.argv) - 3
    to_dof = int(sys.argv[1]) 
    sizeGB_per_rank = float(sys.argv[2])

    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # loop over directories
    work_list = []
    for i in range(ndirec):
        source_direcs = ['MZZ','MXZ_MYZ','MXY_MXX_M_MYY','MXX_P_MYY','PX','PZ']
        direc = sys.argv[i + 3]
        for s in source_direcs:
            infile = direc + "/" + s + "/Data/axisem_output.nc4"

            # check if file exist
            if not os.path.isfile(infile):
                continue

            file_r = h5py.File(infile,"r")
            for dsetstr in ["disp_s","disp_p","disp_z"]:
                dataname = "Snapshots/" + dsetstr
                if dataname not in file_r.keys():
                    continue
                # append to work list
                work_list.append( (infile,dsetstr,direc + "/" + s + "/Data") )
            file_r.close()

    # allocate work to each rank
    njobs = len(work_list)
    for i in range(rank, njobs, size):
        infile,dsetstr,out_dir = work_list[i]
        print(f"Rank {rank} processing file {infile} dataset {dsetstr} ...")
        if to_dof == 1:
            write_trans_data_dof(infile,dsetstr,out_dir, sizeGB=sizeGB_per_rank)
        else:
            write_trans_data_elem(infile,dsetstr,out_dir, sizeGB=sizeGB_per_rank)


    # sync all ranks
    comm.Barrier()

    # delete all variables
    if rank == 0:
        for i in range(njobs):
            infile,dsetstr,_ = work_list[i]
            print(f"deleting variables... Snapshots/{dsetstr} in {infile}")

            file_r = h5py.File(infile,"a")
            del file_r['Snapshots/' + dsetstr]
            file_r.close()

        # find unique filenames
        filenames = set()
        for i in range(njobs):
            infile,_,_ = work_list[i]
            filenames.add(infile)

        # loop over filenames to repack
        for infile in filenames:
            print(f"repacking file {infile} ...")
            newname = infile + '.bak'
            os.rename(infile,newname)
            try:
                subprocess.run(['h5repack',newname,infile],check=True)
            except (subprocess.CalledProcessError,FileNotFoundError) as e:
                print(f"h5repack failed for {infile}: {e}, restoring original file")
                if os.path.isfile(infile):
                    os.remove(infile)
                os.rename(newname,infile)
                continue
            os.remove(newname)

    comm.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    main()

