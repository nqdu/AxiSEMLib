#include "hdf5file.hpp"
#include <Eigen/Core>
#include <iostream>

void allocate_task(int ntasks,int &startid,int &endid)
{
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    int sub_n = ntasks / nprocs;
    int num_larger_procs = ntasks - nprocs * sub_n;
    if (myrank < num_larger_procs){ 
        sub_n = sub_n + 1;
        startid = 0 + myrank * sub_n;
    }
    else if (sub_n > 0){ 
        startid = 0 + num_larger_procs + myrank * sub_n;
    }
    else { // this process has only zero elements
        startid = -1;
        sub_n = 0;
    }
    endid = startid + sub_n - 1;
}

int main(int argc,char *argv[]) {

    // strings
    std::string basedir = "../ak135";
    std::string direc; 

    if(argc < 3){
        printf("Usage ./this basedir MZZ MXX_P_MYY ...\n");
        exit(1);
    } 

    // mpi
    MPI_Init(&argc,&argv);
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    int ndirec = argc - 2;
    basedir = std::string(argv[1]);

    for(int i =0; i < ndirec; i ++){
        // Open the source HDF5 file for reading
        direc = std::string(argv[i+2]);
        std::string infile = basedir + "/" + direc + "/Data/axisem_output.nc4";
        std::string outfile = basedir + "/" + direc + "/Data/axisem_fields.h5";

        HDF5File fs(infile,"r");
        HDF5File fout(outfile,"w");
        herr_t status;
        char dsetstr[3][256] = {"disp_s","disp_p","disp_z"};

        for(int ir = 0; ir < 3; ir ++) {
            std::string dataname = "Snapshots/" + std::string(dsetstr[ir]);
            auto hdtype = H5T_NATIVE_FLOAT;

            // check exist
            if(H5Lexists(fs.file_id, dataname.data(), H5P_DEFAULT) <=0) {
                continue;
            }

            if(myrank == 0) printf("reading %s from %s\n",dataname.data(),direc.data());

            // get space of dataset
            auto dset_id = H5Dopen2(fs.file_id,dataname.c_str(),H5P_DEFAULT);
            auto dataspace = H5Dget_space(dset_id);

            // get dim
            hsize_t mydim[2]; 
            H5Sget_simple_extent_dims(dataspace, mydim, NULL);
            int nt = mydim[0], npts = mydim[1];
            int startid,endid;
            allocate_task(npts,startid,endid);
            int nspec = endid + 1 - startid;
            Eigen::Array<float,-1,-1,1> mydata;
            mydata.resize(nt,nspec);

            // mpi info
            size_t rows_all[nprocs],cols_all[nprocs];
            size_t rows = nt, cols = nspec;

            // get rows and cols from all procs
            MPI_Datatype mysize_t = MPI_UNSIGNED_LONG;
            MPI_Allgather(&rows,1,mysize_t,rows_all,1,mysize_t,MPI_COMM_WORLD);
            MPI_Allgather(&cols,1,mysize_t,cols_all,1,mysize_t,MPI_COMM_WORLD);

            // set data dim and offset for each proc
            hsize_t dims[2] = {(hsize_t)rows,(hsize_t)cols};
            hsize_t offset[2] = {0,0};
            for(int ii = 0;ii < myrank;ii++){
                offset[1] += cols_all[ii];
            }

            // create sub space
            int ndims = 2;
            auto memspace = H5Screate_simple(ndims,dims,NULL);
            H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,dims,NULL);

            // read data
            hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
        #ifdef USE_MPI
            H5Pset_dxpl_mpio(dxpl_id,H5FD_MPIO_COLLECTIVE);
        #endif
            status = H5Dread(dset_id,hdtype,memspace,dataspace,dxpl_id,mydata.data());
            if(status < 0){
                printf("cannot read %s\n",dataname.data());
                MPI_Abort(MPI_COMM_WORLD,1);
            }
            // close dataspace,dataset,group and file
            status = H5Dclose(dset_id);
            status = H5Sclose(dataspace);
            status = H5Sclose(memspace);
            status = H5Pclose(dxpl_id);

            // write data
            mydata.transposeInPlace();
            dataname = std::string(dsetstr[ir]);
            fout.write_data(dataname,mydata.data(),nspec,nt);
        }
        fs.close();
        fout.close();

    }

    MPI_Finalize();

    return 0;
}