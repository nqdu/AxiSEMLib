#include <Eigen/Core>
#include <iostream>
#include <hdf5.h>

void allocate_task(int ntasks,int nprocs,int myrank,
                int &startid,int &endid)
{
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

void printbar(float progress){
    int barWidth = 70;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout << "\n";
    std::cout.flush();
    
}

hid_t
open(const std::string &filename,const std::string &mode) {
    hid_t plist_id = H5P_DEFAULT;
    #ifdef USE_MPI
        plist_id = H5Pcreate(H5P_FILE_ACCESS); //file access pl id
        H5Pset_fapl_mpio(plist_id,MPI_COMM_WORLD, MPI_INFO_NULL);
    #endif

    hid_t file_id;
    if(mode == "w"){
        file_id = H5Fcreate(filename.data(),H5F_ACC_TRUNC,H5P_DEFAULT,plist_id);
    }
    else if(mode == "r"){
        file_id = H5Fopen(filename.data(),H5F_ACC_RDONLY,plist_id);
    }
    else{
        file_id = H5Fopen(filename.data(),H5F_ACC_RDWR,plist_id);
    }
    if(file_id == H5I_INVALID_HID) {
        printf("cannot open file %s\n",filename.data());
        exit(1);
    }

    #ifdef USE_MPI
        H5Pclose(plist_id);
    #endif

    return file_id;
}

void 
write_trans_data(hid_t file_r,hid_t file_w,
                const char *inname,const char *outname)
{
    auto const hdtype = H5T_NATIVE_FLOAT;
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    herr_t status;

    // get space of dataset
    auto dset_id1 = H5Dopen2(file_r,inname,H5P_DEFAULT);
    auto dataspace1 = H5Dget_space(dset_id1);

    // get dim
    hsize_t mydim1[2],mydim2[2];
    H5Sget_simple_extent_dims(dataspace1, mydim1, NULL);
    int nt = mydim1[0], npts = mydim1[1];
    mydim2[0] = npts; mydim2[1] = nt;

    // create enough space
    auto filespace2 = H5Screate_simple(2,mydim2,NULL);
    auto dset_id2 = H5Dcreate2(file_w,outname,hdtype,filespace2,
                                H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Sclose(filespace2);

    // max memory usage
    const double sizeinGB = 10.;
    int istart,iend;
    size_t nspec_one = (sizeinGB *1024 *1024 * 1024) / (nt *sizeof(float));
    size_t count = 0;
    for(size_t i = 0; i < npts; i += nspec_one) {
        int ntasks;
        float per;
        if(i + nspec_one <= (size_t)npts){
            ntasks = nspec_one;
            per = (i +1.) / npts;
        }
        else{
            ntasks = npts - i;
            per = 1.0;
        }

        // progress bar 
        if(myrank == 0)printbar(per);

        //if(myrank == 0 )printf("%d %d %d\n",ntasks,i + ntasks,npts);
        
        // allocate tasks on each proc
        allocate_task(ntasks,nprocs,myrank,istart,iend);
        int nspec = iend - istart +1;
        nspec = nspec > 0 ? nspec : 0;

        // allocate space for data
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
        hsize_t dims1[2] = {(hsize_t)rows,(hsize_t)cols};
        hsize_t offset1[2] = {0,0};
        for(int ii = 0;ii < myrank;ii++){
            offset1[1] += cols_all[ii];
        }
        offset1[1] += i;

        // create sub space
        int ndims = 2;
        auto memspace1 = H5Screate_simple(ndims,dims1,NULL);
        H5Sselect_hyperslab(dataspace1,H5S_SELECT_SET,offset1,NULL,dims1,NULL);

        // read data
        hid_t dxpl_id1 = H5Pcreate(H5P_DATASET_XFER);
    #ifdef USE_MPI
        H5Pset_dxpl_mpio(dxpl_id1,H5FD_MPIO_COLLECTIVE);
    #endif
        status = H5Dread(dset_id1,hdtype,memspace1,dataspace1,dxpl_id1,mydata.data());
        if(status < 0){
            printf("cannot read %s\n",inname);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        // close dataspace,dataset,group and file
        status = H5Sclose(memspace1);
        status = H5Pclose(dxpl_id1);
        //if(myrank == 0 )printf("finish reading ...\n");

        // transpose
        mydata.transposeInPlace();

        // write out 
        hsize_t dims2[2] = {(hsize_t)cols,(hsize_t)rows};
        hsize_t offset2[2] = {0,0};
        for(int ii = 0;ii < myrank;ii++){
            offset2[0] += cols_all[ii];
        }
        offset2[0] += i;
        auto memspace2 = H5Screate_simple(2,dims2,NULL);
        filespace2 = H5Dget_space(dset_id2);
        H5Sselect_hyperslab(filespace2,H5S_SELECT_SET,offset2,NULL,dims2,NULL);
        // write data
        hid_t dxpl_id2 = H5Pcreate(H5P_DATASET_XFER);
    #ifdef USE_MPI
        H5Pset_dxpl_mpio(dxpl_id2,H5FD_MPIO_COLLECTIVE);
    #endif
        status = H5Dwrite(dset_id2,hdtype,memspace2,filespace2,dxpl_id2,mydata.data());
        if(status < 0){
            printf("cannot write %s\n",outname);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        status = H5Sclose(filespace2);
        status = H5Sclose(memspace2);
        status = H5Pclose(dxpl_id2);
    }

    // close
    status = H5Dclose(dset_id1);
    status = H5Sclose(dataspace1);
    status = H5Dclose(dset_id2);
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

        // open file
        hid_t file_r = open(infile,"r"),file_w = open(outfile,"w");
        herr_t status;
        char dsetstr[3][256] = {"disp_s","disp_p","disp_z"};


        for(int ir = 0; ir < 3; ir ++) {
            std::string dataname = "Snapshots/" + std::string(dsetstr[ir]);
            auto hdtype = H5T_NATIVE_FLOAT;

            // check exist
            if(H5Lexists(file_r, dataname.data(), H5P_DEFAULT) <=0) {
                continue;
            }
            if(myrank == 0) printf("reading %s from %s\n",dataname.data(),direc.data());

            write_trans_data(file_r,file_w,dataname.data(),dsetstr[ir]);
        }

        // close h5 file
        H5Fclose(file_r);
        H5Fclose(file_w);
    }

    MPI_Finalize();

    return 0;
}