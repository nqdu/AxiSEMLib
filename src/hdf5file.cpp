#include "hdf5file.hpp"
#include <assert.h>
#include <iostream>

#ifdef USE_MPI
#include <mpi.h>
#endif


/**
 * @brief open a HDF5 file
 * 
 * @param filename filename of this HDF5 file
 * @param mode open mode, "r": readonly, "w": create new one, "rw":read and write 
 */
void HDF5File:: 
open(const std::string &filename,const std::string &mode)
{
    assert(mode == "r" || mode == "w" || mode == "rw");

    hid_t plist_id = H5P_DEFAULT;
    #ifdef USE_MPI
        plist_id = H5Pcreate(H5P_FILE_ACCESS); //file access pl id
        H5Pset_fapl_mpio(plist_id,MPI_COMM_WORLD, MPI_INFO_NULL);
    #endif

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
}

HDF5File::
HDF5File(const std::string &filename,const std::string &mode)
{
    this->open(filename,mode);
}

/**
 * @brief release all resources for this HDF5 file
 * 
 */
void HDF5File:: 
close()
{
    H5Fclose(file_id);
}

/**
 * @brief write data to hdf5 file, mpi support
 * 
 * @param dataname dataset name
 * @param data output data
 * @param rows,cols rows and cols for output data. The cols should be the same for each proc
 * @param hdtype hdf5 datatype
 * @param out_rank if out_rank >0, only write out by this rank
 */
void HDF5File::
__write_data(const std::string &dataname,const void *data,size_t rows,
             size_t cols, hid_t hdtype,int out_rank) const
{
    hsize_t dim_all[2];
    herr_t status;

    // mpi info
    int myrank = 0,nprocs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
#endif
    size_t rows_all[nprocs],cols_all[nprocs];

    // check if only output by one proc
    if(out_rank >= 0 && out_rank < nprocs){
        if(myrank != out_rank){
            rows = 0;
        }
    }

#ifdef USE_MPI
    // get rows and cols from all procs
    MPI_Datatype mysize_t = MPI_UNSIGNED_LONG;
    MPI_Allgather(&rows,1,mysize_t,rows_all,1,mysize_t,MPI_COMM_WORLD);
    MPI_Allgather(&cols,1,mysize_t,cols_all,1,mysize_t,MPI_COMM_WORLD);

    // check if cols are the same for each proc
    bool flag = true;
    for(int i = 1; i< nprocs;i++){
        if(cols_all[i] != cols_all[0]){
            flag = false;
            break;
        }
    }
    if(!flag) {
        printf("the cols for data should be same for each proc!\n");
        printf("cols for each proc:\n");
        for(int i = 0; i < nprocs; i ++){
            printf("rank %d - col %zu\n",i,cols_all[i]);
        }
        exit(-1);
    }

    // create dataset with enough storage
    dim_all[0] = 0;
    for (int i =0;i < nprocs;i++) dim_all[0] += rows_all[i];
    dim_all[1] = cols;
#else
    rows_all[0] = rows; cols_all[0] = cols;
    dim_all[0] = rows;
    dim_all[1] = cols;
#endif

    // create dataset with enough storage
    int ndims = 2;
    if(cols == 1) ndims = 1;
    auto filespace = H5Screate_simple(ndims,dim_all,NULL);
    auto dset_id = H5Dcreate2(file_id,dataname.data(),hdtype,filespace,
                                H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Sclose(filespace);

    // set data dim and offset for each proc
    hsize_t dims[2] = {(hsize_t)rows,(hsize_t)cols};
    hsize_t offset[2] = {(hsize_t)0,(hsize_t)0};
    for(int i = 0;i < myrank;i++){
        offset[0] += rows_all[i];
    }

    // create sub space
    auto memspace = H5Screate_simple(ndims,dims,NULL);
    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace,H5S_SELECT_SET,offset,NULL,dims,NULL);

    // write data
    hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef USE_MPI
    H5Pset_dxpl_mpio(dxpl_id,H5FD_MPIO_COLLECTIVE);
#endif
    status = H5Dwrite(dset_id,hdtype,memspace,filespace,dxpl_id,data);
    if(status < 0){
        printf("cannot write %s\n",dataname.data());
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // close dataspace,dataset,group and file
    status = H5Dclose(dset_id);
    status = H5Sclose(filespace);
    status = H5Sclose(memspace);
    status = H5Pclose(dxpl_id);

    // sychronize
    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief read data from a hdf5 file, mpi support
 * 
 * @param dataname dataset name
 * @param data output data
 * @param rows,cols rows and cols for output data 
 * @param hdtype hdf5 datatype
 * @param in_rank if >0, only read by this rank
 */
void HDF5File::
__read_data(const std::string &dataname,void *data,size_t rows,
         size_t cols, hid_t hdtype,int in_rank) const
{
    herr_t status;

    // mpi info
    int myrank = 0, nprocs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
#endif
    size_t rows_all[nprocs];

    // check if only read by one proc
    if(in_rank >= 0 && in_rank < nprocs){
        if(myrank != in_rank){
            rows = 0;
        }
    }

    // check if this dataset is in hdf5 file
    if(this -> get_info(dataname) != 0){
        printf("dataset %s is not in this file!\n",dataname.data());
        exit(1);
    }

    #ifdef USE_MPI
        // get rows and cols from all procs
        size_t cols_all[nprocs];
        MPI_Datatype mysize_t = MPI_UNSIGNED_LONG;
        MPI_Allgather(&rows,1,mysize_t,rows_all,1,mysize_t,MPI_COMM_WORLD);
        MPI_Allgather(&cols,1,mysize_t,cols_all,1,mysize_t,MPI_COMM_WORLD);

        // check if cols are the same for each proc
        bool flag = true;
        for(int i = 1; i< nprocs;i++){
            if(cols_all[i] != cols_all[0]){
                flag = false;
                break;
            }
        }
        if(!flag) {
            printf("the cols for data should be same for each proc!\n");
            exit(-1);
        }
    #else
        rows_all[0] = rows;
    #endif

    // get space of dataset
    auto dset_id = H5Dopen2(file_id,dataname.c_str(),H5P_DEFAULT);
    auto dataspace = H5Dget_space(dset_id);

    // set data dim and offset for each proc
    hsize_t dims[2] = {(hsize_t)rows,(hsize_t)cols};
    hsize_t offset[2] = {0,0};
    for(int i = 0;i < myrank;i++){
        offset[0] += rows_all[i];
    }

    // create sub space
    int ndims = 2;
    if(cols == 1) ndims = 1;
    auto memspace = H5Screate_simple(ndims,dims,NULL);
    H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,dims,NULL);

    // read data
    hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef USE_MPI
    H5Pset_dxpl_mpio(dxpl_id,H5FD_MPIO_COLLECTIVE);
#endif
    status = H5Dread(dset_id,hdtype,memspace,dataspace,dxpl_id,data);
    if(status < 0){
        printf("cannot read %s\n",dataname.data());
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // close dataspace,dataset,group and file
    status = H5Dclose(dset_id);
    status = H5Sclose(dataspace);
    status = H5Sclose(memspace);
    status = H5Pclose(dxpl_id);
}

/**
 * @brief write a single string to hdf5 file
 * 
 * @param name dataset name 
 * @param datastr string to be written
 */
void HDF5File:: 
write_string(const std::string &name,const std::string &datastr) const
{
    int myrank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#endif
    herr_t status;

    // get datasize and memory size
    size_t size = datastr.size();
    auto memtype = H5Tcopy (H5T_C_S1);
    H5Tset_size(memtype,size);

    // allocate space
    auto space = H5Screate_simple (0, NULL, NULL);
    auto dset = H5Dcreate2 (file_id, name.data(), memtype, space,
                            H5P_DEFAULT, H5P_DEFAULT,H5P_DEFAULT);

    if(myrank == 0){
        status = H5Dwrite (dset, memtype, H5S_ALL, H5S_ALL, 
                          H5P_DEFAULT,datastr.c_str());
        if(status < 0){
            printf("cannot write %s\n",name.c_str());
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    // close space
    status = H5Dclose (dset);
    status = H5Sclose (space);
    status = H5Tclose (memtype);
}

/**
 * @brief write scalar data to hdf5 file
 * 
 * @param name data name
 * @param scalar scalar data itself
 * @param hdtype hdf5 datatype
 */
void HDF5File:: 
__write_scalar(const std::string &name,const void *scalar,hid_t hdtype) const
{
    int myrank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#endif
    herr_t status;

    // create space
    auto filespace = H5Screate(H5S_SCALAR);
    auto dset_id = H5Dcreate2(file_id,name.data(),hdtype,filespace,H5P_DEFAULT,
                              H5P_DEFAULT,H5P_DEFAULT);

    // write data
    if(myrank == 0){
        status = H5Dwrite(dset_id,hdtype,H5S_ALL,filespace,H5S_ALL,scalar);
        if(status < 0){
            printf("cannot write %s\n",name.c_str());
            exit(1);
        }
    }

    // close space
    status = H5Dclose(dset_id);
    status = H5Sclose(filespace);

}

/**
 * @brief read scalar data to hdf5 file
 * 
 * @param name data name
 * @param scalar scalar data itself
 * @param hdtype hdf5 datatype
 */
void HDF5File:: 
__read_scalar(const std::string &name, void *scalar,hid_t hdtype, size_t size) const
{
    int myrank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#endif
    herr_t status;

    // get space
    auto dset_id = H5Dopen2(file_id,name.c_str(),H5P_DEFAULT);
    auto dataspace = H5Dget_space(dset_id);

    // write data
    if(myrank == 0){
        status = H5Dread(dset_id,hdtype,H5P_DEFAULT,dataspace,
                                H5P_DEFAULT,scalar);
        if(status < 0){
            printf("cannot read %s\n",name.c_str());
            exit(1);
        }
    }

    // close space
    status = H5Dclose(dset_id);
    status = H5Sclose(dataspace);

    // bcast if required
    #ifdef USE_MPI
    char *out = (char*)scalar;
    MPI_Bcast(out,size,MPI_CHAR,0,MPI_COMM_WORLD);
    #endif
}

/**
 * @brief read a single string from hdf5 file
 * 
 * @param name dataset name 
 * @param datastr string to be read
 */
void HDF5File:: 
read_string(const std::string &name, std::string &datastr) const
{
    int myrank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#endif
    herr_t status;
    auto dset = H5Dopen (file_id, name.data(), H5P_DEFAULT);

    auto filetype = H5Dget_type (dset);
    size_t sdim = H5Tget_size (filetype);
    sdim += 1;
    char buffer[sdim];

    //Create the memory datatype.
    auto memtype = H5Tcopy (H5T_C_S1);
    status = H5Tset_size (memtype, sdim);

    if(myrank == 0){
        status = H5Dread (dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
        if(status < 0){
            printf("cannot read %s\n",name.c_str());
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    #ifdef USE_MPI
    MPI_Bcast(buffer,sdim,MPI_CHAR,0,MPI_COMM_WORLD);
    #endif
    datastr = std::string(buffer);

    status = H5Dclose (dset);
    status = H5Tclose (filetype);
    status = H5Tclose (memtype);
}


/**
 * @brief Create a group in a hdf5 file
 * 
 * @param groupname name of this group
 */
void HDF5File:: 
create_group(const std::string &groupname) const
{
    auto group_id = H5Gcreate2(file_id,groupname.data(),H5P_DEFAULT,
                               H5P_DEFAULT,H5P_DEFAULT);
    auto status = H5Gclose(group_id);

    if(status < 0){
        printf("cannot create group %s\n",groupname.c_str());
        exit(1);
    }
}

/**
 * @brief rename a link with "oldname" to "newname". 
 *          The hdf5 file should be open with "rw" mode
 * 
 * @param oldname old name for this link
 * @param newname new name for this link
 * @param locname location path , the link is at locname/oldname
 */
void HDF5File::
rename_link(const std::string &oldname,const std::string &newname,
                const std::string &locname)
{
    int myrank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#endif

    auto groupid = H5Gopen2(file_id,locname.c_str(),H5P_DEFAULT);
    auto status = H5Lcreate_hard(groupid,oldname.c_str(),H5L_SAME_LOC,
                            newname.c_str(),H5P_DEFAULT,H5P_DEFAULT);
    status = H5Ldelete(groupid,oldname.c_str(),H5P_DEFAULT);

    if(status < 0){
        std::string info = "cannot delete " + oldname + "\n";
        if(myrank == 0){
            printf("%s\n",info.c_str());
        }
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // close group
    H5Gclose(groupid);
}

/**
 * @brief delete a link in HDF5 file, The hdf5 file should be open with "rw" mode
 * 
 * @param name link name
 * @param locname path, the link is at locname/name
 */
void HDF5File::
delete_link(const std::string &name,const std::string &locname)
{
    int myrank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#endif
    auto groupid = H5Gopen2(file_id,locname.c_str(),H5P_DEFAULT);
    auto status = H5Ldelete(groupid,name.c_str(),H5P_DEFAULT);

    if(status < 0){
        std::string info = "cannot delete " + name + "\n";
        if(myrank == 0){
            printf("%s\n",info.c_str());
        }
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // close group
    H5Gclose(groupid);
}

/**
 * @brief Get the the name for all objects under path
 * 
 * @param path a location in hdf5 file
 * @return std::vector<std::string> names 
 */
std::vector<std::string>  HDF5File:: 
get_objname(const std::string &path) const
{
    auto group_id = H5Gopen2(file_id,path.data(),H5P_DEFAULT);
    std::vector<std::string> objnames;

    // get # of objs
    hsize_t num; H5Gget_num_objs(group_id,&num);
    for(hsize_t i = 0; i < num; i ++){
        char name[8192];
        H5Gget_objname_by_idx(group_id,i,name,sizeof(name));
        objnames.push_back(std::string(name));
    }

    // close current group
    H5Gclose(group_id);

    return objnames;
}

    // error handler
static herr_t silenthandler(hid_t,void*){
    return -1;
}

/**
 * @brief Get the info for a object
 * 
 * @param objname name for this object
 * @return int 0 for dataset,1 for group and -1 for others
 */
int HDF5File:: 
get_info(const std::string &objname) const
{
    // diable error printing
    H5Eset_auto(H5E_DEFAULT,nullptr,nullptr);
    H5Ewalk2(H5E_DEFAULT,H5E_WALK_DOWNWARD,(H5E_walk2_t)silenthandler,nullptr);

    auto obj_id = H5Oopen(file_id,objname.data(),H5P_DEFAULT);
    if(obj_id < 0) {
        H5Oclose(obj_id);
        return -1;
    }
    H5Eset_auto(H5E_DEFAULT,nullptr,nullptr);

    // get information
    H5O_info_t info;
    H5Oget_info(obj_id,&info,H5O_INFO_BASIC);
    int flag{};

    if(info.type == H5O_TYPE_DATASET){
        flag = 0;
    }
    else if(info.type == H5O_TYPE_GROUP){
        flag = 1;
    }
    else{
        flag = -1;
    }

    H5Oclose(obj_id);
    return flag;
}

/**
 * @brief write attribute into a group or a dataset
 * @param attrname name of this attribute
 * @param data data
 * @param n size of data
 * @param datatype type of the data
 * @param vname location of this attribute, could be a group or a dataset
 */
void HDF5File::
 __write_attr(const std::string &attrname,const void *data,size_t n,
              hid_t datatype,const std::string &vname) const
{
    // check vname type
    int vname_type = this -> get_info(vname);
    if(vname_type !=0 && vname_type != 1){
        printf("vname %s must be a dataset/group!",vname.data());
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // open vname 
    hid_t vname_id;
    if(vname_type == 1){
        vname_id =  H5Gopen2(file_id,vname.data(),H5P_DEFAULT);
    }
    else{
        vname_id = H5Dopen2(file_id,vname.data(),H5P_DEFAULT);
    }

    // write attribute
    hsize_t attr_dims[1] = {n};
    auto attr_dataspace = H5Screate_simple(1, attr_dims, NULL);
    auto attr_id = H5Acreate2(vname_id,attrname.data(),datatype,attr_dataspace, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status;
    status = H5Awrite(attr_id,datatype,data);
    if(status < 0){
        printf("cannot write attribute %s\n",attrname.data());
        MPI_Abort(MPI_COMM_WORLD,1);
    } 
    status = H5Aclose (attr_id);
    status = H5Sclose (attr_dataspace);
    if(vname_type == 1){
        status = H5Gclose (vname_id);
    }
    else{
        status = H5Dclose(vname_id);
    }
}

/**
 * @brief write string attribute to a group/dataset 
 * @param attrname name of this attribute
 * @param info input string
 * @param vname vname location of this attribute, could be a group or a dataset
 */
void HDF5File:: 
write_attr(const std::string &attrname,const std::string &info,
           const std::string &vname) const 
{
    // check vname type
    int vname_type = this -> get_info(vname);
    if(vname_type !=0 && vname_type != 1){
        printf("vname %s must be a dataset/group!",vname.data());
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // open vname 
    hid_t vname_id;
    if(vname_type == 1){
        vname_id =  H5Gopen2(file_id,vname.data(),H5P_DEFAULT);
    }
    else{
        vname_id = H5Dopen2(file_id,vname.data(),H5P_DEFAULT);
    }

    // write data type attributes
    auto attr_dataspace = H5Screate(H5S_SCALAR);
    auto attr_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(attr_type, info.size());
    auto attr_id = H5Acreate2(vname_id, attrname.data(), attr_type, attr_dataspace, H5P_DEFAULT, H5P_DEFAULT);
    auto status = H5Awrite(attr_id, attr_type, info.data());
    if(status < 0){
        printf("cannot write string attribute %s\n",attrname.data());
        MPI_Abort(MPI_COMM_WORLD,1);
    } 

    // close 
    status = H5Aclose (attr_id);
    status = H5Sclose (attr_dataspace);
    if(vname_type == 1){
        status = H5Gclose (vname_id);
    }
    else{
        status = H5Dclose(vname_id);
    }
}

/**
 * @brief read  attribute from a group or a dataset
 * @param attrname name of this attribute
 * @param data data
 * @param n size of data
 * @param datatype type of the data
 * @param vname location of this attribute, could be a group or a dataset
 */
void HDF5File::
read_attr(const std::string &attrname,void *data,size_t n,
          const std::string &vname) const
{
    // check vname type
    int vname_type = this -> get_info(vname);
    if(vname_type !=0 && vname_type != 1){
        printf("vname %s must be a dataset/group!",vname.data());
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // open vname 
    hid_t vname_id;
    if(vname_type == 1){
        vname_id =  H5Gopen2(file_id,vname.data(),H5P_DEFAULT);
    }
    else{
        vname_id = H5Dopen2(file_id,vname.data(),H5P_DEFAULT);
    }

    // write attribute
    auto attr_id = H5Aopen(vname_id, attrname.data(), H5P_DEFAULT);
    auto attr_space = H5Aget_space(attr_id);
    hsize_t attr_dims[1];
    H5Sget_simple_extent_dims(attr_space, attr_dims, NULL);
    assert(attr_dims[0] == n);
    auto datatype =  H5Aget_type(attr_id);
    herr_t status;
    status = H5Aread(attr_id,datatype,data);
    if(status < 0){
        printf("cannot read attribute %s\n",attrname.data());
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    //status = H5Sclose (datatype);
    status = H5Aclose (attr_id);
    status = H5Sclose(attr_space);
    status = H5Tclose(datatype);
    
    if(vname_type == 1){
        status = H5Gclose (vname_id);
    }
    else{
        status = H5Dclose(vname_id);
    }
}

/**
 * @brief read string attribute to a group/dataset 
 * @param attrname name of this attribute
 * @param info input string
 * @param vname vname location of this attribute, could be a group or a dataset
 */
void HDF5File:: 
read_attr(const std::string &attrname,std::string &info,
           const std::string &vname) const 
{
    // check vname type
    int vname_type = this -> get_info(vname);
    if(vname_type !=0 && vname_type != 1){
        printf("vname %s must be a dataset/group!",vname.data());
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // open vname 
    hid_t vname_id;
    if(vname_type == 1){
        vname_id =  H5Gopen2(file_id,vname.data(),H5P_DEFAULT);
    }
    else{
        vname_id = H5Dopen2(file_id,vname.data(),H5P_DEFAULT);
    }

    // write data type attributes
    auto attr_id = H5Aopen(vname_id, attrname.data(), H5P_DEFAULT);
    size_t sdim = H5Aget_storage_size(attr_id);
    sdim += 1;
    char buffer[sdim];
    auto attr_dataspace =  H5Aget_space(attr_id);
    auto attr_type = H5Aget_type(attr_id);
    auto status = H5Aread(attr_id,attr_type,buffer);
    if(status < 0){
        printf("cannot read string attribute %s\n",attrname.data());
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    buffer[sdim-1] = '\0';
    info = std::string(buffer);

    // close 
    status = H5Sclose (attr_dataspace);
    status = H5Aclose (attr_id);
    if(vname_type == 1){
        status = H5Gclose (vname_id);
    }
    else{
        status = H5Dclose(vname_id);
    }
}
