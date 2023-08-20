#ifndef _WAVESEM3D_HDF5_H
#define _WAVESEM3D_HDF5_H

#include <string>
#include <vector>
#include <hdf5.h>
#include <type_traits>


class HDF5File {
public:    

public:
    // filename 
    hid_t file_id;

private:
    // get datatype 

    /**
     * @brief Get the datatype for HDF5 file
     * 
     * @return hid_t hdf5 native format
     */
    template<typename T>
    hid_t get_hdf5_type() const
    {
        static_assert(std::is_same<T,float>::value || std::is_same<T,int>::value ||
                     std::is_same<T,bool>::value || std::is_same<T,double>::value 
                     || std::is_same<T,int64_t>::value || std::is_same<T,char>::value);
        hid_t hdtype;
        if(std::is_same<T,float>::value){
            hdtype =  H5T_NATIVE_FLOAT;
        }
        else if (std::is_same<T,int>::value){
            hdtype =  H5T_NATIVE_INT32;
        }
        else if (std::is_same<T,int64_t>::value) {
            hdtype = H5T_NATIVE_INT64;
        }
        else if(std::is_same<T,double>::value){
            hdtype = H5T_NATIVE_DOUBLE;
        }
        else if (std::is_same<T,bool>::value){
            hdtype =  H5T_NATIVE_HBOOL;
        }
        else if (std::is_same<T,char>::value){
            hdtype = H5T_C_S1;
        }
        else {
            hdtype =  H5T_NATIVE_INT64;
        }

        return hdtype;
    }
public:

    HDF5File(){};
    HDF5File(const std::string &filename,const std::string &mode = "r");
    void open(const std::string &filename,const std::string &mode = "r");
    void close();
    void write_string(const std::string &name,const std::string &datastr) const;
    void read_string(const std::string &name, std::string &datastr) const;

    // create group
    void create_group(const std::string &groupname) const;

    // iteration function 
    std::vector<std::string> get_objname(const std::string &path) const;
    int get_info(const std::string &objname) const;

    // link function
    void delete_link(const std::string &name,const std::string &locname="./");
    void rename_link(const std::string &oldname,const std::string &newname,
                        const std::string &locname="./");

    // template function
    /**
     * @brief write data to hdf5 file, mpi support
     * 
     * @param dataname dataset name
     * @param data output data
     * @param rows,cols rows and cols for output data. The cols should be the same for each proc
     */
    template<typename T>
    void write_data(const std::string &dataname,const T *data,
                    size_t rows, size_t cols,int out_rank=-1) const 
    {
        hid_t datatype = this -> get_hdf5_type<T>();
        this -> __write_data(dataname,data,rows,cols,datatype,out_rank);
    }
    
    /**
     * @brief read data from a hdf5 file, mpi support
     * 
     * @param dataname dataset name
     * @param data output data
     * @param rows,cols rows and cols for output data 
     */
    template<typename T>
    void read_data(const std::string &dataname,T *data,size_t rows, 
                  size_t cols,int in_rank = -1) const
    {
        hid_t datatype = this -> get_hdf5_type<T>();
        this -> __read_data(dataname,data,rows,cols,datatype,in_rank);
    }

    /**
     * @brief write scalar data to hdf5 file
     * 
     * @param name data name
     * @param scalar scalar data itself
     */
    template<typename T>
    void write_scalar(const std::string &name,const T scalar) const
    {
        hid_t datatype = this -> get_hdf5_type<T>();
        this -> __write_scalar(name,&scalar,datatype);
    }

    /**
     * @brief write scalar data to hdf5 file
     * 
     * @param name data name
     * @param scalar scalar data itself
     */
    template<typename T>
    void read_scalar(const std::string &name, T *scalar) const
    {
        hid_t datatype = this -> get_hdf5_type<T>();
        this -> __read_scalar(name,scalar,datatype,sizeof(T));
    }

    /**
     * @brief write attribute into a group or a dataset
     * @tparam T typename numerical numbers
     * @param attrname name of this attribute
     * @param data data
     * @param n size of data
     * @param vname location of this attribute, could be a group or a dataset
     */
    template<typename T>
    void write_attr(const std::string &attrname,const T *data,size_t n,
                    const std::string &vname ="./") const
    {
        static_assert(std::is_arithmetic<T>::value || std::is_same<T,bool>::value);
        this -> __write_attr(attrname,data,n,this->get_hdf5_type<T>(),vname);
    }
    void write_attr(const std::string &attrname,const std::string &info,
                    const std::string &vname ="./") const;

    void read_attr(const std::string &attrname,void *data,size_t n,
                  const std::string &vname ="./") const;
    void read_attr(const std::string &attrname,std::string &info,
                    const std::string &vname ="./") const;


private:
    void __write_data(const std::string &dataname,const void *data,size_t rows,
                    size_t cols,hid_t datatype,int out_rank=-1) const ;
    void __write_scalar(const std::string &name,const void *data,hid_t datatype) const;

    void __read_data(const std::string &dataname,void *data,size_t rows,
                    size_t cols,hid_t datatype,int in_rank=-1) const;
    void __read_scalar(const std::string &name,void *data,hid_t datatype,size_t size) const;
    void __write_attr(const std::string &attrname,const void *data,size_t n,
                      hid_t datatype,const std::string &vname ="./") const;
};

#endif