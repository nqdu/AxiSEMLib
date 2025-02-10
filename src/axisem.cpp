#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<iostream>
#include <tuple>
namespace py = pybind11;
using py::arg;

typedef py::array_t<double> dmat;
typedef py::array_t<int> imat;

extern "C"{

// base on nanqiao's surfdisp.hpp
// That's the cpp warpper , packing the fortran code

/**
 * computer the rf 
**/
void inside_element(double s, double z,const double *nodes, 
                   int element_type, double tolerance, bool *in_element, 
                   double *xi, double *eta); 

void strain_monopole_td(const double *u, const double *G, const double *GT, 
                        const double *xi, const double *eta, int npol, int nsamp, 
                        const double *nodes, int element_type, 
                        bool axial, double *strain_tensor);


void strain_dipole_td(const double *u, const double *G, const double *GT, 
                        const double *xi, const double *eta, int npol, int nsamp, 
                        const double *nodes, int element_type, 
                        bool axial, double *strain_tensor);

void strain_quadpole_td(const double *u, const double *G, const double *GT, 
                        const double *xi, const double *eta, int npol, int nsamp, 
                        const double *nodes, int element_type, 
                        bool axial, double *strain_tensor);

void lagrange_interpol_2D_td(int N, int nsamp, const double *points1, 
                                    const double *points2,const double *coefficients, 
                                    double x1, double x2, double *interpolant);

double get_theta(double xi, double eta, const double *nodes, int element_type);

}

std::tuple<bool,double,double>
inside_element_warp(double s, double z,const dmat &nodes, 
                   int element_type, double tolerance)
{
    double xi,eta;
    bool in_element;
    inside_element(s,z,nodes.data(),element_type,tolerance,&in_element,&xi,&eta);

    return std::make_tuple(in_element,xi,eta); 
}

dmat strain_td_warp(const dmat &u,const dmat &G,const dmat &GT,const dmat &xi,
                            const dmat &eta,const dmat &nodes,int elem_type,
                            bool axial,const std::string &stype="monopole")
{
    int npol,nsamp; 
    npol = u.unchecked<4>().shape(1) - 1;
    nsamp = u.unchecked<4>().shape(0);
    dmat strain; strain.resize({nsamp*(npol+1)*(npol+1)*6});

    // compute 
    if(stype == "monopole") {
        strain_monopole_td(u.data(),G.data(),GT.data(),xi.data(),eta.data(),
                            npol,nsamp,nodes.data(),elem_type,axial,strain.mutable_data());
    }
    else if(stype == "dipole"){
        strain_dipole_td(u.data(),G.data(),GT.data(),xi.data(),eta.data(),
                            npol,nsamp,nodes.data(),elem_type,axial,strain.mutable_data());
    }
    else {
        strain_quadpole_td(u.data(),G.data(),GT.data(),xi.data(),eta.data(),
                            npol,nsamp,nodes.data(),elem_type,axial,strain.mutable_data());
    }
    return strain;
}

dmat lagrange_2D(const dmat &points1,const dmat &points2,
                   const dmat &coefficients, double x1, 
                   double x2)
{
    int npol = points1.size() - 1;
    int nsamp = coefficients.unchecked<3>().shape(0);

    dmat out; out.resize({nsamp});
    lagrange_interpol_2D_td(npol,nsamp,points1.data(),points2.data(),
                                    coefficients.data(),x1,x2,out.mutable_data());

    return out;
}

dmat find_theta(const dmat &xi, const dmat &eta,const dmat &nodes,int eltype)
{
    int ngll = xi.size();
    py::array_t<double,py::array::f_style> theta; theta.resize({ngll,ngll});
    auto theta0 = theta.mutable_unchecked<2>();
    auto xi0 = xi.unchecked<1>();
    auto eta0 = eta.unchecked<1>();
    for(int i = 0; i < ngll; i ++ ) {
        for(int j = 0; j < ngll; j ++) {
            theta0(i,j) = get_theta(xi0(i),eta0(j),nodes.data(),eltype);
        }
    }

    return theta;
}

PYBIND11_MODULE(libsem,m){
    m.doc() = "Axisem functions\n";
    m.def("inside_element_warp",&inside_element_warp,arg("s"), arg("z"), arg("nodes"),
          arg("element_type"), arg("tolerance"),
          "locate element wrapper");
    m.def("strain_td_warp",&strain_td_warp,arg("u"), arg("G"), arg("GT"),
          arg("xi"), arg("eta"), arg("nodes"),
          arg("elem_type") ,arg("axial"),arg("stype")="monopole",
          "compute strain c++ wrapper");

    m.def("lagrange_2D",&lagrange_2D,arg("points1"), arg("points2"), arg("coefs"),
          arg("x1"), arg("x2"),"lagrange_2D c++ wrapper");

    m.def("find_theta",&find_theta,arg("xi"),arg("eta"),arg("nodes"),arg("eltype"));
}

