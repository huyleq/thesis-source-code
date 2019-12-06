#include <iostream>
#include <cmath>

#include "myio.h"

#include "gaussian_filter.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
//    int sigma=5;
//    int width=4*sigma+1,width2=width*width,width3=width2*width;
//
//    float *filter=new float[width3];
//    
//    init_filter(filter,sigma);
//    
//    write("filter",filter,width3);
//    to_header("filter","n1",width,"o1",0,"d1",1);
//    to_header("filter","n2",width,"o2",0,"d2",1);
//    to_header("filter","n3",width,"o3",0,"d3",1);

    int nx,ny,nz;
    float ox,oy,oz,dx,dy,dz;
    from_header("x","n1",nx,"o1",ox,"d1",dx);
    from_header("x","n2",ny,"o2",oy,"d2",dy);
    from_header("x","n3",nz,"o3",oz,"d3",dz);
    
    long long nxy=nx*ny,nxyz=nxy*nz;

    float *x=new float[nxyz]; read("x",x,nxyz);
    float *fx=new float[nxyz];

//    apply_filter(fx,x,nx,ny,nz,filter,sigma);
    int npad=28,nwbottom=3;
    float max_depth=2000.f,max_sigma=50.f;
    get_param("max_depth",max_depth,"max_sigma",max_sigma);
    int max_iz=(max_depth-oz)/dz+1;
    smooth_gradient(x,nx,ny,nz,npad,nwbottom,max_iz,max_sigma,dx);

    write("fx",x,nxyz);
    to_header("fx","n1",nx,"o1",ox,"d1",dx);
    to_header("fx","n2",ny,"o2",oy,"d2",dy);
    to_header("fx","n3",nz,"o3",oz,"d3",dz);

//    delete []filter; 
    delete []x;delete []fx;
    myio_close();
    return 0;
}
