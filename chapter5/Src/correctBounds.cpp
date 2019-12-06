#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "myio.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
    float ox,oy,oz,dx,dy,dz;
    int nx,ny,nz;
    
    from_header("v","n1",nx,"o1",ox,"d1",dx);
    from_header("v","n2",ny,"o2",oy,"d2",dy);
    from_header("v","n3",nz,"o3",oz,"d3",dz);
    
    long long nxy=nx*ny;
    long long nxyz=nxy*nz;

    float *v=new float[nxyz]; read("v",v,nxyz);
    float *u=new float[nxyz]; read("upper",u,nxyz);
    float *l=new float[nxyz]; read("lower",l,nxyz);

    float minu,minl;
    get_param("minup",minu,"minlow",minl);

    #pragma omp parallel for
    for(size_t i=0;i<nxyz;i++){
        if(u[i]-v[i]<minu) u[i]=v[i]+minu;
        if(v[i]-l[i]<minl) l[i]=v[i]-minl;
    }
    
    write("upperout",u,nxyz);
    to_header("upperout","n1",nx,"o1",ox,"d1",dx);
    to_header("upperout","n2",ny,"o2",oy,"d2",dy);
    to_header("upperout","n3",nz,"o3",oz,"d3",dz);

    write("lowerout",l,nxyz);
    to_header("lowerout","n1",nx,"o1",ox,"d1",dx);
    to_header("lowerout","n2",ny,"o2",oy,"d2",dy);
    to_header("lowerout","n3",nz,"o3",oz,"d3",dz);

    delete []v;delete []u;delete []l;

    myio_close();
    return 0;
}

