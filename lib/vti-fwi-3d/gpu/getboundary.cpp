#include "myio.h"
#include "boundary.h"

using namespace std;

int main(int argc,char ** argv){
    myio_init(argc,argv);
    
    int nx,ny,nz,npad;
    float ox,oy,oz,ot,dx,dy,dz;
    
    from_header("v","n1",nx,"o1",ox,"d1",dx);
    from_header("v","n2",ny,"o2",oy,"d2",dy);
    from_header("v","n3",nz,"o3",oz,"d3",dz);
    get_param("npad",npad);
    
    long long nxyz=nx*ny*nz;
    long long nboundary=nx*ny*nz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);

    float *v=new float[nxyz];
    read("v",v,nxyz);

    float *boundary=new float[nboundary];

    getBoundary(boundary,v,nx,ny,nz,npad);
   
    write("boundary",boundary,nboundary);
    to_header("boundary","n1",nboundary,"o1",0.,"d1",1.);

    delete []v;
    delete []boundary;

    myio_close();
    return 0;
}
