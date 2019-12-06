#include "myio.h"
#include "boundary.h"

using namespace std;

int main(int argc,char ** argv){
    myio_init(argc,argv);
    
    int nx,ny,nz,npad;
    float ox,oy,oz,ot,dx,dy,dz;
    
    from_header("vin","n1",nx,"o1",ox,"d1",dx);
    from_header("vin","n2",ny,"o2",oy,"d2",dy);
    from_header("vin","n3",nz,"o3",oz,"d3",dz);
    get_param("npad",npad);
    
    long long nxy=nx*ny;
    long long nxyz=nxy*nz;
    long long nboundary=nx*ny*nz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);

    float *v=new float[nxyz];
    read("vin",v,nxyz);

    float *boundary=new float[nboundary];
    read("boundary",boundary,nboundary);

    putBoundary(boundary,v,nx,ny,nz,npad);
  
    write("vout",v,nxyz);
    to_header("vout","n1",nx,"o1",ox,"d1",dx);
    to_header("vout","n2",ny,"o2",oy,"d2",dy);
    to_header("vout","n3",nz,"o3",oz,"d3",dz);

    delete []v;
    delete []boundary;

    myio_close();
    return 0;
}
