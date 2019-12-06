#include "myio.h"
#include "boundary.h"

using namespace std;

int main(int argc,char ** argv){
    myio_init(argc,argv);
    
    int nx0,ny0,nz0,npad;
    float ox,oy,oz,ot,dx,dy,dz;
    
    from_header("v","n1",nx0,"o1",ox,"d1",dx);
    from_header("v","n2",ny0,"o2",oy,"d2",dy);
    from_header("v","n3",nz0,"o3",oz,"d3",dz);
    get_param("npad",npad);

    int nx=nx0+2*npad,ny=ny0+2*npad,nz=nz0+2*npad;
    
    long long nxyz0=nx0*ny0*nz0;
    float *v=new float[nxyz0];
    long long nxyz=nx*ny*nz;
    float *vpad=new float[nxyz];

    read("v",v,nxyz0);
    pad3d(vpad,v,nx,ny,nz,npad);

    write("vpad",vpad,nxyz);
    to_header("vpad","n1",nx,"o1",ox-npad*dx,"d1",dx);
    to_header("vpad","n2",ny,"o2",oy-npad*dy,"d2",dy);
    to_header("vpad","n3",nz,"o3",oz-npad*dz,"d3",dz);

    delete []v;
    delete []vpad;

    myio_close();
    return 0;
}
