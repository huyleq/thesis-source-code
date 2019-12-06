#include "myio.h"
#include "mylib.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
    int nx,nz,n3=1;
    float ox,oz,dx,dz;
    from_header("v","n1",nx,"o1",ox,"d1",dx);
    from_header("v","n2",nz,"o2",oz,"d2",dz);
    from_header("v","n3",n3);

    int nxz=nx*nz,nxz3=nxz*n3;
    float *v=new float[nxz3]; read("v",v,nxz3);
    float *initial=new float[nxz]; read("initial",initial,nxz);
    
    for(int k=0;k<n3;k++) subtract(v+k*nxz,v+k*nxz,initial,nxz);

    write("update",v,nxz3);
    to_header("update","n1",nx,"o1",ox,"d1",dx);
    to_header("update","n2",nz,"o2",oz,"d2",dz);
    to_header("update","n3",n3);

    delete []v;delete []initial;

    myio_close();
    return 0;
}
