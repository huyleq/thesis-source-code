#include <cmath>
#include <cstdlib>
#include <cstdio>

#include "myio.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
    int nx,nz;
    float ox,oz,dx,dz;

    from_header("vp","n1",nx,"o1",ox,"d1",dx);
    from_header("vp","n2",nz,"o2",oz,"d2",dz);

    int nxz=nx*nz;
    float *vp=new float[nxz]; read("vp",vp,nxz);
    float *vs=new float[nxz];
    float *rho=new float[nxz];

    #pragma omp parallel for
    for(int i=0;i<nxz;i++){
        if(fabs(vp[i]-1500.f)<1.f){
            vs[i]=0.f;
            rho[i]=1000.f;
        }
        else{
            vs[i]=0.862f*vp[i]-1172.f; //mudrock line Castagna 1985
            rho[i]=310.f*sqrt(sqrt(vp[i])); //Gardner
        }
    }
    
    write("vs",vs,nxz);
    to_header("vs","n1",nx,"o1",ox,"d1",dx);
    to_header("vs","n2",nz,"o2",oz,"d2",dz);

    write("rho",rho,nxz);
    to_header("rho","n1",nx,"o1",ox,"d1",dx);
    to_header("rho","n2",nz,"o2",oz,"d2",dz);

    delete []vp;delete []vs;delete []rho;

    myio_close();
    return 0;
}
