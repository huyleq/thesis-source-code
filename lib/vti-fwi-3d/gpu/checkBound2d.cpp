#include "myio.h"

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
    float *upper=new float[nxz]; read("upper",upper,nxz);
    float *lower=new float[nxz]; read("lower",lower,nxz);
    
    for(int k=0;k<n3;k++){
        #pragma omp parallel for
        for(size_t i=0;i<nxz;i++){
            if(v[i+k*nxz]>upper[i]) v[i+k*nxz]=1.f;
            else if(v[i+k*nxz]<lower[i]) v[i+k*nxz]=-1.f;
            else v[i+k*nxz]=0.f;
        }
    }

    write("violate",v,nxz3);
    to_header("violate","n1",nx,"o1",ox,"d1",dx);
    to_header("violate","n2",nz,"o2",oz,"d2",dz);
    to_header("violate","n3",n3);

    delete []v;delete []upper;delete []lower;

    myio_close();
    return 0;
}
