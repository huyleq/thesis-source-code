#include <cstdlib>
#include <cstdio>

#include "myio.h"
#include "mylib.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    int nx,nz,niter;
    from_header("invertedmodel","n1",nx,"n2",nz,"n3",niter);
    int nxz=nx*nz;
    float *imodel=new float[nxz*niter];
    read("invertedmodel",imodel,nxz*niter);
    float *tmodel=new float[nxz];
    read("truemodel",tmodel,nxz);
    float *mres=new float[niter];
    for(int i=0;i<niter;i++){
        subtract(imodel+i*nxz,imodel+i*nxz,tmodel,nxz);
        mres[i]=dot_product(imodel+i*nxz,imodel+i*nxz,nxz);
    }
    write("modelres",mres,niter);
    to_header("modelres","n1",niter,"o1",0,"d1",1);
    delete []imodel;delete []tmodel;delete []mres;
    myio_close();
    return 0;
}
