#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "myio.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
    float ox,oy,oz,dx,dy,dz;
    int nx,ny,nz;
    
    from_header("vnmo","n1",nx,"o1",ox,"d1",dx);
    from_header("vnmo","n2",ny,"o2",oy,"d2",dy);
    from_header("vnmo","n3",nz,"o3",oz,"d3",dz);
    
    long long nxy=nx*ny;
    long long nxyz=nxy*nz;

    float *vnmo=new float[nxyz]; read("vnmo",vnmo,nxyz);
    float *vzin=new float[nxyz]; read("vzin",vzin,nxyz);
    float *salt_mask=new float[nxyz]; read("mask",salt_mask,nxyz);

    #pragma omp parallel for
    for(size_t i=0;i<nxyz;i++){
        if(salt_mask[i]==0.f) vzin[i]=vnmo[i];
    }
    
    write("vzout",vzin,nxyz);
    to_header("vzout","n1",nx,"o1",ox,"d1",dx);
    to_header("vzout","n2",ny,"o2",oy,"d2",dy);
    to_header("vzout","n3",nz,"o3",oz,"d3",dz);

    delete []vnmo;delete []salt_mask;delete []vzin;

    myio_close();
    return 0;
}

