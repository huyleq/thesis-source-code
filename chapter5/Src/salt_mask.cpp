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

    float *v=new float[nxyz];read("v",v,nxyz);
    float *eps=new float[nxyz];read("eps",eps,nxyz);
    float *del=new float[nxyz];read("del",del,nxyz);
    float *salt_mask=new float[nxyz];

    float salt_v,salt_eps,salt_del;
    get_param("salt_v",salt_v,"salt_eps",salt_eps,"salt_del",salt_del);
    
    #pragma omp parallel for
    for(size_t i=0;i<nxyz;i++){
        if(v[i]>=salt_v && eps[i]<=salt_eps && del[i]<=salt_del) salt_mask[i]=0.f;
        else salt_mask[i]=1.f;
    }
    
    write("mask",salt_mask,nxyz);
    to_header("mask","n1",nx,"o1",ox,"d1",dx);
    to_header("mask","n2",ny,"o2",oy,"d2",dy);
    to_header("mask","n3",nz,"o3",oz,"d3",dz);

    delete []v;delete []eps;delete []del;delete []salt_mask;

    myio_close();
    return 0;
}

