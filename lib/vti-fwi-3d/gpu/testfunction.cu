#include <cstdlib>
#include <cstdio>
#include <vector>

#include "myio.h"

#define BLOCK_DIM 16

__global__ void testkernel(float *v,int *shotid,int nshot,int nx,int ny,int nz){
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    if(ix<nx && iy<ny){
        for(int iz=0;iz<nz;iz++){
            int i=ix+iy*nx+iz*nx*ny;
            for(int is=0;is<nshot;is++){
                v[i]+=shotid[is];
            }
        }
    }
}

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
    int nx,ny,nz;
    float ox,oy,oz,dx,dy,dz;
    
    from_header("v","n1",nx,"o1",ox,"d1",dx);
    from_header("v","n2",ny,"o2",oy,"d2",dy);
    from_header("v","n3",nz,"o3",oz,"d3",dz);

    long long nxyz=nx*ny*nz;

    vector<int> shotid;
    get_array("shotid",shotid);
    int nshot=shotid.size();

    double objfunc=0.;
    for(int i=0;i<nshot;i++) objfunc+=2*shotid[i];
    fprintf(stderr,"objfunc=%.10f\n",objfunc);

    float *v=new float[nxyz]; read("v",v,nxyz);
    
//    #pragma omp parallel for
//    for(int i=0;i<nxyz;i++){
//        for(int is=0;is<nshot;is++) v[i]+=shotid[is];
//    }

    int *d_shotid;
    cudaMalloc(&d_shotid,nshot*sizeof(int));
    cudaMemcpy(d_shotid,shotid.data(),nshot*sizeof(int),cudaMemcpyHostToDevice);

    float *d_v;
    cudaMalloc(&d_v,nxyz*sizeof(float));
    cudaMemcpy(d_v,v,nxyz*sizeof(float),cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM,BLOCK_DIM);
    dim3 grid((nx+BLOCK_DIM-1)/BLOCK_DIM,(ny+BLOCK_DIM-1)/BLOCK_DIM);

    testkernel<<<grid,block>>>(d_v,d_shotid,nshot,nx,ny,nz);

    cudaMemcpy(v,d_v,nxyz*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_v);
    cudaFree(d_shotid);
    
    write("g",v,nxyz);
    to_header("g","n1",nx,"o1",ox,"d1",dx);
    to_header("g","n2",ny,"o2",oy,"d2",dy);
    to_header("g","n3",nz,"o3",oz,"d3",dz);

    delete []v;

    myio_close();
    return 0;
}
