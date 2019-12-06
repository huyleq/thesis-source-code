#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "init.h"
#include "myio.h"
#include "wave.h"
#include "kernels.h"

#include <vector>

void synthetic_f(float *data,const float *c11,const float *c13,const float *c33,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot){
 int ratio=rate/dt+0.5;
 int ntNeg=std::round(abs(ot)/dt);
 int nnx=nx+2*npad,nnz=nz+2*npad;
 int nnxz=nnx*nnz;
 float dx2=dx*dx,dz2=dz*dz,dt2=dt*dt;
 
 std::vector<int> GPUs;
 get_array("gpu",GPUs);
 int nGPUs=GPUs.size();
 fprintf(stderr,"Total # GPUs = %d\n",nGPUs);
 fprintf(stderr,"GPUs used are:\n");
 for(int i=0;i<nGPUs;i++) fprintf(stderr,"%d",GPUs[i]);
 fprintf(stderr,"\n");

 int **d_rloc=new int*[nGPUs]();
 float **d_c11=new float*[nGPUs]();
 float **d_c13=new float*[nGPUs]();
 float **d_c33=new float*[nGPUs]();
 float **d_taper=new float*[nGPUs]();
 float **d_data=new float*[nGPUs]();
 float **p0=new float*[nGPUs]();
 float **q0=new float*[nGPUs]();
 float **p1=new float*[nGPUs]();
 float **q1=new float*[nGPUs]();

 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(GPUs[i]);
  
  cudaMalloc(&d_c11[i],nnxz*sizeof(float));
  cudaMalloc(&d_c13[i],nnxz*sizeof(float));
  cudaMalloc(&d_c33[i],nnxz*sizeof(float));
  cudaMemcpy(d_c11[i],c11,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_c13[i],c13,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_c33[i],c33,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  
  cudaMalloc(&d_taper[i],nnxz*sizeof(float));
  cudaMemcpy(d_taper[i],taper,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 
  cudaMalloc(&p0[i],nnxz*sizeof(float)); 
  cudaMalloc(&p1[i],nnxz*sizeof(float)); 
  cudaMalloc(&q0[i],nnxz*sizeof(float)); 
  cudaMalloc(&q1[i],nnxz*sizeof(float)); 
 }
 
 int npasses=(ns+nGPUs-1)/nGPUs;
 int shotLeft=ns;

// chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 for(int pass=0;pass<npasses;++pass){
  int nGPUsNeed=min(shotLeft,nGPUs);
  fprintf(stderr,"Pass %d, # GPUs %d\n",pass,nGPUsNeed);
  #pragma omp parallel for num_threads(nGPUsNeed)
  for(int i=0;i<nGPUsNeed;++i){
  cudaSetDevice(GPUs[i]);

   int is=pass*nGPUs+i;
   int slocxz=sloc[0+is*4]+sloc[1+is*4]*nnx;

   cudaMalloc(&d_rloc[i],2*sloc[2+is*4]*sizeof(int));
   cudaMemcpy(d_rloc[i],rloc+2*sloc[3+is*4],2*sloc[2+is*4]*sizeof(int),cudaMemcpyHostToDevice);
   
   cudaMalloc(&d_data[i],sloc[2+is*4]*sizeof(float));
 
   dim3 block(BLOCK_DIM_X,BLOCK_DIM_Y);
   dim3 grid((nnx-2*RADIUS+BLOCK_DIM_X-1)/BLOCK_DIM_X,(nnz-2*RADIUS+BLOCK_DIM_Y-1)/BLOCK_DIM_Y);

   cudaMemset(p0[i],0,nnxz*sizeof(float));
   cudaMemset(q0[i],0,nnxz*sizeof(float));
   cudaMemset(p1[i],0,nnxz*sizeof(float));
   cudaMemset(q1[i],0,nnxz*sizeof(float));

   injectSource<<<1,1>>>(p1[i],q1[i],dt2*wavelet[0],slocxz);
//   injectDipoleSource<<<1,1>>>(p1[i],q1[i],dt2*wavelet[0],slocxz,nnx);
  
   abc<<<grid,block>>>(p1[i],q1[i],d_taper[i],nnx,nnz);
   
   if(ratio==1 && ot==0.){
    recordData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X>>>(d_data[i],p1[i],q1[i],d_rloc[i],sloc[2+is*4],nnx);
//    recordDipoleData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X>>>(d_data[i],p1[i],q1[i],d_rloc[i],sloc[2+is*4],nnx);
    cudaMemcpy(data+nr+sloc[3+is*4],d_data[i],sloc[2+is*4]*sizeof(float),cudaMemcpyDeviceToHost);
   }
   
   for(int it=2;it<nt;++it){
    float t=it*dt+ot;
  
    forwardCD<<<grid,block>>>(p0[i],q0[i],p1[i],q1[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
  
    injectSource<<<1,1>>>(p0[i],q0[i],dt2*wavelet[it-1],slocxz);
//    injectDipoleSource<<<1,1>>>(p0[i],q0[i],dt2*wavelet[it-1],slocxz,nnx);
  
    abc<<<grid,block>>>(p1[i],q1[i],p0[i],q0[i],d_taper[i],nnx,nnz);
    
    if(t>=0. && (it-ntNeg)%ratio==0){
     recordData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X>>>(d_data[i],p0[i],q0[i],d_rloc[i],sloc[2+is*4],nnx);
//     recordDipoleData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X>>>(d_data[i],p0[i],q0[i],d_rloc[i],sloc[2+is*4],nnx);
     cudaMemcpy(data+((it-ntNeg)/ratio)*nr+sloc[3+is*4],d_data[i],sloc[2+is*4]*sizeof(float),cudaMemcpyDeviceToHost);
    }
  
    float *pt=p0[i]; 
    p0[i]=p1[i];
    p1[i]=pt;
    pt=q0[i];
    q0[i]=q1[i];
    q1[i]=pt;
   }
   cudaFree(d_data[i]);cudaFree(d_rloc[i]);
  }
  
  shotLeft-=nGPUsNeed;
 }

// chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
// chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
// std::cout<<"total time "<<time.count()<<" seconds"<<endl;
// std::cout<<"cells per second "<<(nnx-8)*(nnz-8)*nt/time.count()<<endl;
 
 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(GPUs[i]);
  cudaFree(d_c11[i]);cudaFree(d_c13[i]);cudaFree(d_c33[i]);
  cudaFree(d_taper[i]);
  cudaFree(p0[i]);cudaFree(p1[i]);cudaFree(q0[i]);cudaFree(q1[i]);
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"gpu %d error %s\n",GPUs[i],cudaGetErrorString(e));
 }
 
 delete []d_rloc;
 delete []d_c11;
 delete []d_c13;
 delete []d_c33;
 delete []d_taper;
 delete []d_data;
 delete []p0;
 delete []q0;
 delete []p1;
 delete []q1;

 return;
}
