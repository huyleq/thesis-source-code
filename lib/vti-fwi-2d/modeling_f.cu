#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "mylib.h"
#include "init.h"
#include "wave.h"
#include "kernels.h"

void modeling_f(float *wavefield,const float *c11,const float *c13,const float *c33,const float *wavelet,int slocxz,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot){
 fprintf(stderr,"Modeling\n");

 int ratio=std::round(rate/dt);
 int ntNeg=std::round(abs(ot)/dt);
 int nnx=nx+2*npad,nnz=nz+2*npad;
 int nnxz=nnx*nnz;
 float dx2=dx*dx,dz2=dz*dz,dt2=dt*dt;
 
 //cudaSetDevice(5);

 float *d_wavefieldSlice; 
 cudaMalloc(&d_wavefieldSlice,nnxz*sizeof(float));
 cudaMemset(d_wavefieldSlice,0,nnxz*sizeof(float));
 
 float *d_c11; cudaMalloc(&d_c11,nnxz*sizeof(float));
 float *d_c13; cudaMalloc(&d_c13,nnxz*sizeof(float));
 float *d_c33; cudaMalloc(&d_c33,nnxz*sizeof(float));
 cudaMemcpy(d_c11,c11,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 cudaMemcpy(d_c13,c13,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 cudaMemcpy(d_c33,c33,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 
 float *d_taper; cudaMalloc(&d_taper,nnxz*sizeof(float));
 cudaMemcpy(d_taper,taper,nnxz*sizeof(float),cudaMemcpyHostToDevice);

 float *p0; cudaMalloc(&p0,nnxz*sizeof(float)); 
 float *p1; cudaMalloc(&p1,nnxz*sizeof(float)); 
 float *q0; cudaMalloc(&q0,nnxz*sizeof(float)); 
 float *q1; cudaMalloc(&q1,nnxz*sizeof(float)); 

 dim3 block(BLOCK_DIM_X,BLOCK_DIM_Y);
 dim3 grid((nnx-2*RADIUS+BLOCK_DIM_X-1)/BLOCK_DIM_X,(nnz-2*RADIUS+BLOCK_DIM_Y-1)/BLOCK_DIM_Y);

 cudaMemset(p0,0,nnxz*sizeof(float));
 cudaMemset(q0,0,nnxz*sizeof(float));
 cudaMemset(p1,0,nnxz*sizeof(float));
 cudaMemset(q1,0,nnxz*sizeof(float));
  
 cudaError_t e=cudaGetLastError();
 if(e!=cudaSuccess) fprintf(stderr,"error %s\n",cudaGetErrorString(e));

 injectSource<<<1,1>>>(p1,q1,dt2*wavelet[0],slocxz);

 abc<<<grid,block>>>(p1,q1,d_taper,nnx,nnz);
 
 if(ratio==1 && ot==0.f){
  recordWavefieldSlice<<<grid,block>>>(d_wavefieldSlice,p1,q1,nnx,nnz);
  cudaMemcpy(wavefield+nnxz,d_wavefieldSlice,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(wavefield+nnxz,p1,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
 }
 
 for(int it=2;it<nt;++it){
  float t=it*dt+ot;

  forwardCD<<<grid,block>>>(p0,q0,p1,q1,d_c11,d_c13,d_c33,dx2,dz2,dt2,nnx,nnz);

  injectSource<<<1,1>>>(p0,q0,wavelet[it-1],slocxz);

  abc<<<grid,block>>>(p1,q1,p0,q0,d_taper,nnx,nnz);
  
  if(t>=0.f && (it-ntNeg)%ratio==0){
   recordWavefieldSlice<<<grid,block>>>(d_wavefieldSlice,p0,q0,nnx,nnz);
   cudaMemcpy(wavefield+((it-ntNeg)/ratio)*nnxz,d_wavefieldSlice,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(wavefield+((it-ntNeg)/ratio)*nnxz,p0,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
  }

  float *pt=p0; 
  p0=p1;
  p1=pt;
  pt=q0;
  q0=q1;
  q1=pt;
 }

 cudaFree(d_wavefieldSlice);
 cudaFree(d_c11);cudaFree(d_c13);cudaFree(d_c33);
 cudaFree(d_taper);
 cudaFree(p0);cudaFree(p1);cudaFree(q0);cudaFree(q1);
 
 e=cudaGetLastError();
 if(e!=cudaSuccess) fprintf(stderr,"error %s\n",cudaGetErrorString(e));

 return;
}

void modelingR_f(float *wavefield,const float *r11,const float *r13,const float *r33,const float *wavelet,int slocxz,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot){
 fprintf(stderr,"Modeling\n");

 int ratio=std::round(rate/dt);
 int ntNeg=std::round(abs(ot)/dt);
 int nnx=nx+2*npad,nnz=nz+2*npad;
 int nnxz=nnx*nnz;
 float dx2=dx*dx,dz2=dz*dz,dt2=dt*dt;
 
 cudaSetDevice(5);

 float *d_wavefieldSlice; 
 cudaMalloc(&d_wavefieldSlice,nnxz*sizeof(float));
 cudaMemset(d_wavefieldSlice,0,nnxz*sizeof(float));
 
 float *d_r11; cudaMalloc(&d_r11,nnxz*sizeof(float));
 float *d_r13; cudaMalloc(&d_r13,nnxz*sizeof(float));
 float *d_r33; cudaMalloc(&d_r33,nnxz*sizeof(float));
 cudaMemcpy(d_r11,r11,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 cudaMemcpy(d_r13,r13,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 cudaMemcpy(d_r33,r33,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 
 float *d_taper; cudaMalloc(&d_taper,nnxz*sizeof(float));
 cudaMemcpy(d_taper,taper,nnxz*sizeof(float),cudaMemcpyHostToDevice);

 float *p0; cudaMalloc(&p0,nnxz*sizeof(float)); 
 float *p1; cudaMalloc(&p1,nnxz*sizeof(float)); 
 float *q0; cudaMalloc(&q0,nnxz*sizeof(float)); 
 float *q1; cudaMalloc(&q1,nnxz*sizeof(float)); 

 dim3 block(BLOCK_DIM_X,BLOCK_DIM_Y);
 dim3 grid((nnx-2*RADIUS+BLOCK_DIM_X-1)/BLOCK_DIM_X,(nnz-2*RADIUS+BLOCK_DIM_Y-1)/BLOCK_DIM_Y);

 cudaMemset(p0,0,nnxz*sizeof(float));
 cudaMemset(q0,0,nnxz*sizeof(float));
 cudaMemset(p1,0,nnxz*sizeof(float));
 cudaMemset(q1,0,nnxz*sizeof(float));
  
 cudaError_t e=cudaGetLastError();
 if(e!=cudaSuccess) fprintf(stderr,"error %s\n",cudaGetErrorString(e));

 injectSource<<<1,1>>>(p1,q1,dt2*wavelet[0],slocxz);

 abc<<<grid,block>>>(p1,q1,d_taper,nnx,nnz);
 
 if(ratio==1 && ot==0.f){
  recordWavefieldSlice<<<grid,block>>>(d_wavefieldSlice,p1,q1,nnx,nnz);
  cudaMemcpy(wavefield+nnxz,d_wavefieldSlice,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(wavefield+nnxz,p1,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
 }
 
 for(int it=2;it<nt;++it){
  float t=it*dt+ot;

  forwardRDR<<<grid,block>>>(p0,q0,p1,q1,d_r11,d_r13,d_r33,dx2,dz2,dt2,nnx,nnz);

  injectSource<<<1,1>>>(p0,q0,wavelet[it-1],slocxz);

  abc<<<grid,block>>>(p1,q1,p0,q0,d_taper,nnx,nnz);
  
  if(t>=0.f && (it-ntNeg)%ratio==0){
   recordWavefieldSlice<<<grid,block>>>(d_wavefieldSlice,p0,q0,nnx,nnz);
   cudaMemcpy(wavefield+((it-ntNeg)/ratio)*nnxz,d_wavefieldSlice,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(wavefield+((it-ntNeg)/ratio)*nnxz,p0,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
  }

  float *pt=p0; 
  p0=p1;
  p1=pt;
  pt=q0;
  q0=q1;
  q1=pt;
 }

 cudaFree(d_wavefieldSlice);
 cudaFree(d_r11);cudaFree(d_r13);cudaFree(d_r33);
 cudaFree(d_taper);
 cudaFree(p0);cudaFree(p1);cudaFree(q0);cudaFree(q1);
 
 e=cudaGetLastError();
 if(e!=cudaSuccess) fprintf(stderr,"error %s\n",cudaGetErrorString(e));

 return;
}

void modelingABCD_f(float *wavefield,const float *a1,const float *b1c1,const float *d1,const float *a2,const float *b2c2,const float *d2,const float *wavelet,int slocxz,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot){
 fprintf(stderr,"Modeling\n");

 int ratio=std::round(rate/dt);
 int ntNeg=std::round(abs(ot)/dt);
 int nnx=nx+2*npad,nnz=nz+2*npad;
 int nnxz=nnx*nnz;
 float dx2=dx*dx,dz2=dz*dz,dt2=dt*dt;
 
 cudaSetDevice(5);

 float *d_wavefieldSlice; 
 cudaMalloc(&d_wavefieldSlice,nnxz*sizeof(float));
 cudaMemset(d_wavefieldSlice,0,nnxz*sizeof(float));
 
 float *d_a1; cudaMalloc(&d_a1,nnxz*sizeof(float));
 float *d_b1c1; cudaMalloc(&d_b1c1,nnxz*sizeof(float));
 float *d_d1; cudaMalloc(&d_d1,nnxz*sizeof(float));
 cudaMemcpy(d_a1,a1,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 cudaMemcpy(d_b1c1,b1c1,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 cudaMemcpy(d_d1,d1,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 
 float *d_a2; cudaMalloc(&d_a2,nnxz*sizeof(float));
 float *d_b2c2; cudaMalloc(&d_b2c2,nnxz*sizeof(float));
 float *d_d2; cudaMalloc(&d_d2,nnxz*sizeof(float));
 cudaMemcpy(d_a2,a2,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 cudaMemcpy(d_b2c2,b2c2,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 cudaMemcpy(d_d2,d2,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 
 float *d_taper; cudaMalloc(&d_taper,nnxz*sizeof(float));
 cudaMemcpy(d_taper,taper,nnxz*sizeof(float),cudaMemcpyHostToDevice);

 float *p0; cudaMalloc(&p0,nnxz*sizeof(float)); 
 float *p1; cudaMalloc(&p1,nnxz*sizeof(float)); 
 float *q0; cudaMalloc(&q0,nnxz*sizeof(float)); 
 float *q1; cudaMalloc(&q1,nnxz*sizeof(float)); 

 dim3 block(BLOCK_DIM_X,BLOCK_DIM_Y);
 dim3 grid((nnx-2*RADIUS+BLOCK_DIM_X-1)/BLOCK_DIM_X,(nnz-2*RADIUS+BLOCK_DIM_Y-1)/BLOCK_DIM_Y);

 cudaMemset(p0,0,nnxz*sizeof(float));
 cudaMemset(q0,0,nnxz*sizeof(float));
 cudaMemset(p1,0,nnxz*sizeof(float));
 cudaMemset(q1,0,nnxz*sizeof(float));
  
 cudaError_t e=cudaGetLastError();
 if(e!=cudaSuccess) fprintf(stderr,"error %s\n",cudaGetErrorString(e));

 injectSource<<<1,1>>>(p1,q1,dt2*wavelet[0],slocxz);

 abc<<<grid,block>>>(p1,q1,d_taper,nnx,nnz);
 
 if(ratio==1 && ot==0.f){
  recordWavefieldSlice<<<grid,block>>>(d_wavefieldSlice,p1,q1,nnx,nnz);
  cudaMemcpy(wavefield+nnxz,d_wavefieldSlice,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(wavefield+nnxz,p1,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
 }
 
 for(int it=2;it<nt;++it){
  float t=it*dt+ot;

  forwardABCD<<<grid,block>>>(p0,q0,p1,q1,d_a1,d_b1c1,d_d1,d_a2,d_b2c2,d_d2,dx2,dz2,dt2,nnx,nnz);

  injectSource<<<1,1>>>(p0,q0,wavelet[it-1],slocxz);

  abc<<<grid,block>>>(p1,q1,p0,q0,d_taper,nnx,nnz);
  
  if(t>=0.f && (it-ntNeg)%ratio==0){
   recordWavefieldSlice<<<grid,block>>>(d_wavefieldSlice,p0,q0,nnx,nnz);
//   cudaMemcpy(wavefield+((it-ntNeg)/ratio)*nnxz,d_wavefieldSlice,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(wavefield+((it-ntNeg)/ratio)*nnxz,q0,nnxz*sizeof(float),cudaMemcpyDeviceToHost);
  }

  float *pt=p0; 
  p0=p1;
  p1=pt;
  pt=q0;
  q0=q1;
  q1=pt;
 }

 cudaFree(d_wavefieldSlice);
 cudaFree(d_a1);cudaFree(d_b1c1);cudaFree(d_d1);
 cudaFree(d_a2);cudaFree(d_b2c2);cudaFree(d_d2);
 cudaFree(d_taper);
 cudaFree(p0);cudaFree(p1);cudaFree(q0);cudaFree(q1);
 
 e=cudaGetLastError();
 if(e!=cudaSuccess) fprintf(stderr,"error %s\n",cudaGetErrorString(e));

 return;
}

