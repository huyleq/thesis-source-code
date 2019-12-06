#include <cuda_profiler_api.h>
#include <omp.h>
#include <iostream>
#include <cstdio>
#include <chrono>
#include "mylib.h"
#include "myio.h"
#include "wave.h"
#include "check.h"
#include "conversions.h"
#include "kernels.h"

#include <vector>

void hessian_f(float *gc11,float *gc13,float *gc33,const float *data,const float *c11,const float *c13,const float *c33,const float *dc11,const float *dc13,const float *dc33,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot){
// fprintf(stderr,"Calculate obj func and grad\n");
 
 int ratio=rate/dt+0.5f;
 int ntNeg=std::round(abs(ot)/dt);
 int nnt=(nt-1)/ratio+1;
 int nnt_data=(nt-ntNeg-1)/ratio+1;
 int nnx=nx+2*npad,nnz=nz+2*npad;
 int nnxz=nnx*nnz;
 float dx2=dx*dx,dz2=dz*dz,dt2=dt*dt;
 
 memset(gc11,0,nnxz*sizeof(float));
 memset(gc13,0,nnxz*sizeof(float));
 memset(gc33,0,nnxz*sizeof(float));
 
 std::vector<int> GPUs;
 get_array("gpu",GPUs);
 int nGPUs=GPUs.size();
// fprintf(stderr,"Total # GPUs = %d\n",nGPUs);
// fprintf(stderr,"GPUs used are:\n");
// for(int i=0;i<nGPUs;i++) fprintf(stderr,"%d",GPUs[i]);
// fprintf(stderr,"\n");

 float **bgdata=new float*[nGPUs]();
 float **borndata=new float*[nGPUs]();
 int **d_rloc=new int*[nGPUs]();
 float **d_c11=new float*[nGPUs]();
 float **d_c13=new float*[nGPUs]();
 float **d_c33=new float*[nGPUs]();
 float **d_dc11=new float*[nGPUs]();
 float **d_dc13=new float*[nGPUs]();
 float **d_dc33=new float*[nGPUs]();
 float **d_taper=new float*[nGPUs]();
 float **d_bgdata=new float*[nGPUs]();
 float **p0=new float*[nGPUs]();
 float **q0=new float*[nGPUs]();
 float **p1=new float*[nGPUs]();
 float **q1=new float*[nGPUs]();
 float **dp0=new float*[nGPUs]();
 float **dq0=new float*[nGPUs]();
 float **dp1=new float*[nGPUs]();
 float **dq1=new float*[nGPUs]();
 float **d_Dpq=new float*[nGPUs]();
 float **d_Ddpq=new float*[nGPUs]();
 float **d_Dpqa=new float*[nGPUs]();
 float **d_Dpqb=new float*[nGPUs]();
 float **d_Ddpqa=new float*[nGPUs]();
 float **d_Ddpqb=new float*[nGPUs]();
 float **Dp=new float*[nGPUs]();
 float **Dq=new float*[nGPUs]();
 float **Ddp=new float*[nGPUs]();
 float **Ddq=new float*[nGPUs]();
 float **Dpqa=new float*[nGPUs]();
 float **Dpqb=new float*[nGPUs]();
 float **Ddpqa=new float*[nGPUs]();
 float **Ddpqb=new float*[nGPUs]();
 float **d_gc11=new float*[nGPUs]();
 float **d_gc13=new float*[nGPUs]();
 float **d_gc33=new float*[nGPUs]();
 float **d_bgdata0=new float*[nGPUs]();
 float **d_bgdata1=new float*[nGPUs]();
 float **d_borndata0=new float*[nGPUs]();
 float **d_borndata1=new float*[nGPUs]();
 float **d_Dp0=new float*[nGPUs]();
 float **d_Dp1=new float*[nGPUs]();
 float **d_Dq0=new float*[nGPUs]();
 float **d_Dq1=new float*[nGPUs]();
 float **d_Ddp0=new float*[nGPUs]();
 float **d_Ddp1=new float*[nGPUs]();
 float **d_Ddq0=new float*[nGPUs]();
 float **d_Ddq1=new float*[nGPUs]();
 float **tgc11=new float*[nGPUs]();
 float **tgc13=new float*[nGPUs]();
 float **tgc33=new float*[nGPUs]();
 float **d_tDpq=new float*[nGPUs]();

 cudaStream_t *compStream=new cudaStream_t[nGPUs];
 cudaStream_t *transfStream=new cudaStream_t[nGPUs];

 #pragma omp parallel for num_threads(nGPUs)
 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(GPUs[i]);
  
  cudaMalloc(&d_c11[i],nnxz*sizeof(float));
  cudaMalloc(&d_c13[i],nnxz*sizeof(float));
  cudaMalloc(&d_c33[i],nnxz*sizeof(float));
  cudaMemcpy(d_c11[i],c11,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_c13[i],c13,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_c33[i],c33,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  
  cudaMalloc(&d_dc11[i],nnxz*sizeof(float));
  cudaMalloc(&d_dc13[i],nnxz*sizeof(float));
  cudaMalloc(&d_dc33[i],nnxz*sizeof(float));
  cudaMemcpy(d_dc11[i],dc11,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_dc13[i],dc13,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_dc33[i],dc33,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  
  cudaMalloc(&d_taper[i],nnxz*sizeof(float));
  cudaMemcpy(d_taper[i],taper,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 
  cudaMalloc(&p0[i],nnxz*sizeof(float)); 
  cudaMalloc(&p1[i],nnxz*sizeof(float)); 
  cudaMalloc(&q0[i],nnxz*sizeof(float)); 
  cudaMalloc(&q1[i],nnxz*sizeof(float)); 
  
  cudaMalloc(&dp0[i],nnxz*sizeof(float)); 
  cudaMalloc(&dp1[i],nnxz*sizeof(float)); 
  cudaMalloc(&dq0[i],nnxz*sizeof(float)); 
  cudaMalloc(&dq1[i],nnxz*sizeof(float)); 
  
  cudaMalloc(&d_Dpq[i],2*nnxz*sizeof(float)); 
  cudaMemset(d_Dpq[i],0,2*nnxz*sizeof(float));
  
  cudaMalloc(&d_Ddpq[i],2*nnxz*sizeof(float)); 
  cudaMemset(d_Ddpq[i],0,2*nnxz*sizeof(float));
  
  cudaMalloc(&d_Dpqa[i],2*nnxz*sizeof(float)); 
  cudaMalloc(&d_Dpqb[i],2*nnxz*sizeof(float)); 
  cudaMemset(d_Dpqa[i],0,2*nnxz*sizeof(float)); 
  cudaMemset(d_Dpqb[i],0,2*nnxz*sizeof(float)); 
  
  cudaMalloc(&d_Ddpqa[i],2*nnxz*sizeof(float)); 
  cudaMalloc(&d_Ddpqb[i],2*nnxz*sizeof(float)); 
  cudaMemset(d_Ddpqa[i],0,2*nnxz*sizeof(float)); 
  cudaMemset(d_Ddpqb[i],0,2*nnxz*sizeof(float)); 
  
  Dp[i]=new float[nnxz*nnt]();
  Dq[i]=new float[nnxz*nnt]();
  
  Ddp[i]=new float[nnxz*nnt]();
  Ddq[i]=new float[nnxz*nnt]();
  
  cudaHostAlloc(&Dpqa[i],2*nnxz*sizeof(float),cudaHostAllocDefault);
  cudaHostAlloc(&Dpqb[i],2*nnxz*sizeof(float),cudaHostAllocDefault);
  cudaMemset(Dpqa[i],0,2*nnxz*sizeof(float));
  cudaMemset(Dpqb[i],0,2*nnxz*sizeof(float));
   
  cudaHostAlloc(&Ddpqa[i],2*nnxz*sizeof(float),cudaHostAllocDefault);
  cudaHostAlloc(&Ddpqb[i],2*nnxz*sizeof(float),cudaHostAllocDefault);
  cudaMemset(Ddpqa[i],0,2*nnxz*sizeof(float));
  cudaMemset(Ddpqb[i],0,2*nnxz*sizeof(float));
   
  cudaStreamCreate(&compStream[i]); 
  cudaStreamCreate(&transfStream[i]); 
  
  cudaMalloc(&d_gc11[i],nnxz*sizeof(float));
  cudaMalloc(&d_gc13[i],nnxz*sizeof(float));
  cudaMalloc(&d_gc33[i],nnxz*sizeof(float));
  
  cudaMemset(d_gc11[i],0,nnxz*sizeof(float));
  cudaMemset(d_gc13[i],0,nnxz*sizeof(float));
  cudaMemset(d_gc33[i],0,nnxz*sizeof(float));
   
  d_Dp1[i]=d_Dpq[i];
  d_Dq1[i]=d_Dpq[i]+nnxz;
  d_Dp0[i]=d_Dpqa[i];
  d_Dq0[i]=d_Dpqa[i]+nnxz;
  
  d_Ddp1[i]=d_Ddpq[i];
  d_Ddq1[i]=d_Ddpq[i]+nnxz;
  d_Ddp0[i]=d_Ddpqa[i];
  d_Ddq0[i]=d_Ddpqa[i]+nnxz;
  
  cudaHostAlloc(&tgc11[i],nnxz*sizeof(float),cudaHostAllocDefault);
  cudaHostAlloc(&tgc13[i],nnxz*sizeof(float),cudaHostAllocDefault);
  cudaHostAlloc(&tgc33[i],nnxz*sizeof(float),cudaHostAllocDefault);
  
  cudaMalloc(&d_tDpq[i],2*nnxz*sizeof(float)); 
  cudaMemset(d_tDpq[i],0,2*nnxz*sizeof(float));
 }

 int npasses=(ns+nGPUs-1)/nGPUs;
 int shotLeft=ns;

// chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 for(int pass=0;pass<npasses;++pass){
  int nGPUsNeed=min(shotLeft,nGPUs);
//  fprintf(stderr,"Pass %d, # GPUs %d\n",pass,nGPUsNeed);
  
  #pragma omp parallel for num_threads(nGPUsNeed)
  for(int i=0;i<nGPUsNeed;++i){
  cudaSetDevice(GPUs[i]);

   int is=pass*nGPUs+i;
   int slocxz=sloc[0+is*4]+sloc[1+is*4]*nnx;

   cudaMalloc(&d_rloc[i],2*sloc[2+is*4]*sizeof(int));
   cudaMemcpy(d_rloc[i],rloc+2*sloc[3+is*4],2*sloc[2+is*4]*sizeof(int),cudaMemcpyHostToDevice);
   
   cudaHostAlloc(&bgdata[i],sloc[2+is*4]*nnt_data*sizeof(float),cudaHostAllocDefault);
   
   cudaMalloc(&d_bgdata[i],sloc[2+is*4]*sizeof(float));
 
   cudaHostAlloc(&borndata[i],sloc[2+is*4]*nnt_data*sizeof(float),cudaHostAllocDefault);
   
   dim3 block(BLOCK_DIM_X,BLOCK_DIM_Y);
   dim3 grid((nnx-2*RADIUS+BLOCK_DIM_X-1)/BLOCK_DIM_X,(nnz-2*RADIUS+BLOCK_DIM_Y-1)/BLOCK_DIM_Y);

   cudaMemset(p0[i],0,nnxz*sizeof(float));
   cudaMemset(q0[i],0,nnxz*sizeof(float));
   cudaMemset(p1[i],0,nnxz*sizeof(float));
   cudaMemset(q1[i],0,nnxz*sizeof(float));
 
   cudaMemset(dp0[i],0,nnxz*sizeof(float));
   cudaMemset(dq0[i],0,nnxz*sizeof(float));
   cudaMemset(dp1[i],0,nnxz*sizeof(float));
   cudaMemset(dq1[i],0,nnxz*sizeof(float));
 
   injectSource<<<1,1,0,compStream[i]>>>(p1[i],q1[i],dt2*wavelet[0],slocxz);
//   injectDipoleSource<<<1,1,0,compStream[i]>>>(p1[i],q1[i],dt2*wavelet[0],slocxz,nnx);
  
   D<<<grid,block,0,compStream[i]>>>(d_Dpq[i],d_Dpq[i]+nnxz,p1[i],q1[i],dx2,dz2,nnx,nnz);
  
   for(int it=2;it<nt;++it){
    float t=it*dt+ot;
	
	forwardC<<<grid,block,0,compStream[i]>>>(p0[i],q0[i],p1[i],q1[i],d_Dpq[i],d_Dpq[i]+nnxz,d_c11[i],d_c13[i],d_c33[i],dt2,nnx,nnz);
  
	forwardC<<<grid,block,0,compStream[i]>>>(dp0[i],dq0[i],dp1[i],dq1[i],d_Ddpq[i],d_Ddpq[i]+nnxz,d_c11[i],d_c13[i],d_c33[i],dt2,nnx,nnz);
  
    injectSource<<<1,1,0,compStream[i]>>>(p0[i],q0[i],dt2*wavelet[it-1],slocxz);
//    injectDipoleSource<<<1,1,0,compStream[i]>>>(p0[i],q0[i],dt2*wavelet[it-1],slocxz,nnx);
    
	scattering<<<grid,block,0,compStream[i]>>>(dp0[i],dq0[i],d_dc11[i],d_dc13[i],d_dc33[i],d_Dpq[i],d_Dpq[i]+nnxz,dt2,nnx,nnz);
    
    abc<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_taper[i],nnx,nnz);
    
    abc<<<grid,block,0,compStream[i]>>>(dp1[i],dq1[i],dp0[i],dq0[i],d_taper[i],nnx,nnz);
    
    D<<<grid,block,0,compStream[i]>>>(d_Dpq[i],d_Dpq[i]+nnxz,p0[i],q0[i],dx2,dz2,nnx,nnz);
  
    D<<<grid,block,0,compStream[i]>>>(d_Ddpq[i],d_Ddpq[i]+nnxz,dp0[i],dq0[i],dx2,dz2,nnx,nnz);
  
    if(t>=0.f && (it-ntNeg)%ratio==0){
	 recordData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(d_bgdata[i],p0[i],q0[i],d_rloc[i],sloc[2+is*4],nnx);
//	 recordDipoleData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(d_bgdata[i],p0[i],q0[i],d_rloc[i],sloc[2+is*4],nnx);
	 
     cudaMemcpyAsync(bgdata[i]+((it-ntNeg)/ratio)*sloc[2+is*4],d_bgdata[i],sloc[2+is*4]*sizeof(float),cudaMemcpyDeviceToHost,compStream[i]);
   	 
     recordData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(d_bgdata[i],dp0[i],dq0[i],d_rloc[i],sloc[2+is*4],nnx);
   	 
     cudaMemcpyAsync(borndata[i]+((it-ntNeg)/ratio)*sloc[2+is*4],d_bgdata[i],sloc[2+is*4]*sizeof(float),cudaMemcpyDeviceToHost,compStream[i]);
    }

	if(it%ratio==0){
   	 cudaStreamSynchronize(transfStream[i]);
 	 if(it!=nt-1){
      cudaMemcpyAsync(d_Dpqa[i],d_Dpq[i],2*nnxz*sizeof(float),cudaMemcpyDeviceToDevice,compStream[i]);
      cudaMemcpyAsync(d_Ddpqa[i],d_Ddpq[i],2*nnxz*sizeof(float),cudaMemcpyDeviceToDevice,compStream[i]);
     }
   	 if(it>ratio){
      memcpy(Dp[i]+(it/ratio-1)*nnxz,Dpqa[i],nnxz*sizeof(float));
      memcpy(Dq[i]+(it/ratio-1)*nnxz,Dpqa[i]+nnxz,nnxz*sizeof(float));
      memcpy(Ddp[i]+(it/ratio-1)*nnxz,Ddpqa[i],nnxz*sizeof(float));
      memcpy(Ddq[i]+(it/ratio-1)*nnxz,Ddpqa[i]+nnxz,nnxz*sizeof(float));
     }
	 cudaStreamSynchronize(compStream[i]);
    }
    
	if(it%ratio==1 && it!=nt-ratio){
     cudaMemcpyAsync(Dpqa[i],d_Dpqa[i],2*nnxz*sizeof(float),cudaMemcpyDeviceToHost,transfStream[i]);
     cudaMemcpyAsync(Ddpqa[i],d_Ddpqa[i],2*nnxz*sizeof(float),cudaMemcpyDeviceToHost,transfStream[i]);
    }
    
	float *pt=p0[i]; 
    p0[i]=p1[i];
    p1[i]=pt;
    pt=q0[i];
    q0[i]=q1[i];
    q1[i]=pt;
    
    pt=dp0[i];
    dp0[i]=dp1[i];
    dp1[i]=pt;
    pt=dq0[i];
    dq0[i]=dq1[i];
    dq1[i]=pt;
   }
   
   cudaStreamSynchronize(transfStream[i]);
   
//   write("bgdata",bgdata[i],sloc[2+is*4]*nnt_data);
//   to_header("bgdata","n1",sloc[2+is*4],"o1",0,"d1",1);
//   to_header("bgdata","n2",nnt_data,"o2",0,"d2",rate);
//   
//   write("borndata",borndata[i],sloc[2+is*4]*nnt_data);
//   to_header("borndata","n1",sloc[2+is*4],"o1",0,"d1",1);
//   to_header("borndata","n2",nnt_data,"o2",0,"d2",rate);
//   
//   write("bgwfld",Dp[i],nnxz*nnt);
//   to_header("bgwfld","n1",nnx,"o1",0,"d1",dx);
//   to_header("bgwfld","n2",nnz,"o2",0,"d2",dz);
//   to_header("bgwfld","n3",nnt,"o3",0,"d3",rate);
//   memset(Dp[i],0,nnxz*nnt*sizeof(float));
//   
//   write("bornwfld",Ddp[i],nnxz*nnt);
//   to_header("bornwfld","n1",nnx,"o1",0,"d1",dx);
//   to_header("bornwfld","n2",nnz,"o2",0,"d2",dz);
//   to_header("bornwfld","n3",nnt,"o3",0,"d3",rate);
//   memset(Ddp[i],0,nnxz*nnt*sizeof(float));
   
   for(int it=0;it<nnt_data;++it){
    for(int ir=0;ir<sloc[2+is*4];++ir){
     bgdata[i][it*sloc[2+is*4]+ir]-=data[it*nr+sloc[3+is*4]+ir];
	}
   }

   cudaMemset(p0[i],0,nnxz*sizeof(float));
   cudaMemset(q0[i],0,nnxz*sizeof(float));
   cudaMemset(p1[i],0,nnxz*sizeof(float));
   cudaMemset(q1[i],0,nnxz*sizeof(float));
  
   cudaMemset(dp0[i],0,nnxz*sizeof(float));
   cudaMemset(dq0[i],0,nnxz*sizeof(float));
   cudaMemset(dp1[i],0,nnxz*sizeof(float));
   cudaMemset(dq1[i],0,nnxz*sizeof(float));
 
   cudaMalloc(&d_bgdata0[i],sloc[2+is*4]*sizeof(float));
   cudaMalloc(&d_bgdata1[i],sloc[2+is*4]*sizeof(float));
   cudaMemset(d_bgdata0[i],0,sloc[2+is*4]*sizeof(float));
   cudaMemset(d_bgdata1[i],0,sloc[2+is*4]*sizeof(float));
  
   cudaMalloc(&d_borndata0[i],sloc[2+is*4]*sizeof(float));
   cudaMalloc(&d_borndata1[i],sloc[2+is*4]*sizeof(float));
   cudaMemset(d_borndata0[i],0,sloc[2+is*4]*sizeof(float));
   cudaMemset(d_borndata1[i],0,sloc[2+is*4]*sizeof(float));
  
   cudaMemcpyAsync(d_bgdata0[i],bgdata[i]+(nnt_data-1)*sloc[2+is*4],sloc[2+is*4]*sizeof(float),cudaMemcpyHostToDevice,compStream[i]);
 
   injectData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(p0[i],q0[i],d_bgdata0[i],d_bgdata1[i],0.f,d_rloc[i],sloc[2+is*4],nnx,dt2);
//   injectDipoleData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(p0[i],q0[i],d_bgdata0[i],d_bgdata1[i],0.f,d_rloc[i],sloc[2+is*4],nnx,dt2);
  
   cudaMemcpyAsync(d_borndata0[i],borndata[i]+(nnt_data-1)*sloc[2+is*4],sloc[2+is*4]*sizeof(float),cudaMemcpyHostToDevice,compStream[i]);
 
   injectData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(dp0[i],dq0[i],d_borndata0[i],d_borndata1[i],0.f,d_rloc[i],sloc[2+is*4],nnx,dt2);
   
   float f=(nt-2.)/ratio;
   int i1=f;
   
   memcpy(Dpqb[i],Dp[i]+(nnt-3)*nnxz,nnxz*sizeof(float));
   memcpy(Dpqb[i]+nnxz,Dq[i]+(nnt-3)*nnxz,nnxz*sizeof(float));
   
   memcpy(Ddpqb[i],Dp[i]+(nnt-3)*nnxz,nnxz*sizeof(float));
   memcpy(Ddpqb[i]+nnxz,Dq[i]+(nnt-3)*nnxz,nnxz*sizeof(float));
   
   f=f-i1;
   gradientCrossCor<<<grid,block,0,compStream[i]>>>(d_gc11[i],d_gc13[i],d_gc33[i],p0[i],q0[i],d_Ddp0[i],d_Ddq0[i],d_Ddp1[i],d_Ddq1[i],f,nnx,nnz);
   gradientCrossCor<<<grid,block,0,compStream[i]>>>(d_gc11[i],d_gc13[i],d_gc33[i],dp0[i],dq0[i],d_Dp0[i],d_Dq0[i],d_Dp1[i],d_Dq1[i],f,nnx,nnz);
  
   for(int it=nt-3;it>=0;--it){
//    D<<<grid,block,0,compStream[i]>>>(d_tDpq[i],d_tDpq[i]+nnxz,p0[i],q0[i],dx2,dz2,nnx,nnz);
//	forwardC<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_tDpq[i],d_tDpq[i]+nnxz,d_c11[i],d_c13[i],d_c33[i],dt2,nnx,nnz);
    
//    forwardCD<<<grid,block,0,compStream[i]>>>(dp1[i],dq1[i],dp0[i],dq0[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);

// this is the correct adjoint
    backwardDC<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
    backwardDC<<<grid,block,0,compStream[i]>>>(dp1[i],dq1[i],dp0[i],dq0[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
    
    if(it%ratio==0){
	 float *pt=Dpqa[i];Dpqa[i]=Dpqb[i];Dpqb[i]=pt;
     pt=Ddpqa[i];Ddpqa[i]=Ddpqb[i];Ddpqb[i]=pt;
     
     cudaMemcpyAsync(d_Dpqb[i],Dpqa[i],2*nnxz*sizeof(float),cudaMemcpyHostToDevice,transfStream[i]);
     cudaMemcpyAsync(d_Ddpqb[i],Ddpqa[i],2*nnxz*sizeof(float),cudaMemcpyHostToDevice,transfStream[i]);
	 
     if(it>=2*ratio){
      memcpy(Dpqb[i],Dp[i]+(it/ratio-2)*nnxz,nnxz*sizeof(float));
      memcpy(Dpqb[i]+nnxz,Dq[i]+(it/ratio-2)*nnxz,nnxz*sizeof(float));
      memcpy(Ddpqb[i],Ddp[i]+(it/ratio-2)*nnxz,nnxz*sizeof(float));
      memcpy(Ddpqb[i]+nnxz,Ddq[i]+(it/ratio-2)*nnxz,nnxz*sizeof(float));
 	 }
	}
    
	if(it>=ntNeg){
	 f=(it-ntNeg+1.)/ratio;
     i1=f;
	 if((it-ntNeg+2)%ratio==0){
      cudaMemcpyAsync(d_bgdata1[i],bgdata[i]+i1*sloc[2+is*4],sloc[2+is*4]*sizeof(float),cudaMemcpyHostToDevice,compStream[i]);
	  cudaStreamSynchronize(compStream[i]);
      float *pt=d_bgdata0[i]; 
      d_bgdata0[i]=d_bgdata1[i];
      d_bgdata1[i]=pt;
      
      cudaMemcpyAsync(d_borndata1[i],borndata[i]+i1*sloc[2+is*4],sloc[2+is*4]*sizeof(float),cudaMemcpyHostToDevice,compStream[i]);
	  cudaStreamSynchronize(compStream[i]);
      pt=d_borndata0[i]; 
      d_borndata0[i]=d_borndata1[i];
      d_borndata1[i]=pt;
     }
	 f=f-i1;
     injectData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(p1[i],q1[i],d_bgdata0[i],d_bgdata1[i],f,d_rloc[i],sloc[2+is*4],nnx,dt2);
     injectData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(dp1[i],dq1[i],d_borndata0[i],d_borndata1[i],f,d_rloc[i],sloc[2+is*4],nnx,dt2);
//     injectDipoleData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(p1[i],q1[i],d_bgdata0[i],d_bgdata1[i],f,d_rloc[i],sloc[2+is*4],nnx,dt2);
	 
//     scattering<<<grid,block,0,compStream[i]>>>(dp1[i],dq1[i],d_dc11[i],d_dc13[i],d_dc33[i],d_tDpq[i],d_tDpq[i]+nnxz,dt2,nnx,nnz);

    // this is for the correct adjoint
     scatteringa<<<grid,block,0,compStream[i]>>>(dp1[i],dq1[i],p0[i],q0[i],d_dc11[i],d_dc13[i],d_dc33[i],dx2,dz2,dt2,nnx,nnz);
	}

    abc<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_taper[i],nnx,nnz);
  
    abc<<<grid,block,0,compStream[i]>>>(dp1[i],dq1[i],dp0[i],dq0[i],d_taper[i],nnx,nnz);
  
    f=(float)it/ratio;
    i1=f;
    
	if((it+1)%ratio==0){
	 cudaStreamSynchronize(transfStream[i]);
     float *pt=d_Dpqb[i];d_Dpqb[i]=d_Dpq[i];d_Dpq[i]=d_Dpqa[i];d_Dpqa[i]=pt;
     d_Dp1[i]=d_Dpq[i];
     d_Dq1[i]=d_Dpq[i]+nnxz;
     d_Dp0[i]=d_Dpqa[i];
     d_Dq0[i]=d_Dpqa[i]+nnxz;
     
     pt=d_Ddpqb[i];d_Ddpqb[i]=d_Ddpq[i];d_Ddpq[i]=d_Ddpqa[i];d_Ddpqa[i]=pt;
     d_Ddp1[i]=d_Ddpq[i];
     d_Ddq1[i]=d_Ddpq[i]+nnxz;
     d_Ddp0[i]=d_Ddpqa[i];
     d_Ddq0[i]=d_Ddpqa[i]+nnxz;
	}
    
	f=f-i1;
    gradientCrossCor<<<grid,block,0,compStream[i]>>>(d_gc11[i],d_gc13[i],d_gc33[i],p1[i],q1[i],d_Ddp0[i],d_Ddq0[i],d_Ddp1[i],d_Ddq1[i],f,nnx,nnz);
    gradientCrossCor<<<grid,block,0,compStream[i]>>>(d_gc11[i],d_gc13[i],d_gc33[i],dp1[i],dq1[i],d_Dp0[i],d_Dq0[i],d_Dp1[i],d_Dq1[i],f,nnx,nnz);
 
 //from here
//	if(it%ratio==0){
//   	 cudaStreamSynchronize(transfStream[i]);
// 	 if(it!=nt-1){
//      cudaMemcpyAsync(d_Dpqa[i],p1[i],nnxz*sizeof(float),cudaMemcpyDeviceToDevice,compStream[i]);
//      cudaMemcpyAsync(d_Ddpqa[i],dp1[i],nnxz*sizeof(float),cudaMemcpyDeviceToDevice,compStream[i]);
//     }
//   	 if(it>ratio){
//      memcpy(Dp[i]+(it/ratio-1)*nnxz,Dpqa[i],nnxz*sizeof(float));
//      memcpy(Ddp[i]+(it/ratio-1)*nnxz,Ddpqa[i],nnxz*sizeof(float));
//     }
//	 cudaStreamSynchronize(compStream[i]);
//    }
//    
//	if(it%ratio==1 && it!=nt-ratio){
//     cudaMemcpyAsync(Dpqa[i],d_Dpqa[i],nnxz*sizeof(float),cudaMemcpyDeviceToHost,transfStream[i]);
//     cudaMemcpyAsync(Ddpqa[i],d_Ddpqa[i],nnxz*sizeof(float),cudaMemcpyDeviceToHost,transfStream[i]);
//    }
 //to here just to save the adjoint wavefields, for debugging purposes. this is not necessary

	float *pt=p0[i]; 
    p0[i]=p1[i];
    p1[i]=pt;
    pt=q0[i];
    q0[i]=q1[i];
    q1[i]=pt;
	
    pt=dp0[i]; 
    dp0[i]=dp1[i];
    dp1[i]=pt;
    pt=dq0[i];
    dq0[i]=dq1[i];
    dq1[i]=pt;
   }
   
   cudaFreeHost(bgdata[i]);
   cudaFreeHost(borndata[i]);
   cudaFree(d_bgdata[i]);cudaFree(d_rloc[i]);
   cudaFree(d_bgdata0[i]);cudaFree(d_bgdata1[i]);
   cudaFree(d_borndata0[i]);cudaFree(d_borndata1[i]);
   
 //from here
//   write("awfld",Dp[i],nnxz*nnt);
//   to_header("awfld","n1",nnx,"o1",0,"d1",dx);
//   to_header("awfld","n2",nnz,"o2",0,"d2",dz);
//   to_header("awfld","n3",nnt,"o3",0,"d3",rate);
//   
//   write("scatawfld",Ddp[i],nnxz*nnt);
//   to_header("scatawfld","n1",nnx,"o1",0,"d1",dx);
//   to_header("scatawfld","n2",nnz,"o2",0,"d2",dz);
//   to_header("scatawfld","n3",nnt,"o3",0,"d3",rate);
 //to here just to save the adjoint wavefields, for debugging purposes. this is not necessary
  }
  
  shotLeft-=nGPUsNeed;
 }

 #pragma omp parallel for num_threads(nGPUs)
 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(GPUs[i]);
  cudaStreamSynchronize(compStream[i]);
  cudaStreamSynchronize(transfStream[i]);
  cudaMemcpyAsync(tgc11[i],d_gc11[i],nnxz*sizeof(float),cudaMemcpyDeviceToHost,compStream[i]);
  cudaMemcpyAsync(tgc13[i],d_gc13[i],nnxz*sizeof(float),cudaMemcpyDeviceToHost,compStream[i]);
  cudaMemcpyAsync(tgc33[i],d_gc33[i],nnxz*sizeof(float),cudaMemcpyDeviceToHost,transfStream[i]);
  cudaDeviceSynchronize();
 }

 for(int i=0;i<nGPUs;++i){
  #pragma omp parallel for num_threads(16) shared(i)
  for(int ixz=0;ixz<nnxz;++ixz){
   gc11[ixz]+=tgc11[i][ixz];
   gc13[ixz]+=tgc13[i][ixz];
   gc33[ixz]+=tgc33[i][ixz];
  }
 }

// chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
// chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
// cout<<"done "<<time.count()<<" seconds"<<endl;
 
 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(GPUs[i]);
  cudaFree(d_c11[i]);cudaFree(d_c13[i]);cudaFree(d_c33[i]); 
  cudaFree(d_dc11[i]);cudaFree(d_dc13[i]);cudaFree(d_dc33[i]); 
  cudaFree(d_taper[i]);
  cudaFree(p0[i]);cudaFree(p1[i]);cudaFree(q0[i]);cudaFree(q1[i]);
  cudaFree(dp0[i]);cudaFree(dp1[i]);cudaFree(dq0[i]);cudaFree(dq1[i]);
  cudaFree(d_Dpq[i]);cudaFree(d_Dpqa[i]);cudaFree(d_Dpqb[i]);
  cudaFree(d_Ddpq[i]);cudaFree(d_Ddpqa[i]);cudaFree(d_Ddpqb[i]);
  delete []Dp[i];delete []Dq[i]; 
  delete []Ddp[i];delete []Ddq[i]; 
  cudaFree(d_gc11[i]);cudaFree(d_gc13[i]);cudaFree(d_gc33[i]);
  cudaFreeHost(tgc11[i]);cudaFreeHost(tgc13[i]);cudaFreeHost(tgc33[i]);
  cudaFree(d_tDpq[i]);
  cudaStreamDestroy(compStream[i]);
  cudaStreamDestroy(transfStream[i]);
  cudaFreeHost(Dpqa[i]);cudaFreeHost(Dpqb[i]);
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"gpu %d error %s\n",GPUs[i],cudaGetErrorString(e));
 }
 
 delete []bgdata;
 delete []borndata;
 delete []d_rloc;
 delete []d_c11;delete []d_c13;delete []d_c33;
 delete []d_dc11;delete []d_dc13;delete []d_dc33;
 delete []d_taper;
 delete []d_bgdata;
 delete []p0;delete []p1;delete []q0;delete []q1;
 delete []dp0;delete []dp1;delete []dq0;delete []dq1;
 delete []d_Dpq;delete []d_Dpqa;delete []d_Dpqb;
 delete []d_Ddpq;delete []d_Ddpqa;delete []d_Ddpqb;
 delete []Dp;delete []Dq;
 delete []Ddp;delete []Ddq;
 delete []d_gc11;delete []d_gc13;delete []d_gc33;
 delete []d_bgdata0;delete []d_bgdata1;
 delete []d_borndata0;delete []d_borndata1;
 delete []d_Dp0;delete []d_Dp1;delete []d_Dq0;delete []d_Dq1;
 delete []d_Ddp0;delete []d_Ddp1;delete []d_Ddq0;delete []d_Ddq1;
 delete []tgc11;delete []tgc13;delete []tgc33;
 delete []Dpqa;
 delete []Dpqb;
 delete []Ddpqa;
 delete []Ddpqb;
 delete []d_tDpq;

// cudaProfilerStop();

 return;
}

void GNhessian_f(float *gc11,float *gc13,float *gc33,const float *data,const float *c11,const float *c13,const float *c33,const float *dc11,const float *dc13,const float *dc33,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot){
// fprintf(stderr,"Calculate obj func and grad\n");
 
 int ratio=rate/dt+0.5f;
 int ntNeg=std::round(abs(ot)/dt);
 int nnt=(nt-1)/ratio+1;
 int nnt_data=(nt-ntNeg-1)/ratio+1;
 int nnx=nx+2*npad,nnz=nz+2*npad;
 int nnxz=nnx*nnz;
 float dx2=dx*dx,dz2=dz*dz,dt2=dt*dt;
 
 memset(gc11,0,nnxz*sizeof(float));
 memset(gc13,0,nnxz*sizeof(float));
 memset(gc33,0,nnxz*sizeof(float));
 
 std::vector<int> GPUs;
 get_array("gpu",GPUs);
 int nGPUs=GPUs.size();
// fprintf(stderr,"Total # GPUs = %d\n",nGPUs);
// fprintf(stderr,"GPUs used are:\n");
// for(int i=0;i<nGPUs;i++) fprintf(stderr,"%d",GPUs[i]);
// fprintf(stderr,"\n");

 float **borndata=new float*[nGPUs]();
 int **d_rloc=new int*[nGPUs]();
 float **d_c11=new float*[nGPUs]();
 float **d_c13=new float*[nGPUs]();
 float **d_c33=new float*[nGPUs]();
 float **d_dc11=new float*[nGPUs]();
 float **d_dc13=new float*[nGPUs]();
 float **d_dc33=new float*[nGPUs]();
 float **d_taper=new float*[nGPUs]();
 float **d_bgdata=new float*[nGPUs]();
 float **p0=new float*[nGPUs]();
 float **q0=new float*[nGPUs]();
 float **p1=new float*[nGPUs]();
 float **q1=new float*[nGPUs]();
 float **dp0=new float*[nGPUs]();
 float **dq0=new float*[nGPUs]();
 float **dp1=new float*[nGPUs]();
 float **dq1=new float*[nGPUs]();
 float **d_Dpq=new float*[nGPUs]();
 float **d_Ddpq=new float*[nGPUs]();
 float **d_Dpqa=new float*[nGPUs]();
 float **d_Dpqb=new float*[nGPUs]();
 float **Dp=new float*[nGPUs]();
 float **Dq=new float*[nGPUs]();
 float **Dpqa=new float*[nGPUs]();
 float **Dpqb=new float*[nGPUs]();
 float **d_gc11=new float*[nGPUs]();
 float **d_gc13=new float*[nGPUs]();
 float **d_gc33=new float*[nGPUs]();
 float **d_borndata0=new float*[nGPUs]();
 float **d_borndata1=new float*[nGPUs]();
 float **d_Dp0=new float*[nGPUs]();
 float **d_Dp1=new float*[nGPUs]();
 float **d_Dq0=new float*[nGPUs]();
 float **d_Dq1=new float*[nGPUs]();
 float **tgc11=new float*[nGPUs]();
 float **tgc13=new float*[nGPUs]();
 float **tgc33=new float*[nGPUs]();

 cudaStream_t *compStream=new cudaStream_t[nGPUs];
 cudaStream_t *transfStream=new cudaStream_t[nGPUs];

 #pragma omp parallel for num_threads(nGPUs)
 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(GPUs[i]);
  
  cudaMalloc(&d_c11[i],nnxz*sizeof(float));
  cudaMalloc(&d_c13[i],nnxz*sizeof(float));
  cudaMalloc(&d_c33[i],nnxz*sizeof(float));
  cudaMemcpy(d_c11[i],c11,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_c13[i],c13,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_c33[i],c33,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  
  cudaMalloc(&d_dc11[i],nnxz*sizeof(float));
  cudaMalloc(&d_dc13[i],nnxz*sizeof(float));
  cudaMalloc(&d_dc33[i],nnxz*sizeof(float));
  cudaMemcpy(d_dc11[i],dc11,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_dc13[i],dc13,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_dc33[i],dc33,nnxz*sizeof(float),cudaMemcpyHostToDevice);
  
  cudaMalloc(&d_taper[i],nnxz*sizeof(float));
  cudaMemcpy(d_taper[i],taper,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 
  cudaMalloc(&p0[i],nnxz*sizeof(float)); 
  cudaMalloc(&p1[i],nnxz*sizeof(float)); 
  cudaMalloc(&q0[i],nnxz*sizeof(float)); 
  cudaMalloc(&q1[i],nnxz*sizeof(float)); 
  
  cudaMalloc(&dp0[i],nnxz*sizeof(float)); 
  cudaMalloc(&dp1[i],nnxz*sizeof(float)); 
  cudaMalloc(&dq0[i],nnxz*sizeof(float)); 
  cudaMalloc(&dq1[i],nnxz*sizeof(float)); 
  
  cudaMalloc(&d_Dpq[i],2*nnxz*sizeof(float)); 
  cudaMemset(d_Dpq[i],0,2*nnxz*sizeof(float));
  
  cudaMalloc(&d_Ddpq[i],2*nnxz*sizeof(float)); 
  cudaMemset(d_Ddpq[i],0,2*nnxz*sizeof(float));
  
  cudaMalloc(&d_Dpqa[i],2*nnxz*sizeof(float)); 
  cudaMalloc(&d_Dpqb[i],2*nnxz*sizeof(float)); 
  cudaMemset(d_Dpqa[i],0,2*nnxz*sizeof(float)); 
  cudaMemset(d_Dpqb[i],0,2*nnxz*sizeof(float)); 
  
  Dp[i]=new float[nnxz*nnt]();
  Dq[i]=new float[nnxz*nnt]();
  
  cudaHostAlloc(&Dpqa[i],2*nnxz*sizeof(float),cudaHostAllocDefault);
  cudaHostAlloc(&Dpqb[i],2*nnxz*sizeof(float),cudaHostAllocDefault);
  cudaMemset(Dpqa[i],0,2*nnxz*sizeof(float));
  cudaMemset(Dpqb[i],0,2*nnxz*sizeof(float));
   
  cudaStreamCreate(&compStream[i]); 
  cudaStreamCreate(&transfStream[i]); 
  
  cudaMalloc(&d_gc11[i],nnxz*sizeof(float));
  cudaMalloc(&d_gc13[i],nnxz*sizeof(float));
  cudaMalloc(&d_gc33[i],nnxz*sizeof(float));
  
  cudaMemset(d_gc11[i],0,nnxz*sizeof(float));
  cudaMemset(d_gc13[i],0,nnxz*sizeof(float));
  cudaMemset(d_gc33[i],0,nnxz*sizeof(float));
   
  d_Dp1[i]=d_Dpq[i];
  d_Dq1[i]=d_Dpq[i]+nnxz;
  d_Dp0[i]=d_Dpqa[i];
  d_Dq0[i]=d_Dpqa[i]+nnxz;
  
  cudaHostAlloc(&tgc11[i],nnxz*sizeof(float),cudaHostAllocDefault);
  cudaHostAlloc(&tgc13[i],nnxz*sizeof(float),cudaHostAllocDefault);
  cudaHostAlloc(&tgc33[i],nnxz*sizeof(float),cudaHostAllocDefault);
 }

 int npasses=(ns+nGPUs-1)/nGPUs;
 int shotLeft=ns;

// chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 for(int pass=0;pass<npasses;++pass){
  int nGPUsNeed=min(shotLeft,nGPUs);
//  fprintf(stderr,"Pass %d, # GPUs %d\n",pass,nGPUsNeed);
  
  #pragma omp parallel for num_threads(nGPUsNeed)
  for(int i=0;i<nGPUsNeed;++i){
  cudaSetDevice(GPUs[i]);

   int is=pass*nGPUs+i;
   int slocxz=sloc[0+is*4]+sloc[1+is*4]*nnx;

   cudaMalloc(&d_rloc[i],2*sloc[2+is*4]*sizeof(int));
   cudaMemcpy(d_rloc[i],rloc+2*sloc[3+is*4],2*sloc[2+is*4]*sizeof(int),cudaMemcpyHostToDevice);
   
   cudaMalloc(&d_bgdata[i],sloc[2+is*4]*sizeof(float));
 
   cudaHostAlloc(&borndata[i],sloc[2+is*4]*nnt_data*sizeof(float),cudaHostAllocDefault);
   
   dim3 block(BLOCK_DIM_X,BLOCK_DIM_Y);
   dim3 grid((nnx-2*RADIUS+BLOCK_DIM_X-1)/BLOCK_DIM_X,(nnz-2*RADIUS+BLOCK_DIM_Y-1)/BLOCK_DIM_Y);

   cudaMemset(p0[i],0,nnxz*sizeof(float));
   cudaMemset(q0[i],0,nnxz*sizeof(float));
   cudaMemset(p1[i],0,nnxz*sizeof(float));
   cudaMemset(q1[i],0,nnxz*sizeof(float));
 
   cudaMemset(dp0[i],0,nnxz*sizeof(float));
   cudaMemset(dq0[i],0,nnxz*sizeof(float));
   cudaMemset(dp1[i],0,nnxz*sizeof(float));
   cudaMemset(dq1[i],0,nnxz*sizeof(float));
 
   injectSource<<<1,1,0,compStream[i]>>>(p1[i],q1[i],dt2*wavelet[0],slocxz);
//   injectDipoleSource<<<1,1,0,compStream[i]>>>(p1[i],q1[i],dt2*wavelet[0],slocxz,nnx);
  
   D<<<grid,block,0,compStream[i]>>>(d_Dpq[i],d_Dpq[i]+nnxz,p1[i],q1[i],dx2,dz2,nnx,nnz);
  
   for(int it=2;it<nt;++it){
    float t=it*dt+ot;
	
	forwardC<<<grid,block,0,compStream[i]>>>(p0[i],q0[i],p1[i],q1[i],d_Dpq[i],d_Dpq[i]+nnxz,d_c11[i],d_c13[i],d_c33[i],dt2,nnx,nnz);
  
	forwardC<<<grid,block,0,compStream[i]>>>(dp0[i],dq0[i],dp1[i],dq1[i],d_Ddpq[i],d_Ddpq[i]+nnxz,d_c11[i],d_c13[i],d_c33[i],dt2,nnx,nnz);
  
    injectSource<<<1,1,0,compStream[i]>>>(p0[i],q0[i],dt2*wavelet[it-1],slocxz);
//    injectDipoleSource<<<1,1,0,compStream[i]>>>(p0[i],q0[i],dt2*wavelet[it-1],slocxz,nnx);
    
	scattering<<<grid,block,0,compStream[i]>>>(dp0[i],dq0[i],d_dc11[i],d_dc13[i],d_dc33[i],d_Dpq[i],d_Dpq[i]+nnxz,dt2,nnx,nnz);
    
    abc<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_taper[i],nnx,nnz);
    
    abc<<<grid,block,0,compStream[i]>>>(dp1[i],dq1[i],dp0[i],dq0[i],d_taper[i],nnx,nnz);
    
    D<<<grid,block,0,compStream[i]>>>(d_Dpq[i],d_Dpq[i]+nnxz,p0[i],q0[i],dx2,dz2,nnx,nnz);
  
    D<<<grid,block,0,compStream[i]>>>(d_Ddpq[i],d_Ddpq[i]+nnxz,dp0[i],dq0[i],dx2,dz2,nnx,nnz);
  
    if(t>=0.f && (it-ntNeg)%ratio==0){
     recordData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(d_bgdata[i],dp0[i],dq0[i],d_rloc[i],sloc[2+is*4],nnx);
     cudaMemcpyAsync(borndata[i]+((it-ntNeg)/ratio)*sloc[2+is*4],d_bgdata[i],sloc[2+is*4]*sizeof(float),cudaMemcpyDeviceToHost,compStream[i]);
    }

	if(it%ratio==0){
   	 cudaStreamSynchronize(transfStream[i]);
 	 if(it!=nt-1){
      cudaMemcpyAsync(d_Dpqa[i],d_Dpq[i],2*nnxz*sizeof(float),cudaMemcpyDeviceToDevice,compStream[i]);
     }
   	 if(it>ratio){
      memcpy(Dp[i]+(it/ratio-1)*nnxz,Dpqa[i],nnxz*sizeof(float));
      memcpy(Dq[i]+(it/ratio-1)*nnxz,Dpqa[i]+nnxz,nnxz*sizeof(float));
     }
	 cudaStreamSynchronize(compStream[i]);
    }
    
	if(it%ratio==1 && it!=nt-ratio){
     cudaMemcpyAsync(Dpqa[i],d_Dpqa[i],2*nnxz*sizeof(float),cudaMemcpyDeviceToHost,transfStream[i]);
    }
    
	float *pt=p0[i]; 
    p0[i]=p1[i];
    p1[i]=pt;
    pt=q0[i];
    q0[i]=q1[i];
    q1[i]=pt;
    
    pt=dp0[i];
    dp0[i]=dp1[i];
    dp1[i]=pt;
    pt=dq0[i];
    dq0[i]=dq1[i];
    dq1[i]=pt;
   }
   
   cudaStreamSynchronize(transfStream[i]);
   
//   write("bgdata",bgdata[i],sloc[2+is*4]*nnt_data);
//   to_header("bgdata","n1",sloc[2+is*4],"o1",0,"d1",1);
//   to_header("bgdata","n2",nnt_data,"o2",0,"d2",rate);
//   
//   write("borndata",borndata[i],sloc[2+is*4]*nnt_data);
//   to_header("borndata","n1",sloc[2+is*4],"o1",0,"d1",1);
//   to_header("borndata","n2",nnt_data,"o2",0,"d2",rate);
//   
//   write("bgwfld",Dp[i],nnxz*nnt);
//   to_header("bgwfld","n1",nnx,"o1",0,"d1",dx);
//   to_header("bgwfld","n2",nnz,"o2",0,"d2",dz);
//   to_header("bgwfld","n3",nnt,"o3",0,"d3",rate);
//   memset(Dp[i],0,nnxz*nnt*sizeof(float));
//   
//   write("bornwfld",Ddp[i],nnxz*nnt);
//   to_header("bornwfld","n1",nnx,"o1",0,"d1",dx);
//   to_header("bornwfld","n2",nnz,"o2",0,"d2",dz);
//   to_header("bornwfld","n3",nnt,"o3",0,"d3",rate);
//   memset(Ddp[i],0,nnxz*nnt*sizeof(float));
   
   cudaMemset(p0[i],0,nnxz*sizeof(float));
   cudaMemset(q0[i],0,nnxz*sizeof(float));
   cudaMemset(p1[i],0,nnxz*sizeof(float));
   cudaMemset(q1[i],0,nnxz*sizeof(float));
  
   cudaMalloc(&d_borndata0[i],sloc[2+is*4]*sizeof(float));
   cudaMalloc(&d_borndata1[i],sloc[2+is*4]*sizeof(float));
   cudaMemset(d_borndata0[i],0,sloc[2+is*4]*sizeof(float));
   cudaMemset(d_borndata1[i],0,sloc[2+is*4]*sizeof(float));
  
   cudaMemcpyAsync(d_borndata0[i],borndata[i]+(nnt_data-1)*sloc[2+is*4],sloc[2+is*4]*sizeof(float),cudaMemcpyHostToDevice,compStream[i]);
 
   injectData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(p0[i],q0[i],d_borndata0[i],d_borndata1[i],0.f,d_rloc[i],sloc[2+is*4],nnx,dt2);
   
   float f=(nt-2.)/ratio;
   int i1=f;
   
   memcpy(Dpqb[i],Dp[i]+(nnt-3)*nnxz,nnxz*sizeof(float));
   memcpy(Dpqb[i]+nnxz,Dq[i]+(nnt-3)*nnxz,nnxz*sizeof(float));
   
   f=f-i1;
   gradientCrossCor<<<grid,block,0,compStream[i]>>>(d_gc11[i],d_gc13[i],d_gc33[i],p0[i],q0[i],d_Dp0[i],d_Dq0[i],d_Dp1[i],d_Dq1[i],f,nnx,nnz);
  
   for(int it=nt-3;it>=0;--it){
    forwardCD<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
//    backwardDC<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
    
    if(it%ratio==0){
	 float *pt=Dpqa[i];Dpqa[i]=Dpqb[i];Dpqb[i]=pt;
     
     cudaMemcpyAsync(d_Dpqb[i],Dpqa[i],2*nnxz*sizeof(float),cudaMemcpyHostToDevice,transfStream[i]);
	 
     if(it>=2*ratio){
      memcpy(Dpqb[i],Dp[i]+(it/ratio-2)*nnxz,nnxz*sizeof(float));
      memcpy(Dpqb[i]+nnxz,Dq[i]+(it/ratio-2)*nnxz,nnxz*sizeof(float));
 	 }
	}
    
	if(it>=ntNeg){
	 f=(it-ntNeg+1.)/ratio;
     i1=f;
	 if((it-ntNeg+2)%ratio==0){
      cudaMemcpyAsync(d_borndata1[i],borndata[i]+i1*sloc[2+is*4],sloc[2+is*4]*sizeof(float),cudaMemcpyHostToDevice,compStream[i]);
	  cudaStreamSynchronize(compStream[i]);
      float *pt=d_borndata0[i]; 
      d_borndata0[i]=d_borndata1[i];
      d_borndata1[i]=pt;
     }
	 f=f-i1;
     injectData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(p1[i],q1[i],d_borndata0[i],d_borndata1[i],f,d_rloc[i],sloc[2+is*4],nnx,dt2);
//     injectDipoleData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(p1[i],q1[i],d_bgdata0[i],d_bgdata1[i],f,d_rloc[i],sloc[2+is*4],nnx,dt2);
	}

    abc<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_taper[i],nnx,nnz);
  
    f=(float)it/ratio;
    i1=f;
    
	if((it+1)%ratio==0){
	 cudaStreamSynchronize(transfStream[i]);
     float *pt=d_Dpqb[i];d_Dpqb[i]=d_Dpq[i];d_Dpq[i]=d_Dpqa[i];d_Dpqa[i]=pt;
     d_Dp1[i]=d_Dpq[i];
     d_Dq1[i]=d_Dpq[i]+nnxz;
     d_Dp0[i]=d_Dpqa[i];
     d_Dq0[i]=d_Dpqa[i]+nnxz;
	}
    
	f=f-i1;
    gradientCrossCor<<<grid,block,0,compStream[i]>>>(d_gc11[i],d_gc13[i],d_gc33[i],p1[i],q1[i],d_Dp0[i],d_Dq0[i],d_Dp1[i],d_Dq1[i],f,nnx,nnz);
 
 //from here
//	if(it%ratio==0){
//   	 cudaStreamSynchronize(transfStream[i]);
// 	 if(it!=nt-1){
//      cudaMemcpyAsync(d_Dpqa[i],p1[i],nnxz*sizeof(float),cudaMemcpyDeviceToDevice,compStream[i]);
//      cudaMemcpyAsync(d_Ddpqa[i],dp1[i],nnxz*sizeof(float),cudaMemcpyDeviceToDevice,compStream[i]);
//     }
//   	 if(it>ratio){
//      memcpy(Dp[i]+(it/ratio-1)*nnxz,Dpqa[i],nnxz*sizeof(float));
//      memcpy(Ddp[i]+(it/ratio-1)*nnxz,Ddpqa[i],nnxz*sizeof(float));
//     }
//	 cudaStreamSynchronize(compStream[i]);
//    }
//    
//	if(it%ratio==1 && it!=nt-ratio){
//     cudaMemcpyAsync(Dpqa[i],d_Dpqa[i],nnxz*sizeof(float),cudaMemcpyDeviceToHost,transfStream[i]);
//     cudaMemcpyAsync(Ddpqa[i],d_Ddpqa[i],nnxz*sizeof(float),cudaMemcpyDeviceToHost,transfStream[i]);
//    }
 //to here just to save the adjoint wavefields, for debugging purposes. this is not necessary

	float *pt=p0[i]; 
    p0[i]=p1[i];
    p1[i]=pt;
    pt=q0[i];
    q0[i]=q1[i];
    q1[i]=pt;
   }
   
   cudaFreeHost(borndata[i]);
   cudaFree(d_bgdata[i]);cudaFree(d_rloc[i]);
   cudaFree(d_borndata0[i]);cudaFree(d_borndata1[i]);
   
 //from here
//   write("awfld",Dp[i],nnxz*nnt);
//   to_header("awfld","n1",nnx,"o1",0,"d1",dx);
//   to_header("awfld","n2",nnz,"o2",0,"d2",dz);
//   to_header("awfld","n3",nnt,"o3",0,"d3",rate);
//   
//   write("scatawfld",Ddp[i],nnxz*nnt);
//   to_header("scatawfld","n1",nnx,"o1",0,"d1",dx);
//   to_header("scatawfld","n2",nnz,"o2",0,"d2",dz);
//   to_header("scatawfld","n3",nnt,"o3",0,"d3",rate);
 //to here just to save the adjoint wavefields, for debugging purposes. this is not necessary
  }
  
  shotLeft-=nGPUsNeed;
 }

 #pragma omp parallel for num_threads(nGPUs)
 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(GPUs[i]);
  cudaStreamSynchronize(compStream[i]);
  cudaStreamSynchronize(transfStream[i]);
  cudaMemcpyAsync(tgc11[i],d_gc11[i],nnxz*sizeof(float),cudaMemcpyDeviceToHost,compStream[i]);
  cudaMemcpyAsync(tgc13[i],d_gc13[i],nnxz*sizeof(float),cudaMemcpyDeviceToHost,compStream[i]);
  cudaMemcpyAsync(tgc33[i],d_gc33[i],nnxz*sizeof(float),cudaMemcpyDeviceToHost,transfStream[i]);
  cudaDeviceSynchronize();
 }

 for(int i=0;i<nGPUs;++i){
  #pragma omp parallel for num_threads(16) shared(i)
  for(int ixz=0;ixz<nnxz;++ixz){
   gc11[ixz]+=tgc11[i][ixz];
   gc13[ixz]+=tgc13[i][ixz];
   gc33[ixz]+=tgc33[i][ixz];
  }
 }

// chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
// chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
// cout<<"done "<<time.count()<<" seconds"<<endl;
 
 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(GPUs[i]);
  cudaFree(d_c11[i]);cudaFree(d_c13[i]);cudaFree(d_c33[i]); 
  cudaFree(d_dc11[i]);cudaFree(d_dc13[i]);cudaFree(d_dc33[i]); 
  cudaFree(d_taper[i]);
  cudaFree(p0[i]);cudaFree(p1[i]);cudaFree(q0[i]);cudaFree(q1[i]);
  cudaFree(dp0[i]);cudaFree(dp1[i]);cudaFree(dq0[i]);cudaFree(dq1[i]);
  cudaFree(d_Dpq[i]);cudaFree(d_Dpqa[i]);cudaFree(d_Dpqb[i]);cudaFree(d_Ddpq[i]);
  delete []Dp[i];delete []Dq[i]; 
  cudaFree(d_gc11[i]);cudaFree(d_gc13[i]);cudaFree(d_gc33[i]);
  cudaFreeHost(tgc11[i]);cudaFreeHost(tgc13[i]);cudaFreeHost(tgc33[i]);
  cudaStreamDestroy(compStream[i]);
  cudaStreamDestroy(transfStream[i]);
  cudaFreeHost(Dpqa[i]);cudaFreeHost(Dpqb[i]);
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"gpu %d error %s\n",GPUs[i],cudaGetErrorString(e));
 }
 
 delete []borndata;
 delete []d_rloc;
 delete []d_c11;delete []d_c13;delete []d_c33;
 delete []d_dc11;delete []d_dc13;delete []d_dc33;
 delete []d_taper;
 delete []d_bgdata;
 delete []p0;delete []p1;delete []q0;delete []q1;
 delete []dp0;delete []dp1;delete []dq0;delete []dq1;
 delete []d_Dpq;delete []d_Dpqa;delete []d_Dpqb;delete []d_Ddpq;
 delete []Dp;delete []Dq;
 delete []d_gc11;delete []d_gc13;delete []d_gc33;
 delete []d_borndata0;delete []d_borndata1;
 delete []d_Dp0;delete []d_Dp1;delete []d_Dq0;delete []d_Dq1;
 delete []tgc11;delete []tgc13;delete []tgc33;
 delete []Dpqa;
 delete []Dpqb;

// cudaProfilerStop();

 return;
}
