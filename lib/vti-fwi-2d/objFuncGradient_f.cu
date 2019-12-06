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

float objFuncGradient_f(float *gc11,float *gc13,float *gc33,const float *data,const float *c11,const float *c13,const float *c33,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot){
 
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

 vector<double> val(nGPUs,0.);
 
 float **tbgdata=new float*[nGPUs]();
 float **tdata=new float*[nGPUs]();
 float **tres=new float*[nGPUs]();
 float **tadjsou=new float*[nGPUs]();
 float **bgdata=new float*[nGPUs]();
 int **d_rloc=new int*[nGPUs]();
 float **d_c11=new float*[nGPUs]();
 float **d_c13=new float*[nGPUs]();
 float **d_c33=new float*[nGPUs]();
 float **d_taper=new float*[nGPUs]();
 float **d_bgdata=new float*[nGPUs]();
 float **p0=new float*[nGPUs]();
 float **q0=new float*[nGPUs]();
 float **p1=new float*[nGPUs]();
 float **q1=new float*[nGPUs]();
 float **d_Dpq=new float*[nGPUs]();
 float **d_Dpqa=new float*[nGPUs]();
 float **d_Dpqb=new float*[nGPUs]();
 float **Dp=new float*[nGPUs]();
 float **Dq=new float*[nGPUs]();
 float **Dpqa=new float*[nGPUs]();
 float **Dpqb=new float*[nGPUs]();
 float **d_gc11=new float*[nGPUs]();
 float **d_gc13=new float*[nGPUs]();
 float **d_gc33=new float*[nGPUs]();
 float **d_bgdata0=new float*[nGPUs]();
 float **d_bgdata1=new float*[nGPUs]();
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
  
  cudaMalloc(&d_taper[i],nnxz*sizeof(float));
  cudaMemcpy(d_taper[i],taper,nnxz*sizeof(float),cudaMemcpyHostToDevice);
 
  cudaMalloc(&p0[i],nnxz*sizeof(float)); 
  cudaMalloc(&p1[i],nnxz*sizeof(float)); 
  cudaMalloc(&q0[i],nnxz*sizeof(float)); 
  cudaMalloc(&q1[i],nnxz*sizeof(float)); 
  
  cudaMalloc(&d_Dpq[i],2*nnxz*sizeof(float)); 
  cudaMemset(d_Dpq[i],0,2*nnxz*sizeof(float));
  
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
   int nri=sloc[2+is*4];

   cudaMalloc(&d_rloc[i],2*nri*sizeof(int));
   cudaMemcpy(d_rloc[i],rloc+2*sloc[3+is*4],2*nri*sizeof(int),cudaMemcpyHostToDevice);
   
   cudaHostAlloc(&bgdata[i],nri*nnt_data*sizeof(float),cudaHostAllocDefault);
   
   cudaMalloc(&d_bgdata[i],nri*sizeof(float));
 
   dim3 block(BLOCK_DIM_X,BLOCK_DIM_Y);
   dim3 grid((nnx-2*RADIUS+BLOCK_DIM_X-1)/BLOCK_DIM_X,(nnz-2*RADIUS+BLOCK_DIM_Y-1)/BLOCK_DIM_Y);

   cudaMemset(p0[i],0,nnxz*sizeof(float));
   cudaMemset(q0[i],0,nnxz*sizeof(float));
   cudaMemset(p1[i],0,nnxz*sizeof(float));
   cudaMemset(q1[i],0,nnxz*sizeof(float));
 
   injectSource<<<1,1,0,compStream[i]>>>(p1[i],q1[i],dt2*wavelet[0],slocxz);
//   injectDipoleSource<<<1,1,0,compStream[i]>>>(p1[i],q1[i],dt2*wavelet[0],slocxz,nnx);
  
   D<<<grid,block,0,compStream[i]>>>(d_Dpq[i],d_Dpq[i]+nnxz,p1[i],q1[i],dx2,dz2,nnx,nnz);
  
   for(int it=2;it<nt;++it){
    float t=it*dt+ot;
	
	forwardC<<<grid,block,0,compStream[i]>>>(p0[i],q0[i],p1[i],q1[i],d_Dpq[i],d_Dpq[i]+nnxz,d_c11[i],d_c13[i],d_c33[i],dt2,nnx,nnz);
  
    injectSource<<<1,1,0,compStream[i]>>>(p0[i],q0[i],dt2*wavelet[it-1],slocxz);
//    injectDipoleSource<<<1,1,0,compStream[i]>>>(p0[i],q0[i],dt2*wavelet[it-1],slocxz,nnx);
  
    abc<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_taper[i],nnx,nnz);
    
    D<<<grid,block,0,compStream[i]>>>(d_Dpq[i],d_Dpq[i]+nnxz,p0[i],q0[i],dx2,dz2,nnx,nnz);
  
    if(t>=0.f && (it-ntNeg)%ratio==0){
	 recordData<<<(nri+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(d_bgdata[i],p0[i],q0[i],d_rloc[i],nri,nnx);
//	 recordDipoleData<<<(nri+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(d_bgdata[i],p0[i],q0[i],d_rloc[i],nri,nnx);
   	 cudaMemcpyAsync(bgdata[i]+((it-ntNeg)/ratio)*nri,d_bgdata[i],nri*sizeof(float),cudaMemcpyDeviceToHost,compStream[i]);
    }

	if(it%ratio==0){
   	 cudaStreamSynchronize(transfStream[i]);
 	 if(it!=nt-1) cudaMemcpyAsync(d_Dpqa[i],d_Dpq[i],2*nnxz*sizeof(float),cudaMemcpyDeviceToDevice,compStream[i]);
   	 if(it>ratio){
      memcpy(Dp[i]+(it/ratio-1)*nnxz,Dpqa[i],nnxz*sizeof(float));
      memcpy(Dq[i]+(it/ratio-1)*nnxz,Dpqa[i]+nnxz,nnxz*sizeof(float));
//      omp_memcpy(Dp[i]+(it/ratio-1)*nnxz,Dpqa[i],nnxz*sizeof(float));
//      omp_memcpy(Dq[i]+(it/ratio-1)*nnxz,Dpqa[i]+nnxz,nnxz*sizeof(float));
//      tbb_memcpy(Dp[i]+(it/ratio-1)*nnxz,Dpqa[i],nnxz);
//      tbb_memcpy(Dq[i]+(it/ratio-1)*nnxz,Dpqa[i]+nnxz,nnxz);
     }
	 cudaStreamSynchronize(compStream[i]);
    }
    
	if(it%ratio==1 && it!=nt-ratio) cudaMemcpyAsync(Dpqa[i],d_Dpqa[i],2*nnxz*sizeof(float),cudaMemcpyDeviceToHost,transfStream[i]);
    
	float *pt=p0[i]; 
    p0[i]=p1[i];
    p1[i]=pt;
    pt=q0[i];
    q0[i]=q1[i];
    q1[i]=pt;
   }
   
   cudaStreamSynchronize(transfStream[i]);

    for(int it=0;it<nnt_data;++it){
     for(int ir=0;ir<nri;++ir){
      bgdata[i][it*nri+ir]-=data[it*nr+sloc[3+is*4]+ir];
      val[i]+=bgdata[i][it*nri+ir]*bgdata[i][it*nri+ir];
     }
    }

//   write("bgdata",bgdata[i],nri*nnt_data);
//   to_header("bgdata","n1",nri,"o1",0,"d1",1);
//   to_header("bgdata","n2",nnt_data,"o2",0,"d2",rate);
   
   cudaMemset(p0[i],0,nnxz*sizeof(float));
   cudaMemset(q0[i],0,nnxz*sizeof(float));
   cudaMemset(p1[i],0,nnxz*sizeof(float));
   cudaMemset(q1[i],0,nnxz*sizeof(float));
  
   cudaMalloc(&d_bgdata0[i],nri*sizeof(float));
   cudaMalloc(&d_bgdata1[i],nri*sizeof(float));
   cudaMemset(d_bgdata0[i],0,nri*sizeof(float));
   cudaMemset(d_bgdata1[i],0,nri*sizeof(float));
  
   cudaMemcpyAsync(d_bgdata0[i],bgdata[i]+(nnt_data-1)*nri,nri*sizeof(float),cudaMemcpyHostToDevice,compStream[i]);
 
   injectData<<<(nri+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(p0[i],q0[i],d_bgdata0[i],d_bgdata1[i],0.f,d_rloc[i],nri,nnx,dt2);
//   injectDipoleData<<<(nri+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(p0[i],q0[i],d_bgdata0[i],d_bgdata1[i],0.f,d_rloc[i],nri,nnx,dt2);
  
   float f=(nt-2.)/ratio;
   int i1=f;
   
   memcpy(Dpqb[i],Dp[i]+(nnt-3)*nnxz,nnxz*sizeof(float));
   memcpy(Dpqb[i]+nnxz,Dq[i]+(nnt-3)*nnxz,nnxz*sizeof(float));
   
   f=f-i1;
   gradientCrossCor<<<grid,block,0,compStream[i]>>>(d_gc11[i],d_gc13[i],d_gc33[i],p0[i],q0[i],d_Dp0[i],d_Dq0[i],d_Dp1[i],d_Dq1[i],f,nnx,nnz);
  
   for(int it=nt-3;it>=0;--it){
    backwardDC<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
//    forwardCD<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
   
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
      cudaMemcpyAsync(d_bgdata1[i],bgdata[i]+i1*nri,nri*sizeof(float),cudaMemcpyHostToDevice,compStream[i]);
	  cudaStreamSynchronize(compStream[i]);
      float *pt=d_bgdata0[i]; 
      d_bgdata0[i]=d_bgdata1[i];
      d_bgdata1[i]=pt;
     }
	 f=f-i1;
     injectData<<<(nri+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(p1[i],q1[i],d_bgdata0[i],d_bgdata1[i],f,d_rloc[i],nri,nnx,dt2);
//     injectDipoleData<<<(nri+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X,0,compStream[i]>>>(p1[i],q1[i],d_bgdata0[i],d_bgdata1[i],f,d_rloc[i],nri,nnx,dt2);
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
  
	float *pt=p0[i]; 
    p0[i]=p1[i];
    p1[i]=pt;
    pt=q0[i];
    q0[i]=q1[i];
    q1[i]=pt;
   }
   
   cudaFreeHost(bgdata[i]);
   cudaFree(d_bgdata[i]);cudaFree(d_rloc[i]);
   cudaFree(d_bgdata0[i]);cudaFree(d_bgdata1[i]);
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

 double objfunc=0;
 for(int i=0;i<nGPUs;++i){
  objfunc+=val[i];
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
  cudaFree(d_taper[i]);
  cudaFree(p0[i]);cudaFree(p1[i]);cudaFree(q0[i]);cudaFree(q1[i]);
  cudaFree(d_Dpq[i]);cudaFree(d_Dpqa[i]);cudaFree(d_Dpqb[i]);
  delete []Dp[i];delete []Dq[i]; 
  cudaFree(d_gc11[i]);cudaFree(d_gc13[i]);cudaFree(d_gc33[i]);
  cudaFreeHost(tgc11[i]);cudaFreeHost(tgc13[i]);cudaFreeHost(tgc33[i]);
  cudaStreamDestroy(compStream[i]);
  cudaStreamDestroy(transfStream[i]);
  cudaFreeHost(Dpqa[i]);cudaFreeHost(Dpqb[i]);
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"gpu %d error %s\n",GPUs[i],cudaGetErrorString(e));
 }
 
 delete []bgdata;
 delete []tbgdata;delete []tdata;delete []tres;delete []tadjsou;
 delete []d_rloc;
 delete []d_c11;delete []d_c13;delete []d_c33;
 delete []d_taper;
 delete []d_bgdata;
 delete []p0;delete []p1;delete []q0;delete []q1;
 delete []d_Dpq;delete []d_Dpqa;delete []d_Dpqb;
 delete []Dp;delete []Dq;
 delete []d_gc11;delete []d_gc13;delete []d_gc33;
 delete []d_bgdata0;delete []d_bgdata1;
 delete []d_Dp0;delete []d_Dp1;delete []d_Dq0;delete []d_Dq1;
 delete []tgc11;delete []tgc13;delete []tgc33;
 delete []Dpqa;
 delete []Dpqb;

// cudaProfilerStop();

 return 0.5f*objfunc;
}

void waveletGradient_f(float *gwavelet,const float *d0,const float *c11,const float *c13,const float *c33,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate){
 
 int ratio=rate/dt+0.5;
 int nnt=(nt-1)/ratio+1;
 int nnx=nx+2*npad,nnz=nz+2*npad;
 float dx2=dx*dx,dz2=dz*dz,dt2=dt*dt;
 
 memset(gwavelet,0,nt*sizeof(float));
 
 int nGPUs;
 cudaGetDeviceCount(&nGPUs);
 //fprintf(stderr,"Total # GPUs = %d\n",nGPUs);

 float **d=new float*[nGPUs]();
 int **d_rloc=new int*[nGPUs]();
 float **d_c11=new float*[nGPUs]();
 float **d_c13=new float*[nGPUs]();
 float **d_c33=new float*[nGPUs]();
 float **d_taper=new float*[nGPUs]();
 float **d_d=new float*[nGPUs]();
 float **p0=new float*[nGPUs]();
 float **q0=new float*[nGPUs]();
 float **p1=new float*[nGPUs]();
 float **q1=new float*[nGPUs]();
 float **tp=new float*[nGPUs]();
 float **tq=new float*[nGPUs]();
 float **d_gwavelet=new float*[nGPUs]();
 float **d_d0=new float*[nGPUs]();
 float **d_d1=new float*[nGPUs]();
 float **d_d00=new float*[nGPUs]();
 float **d_d01=new float*[nGPUs]();
 float **tgwavelet=new float*[nGPUs]();
 
 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(i);
  
  cudaMalloc(&d_c11[i],nnx*nnz*sizeof(float));
  cudaMalloc(&d_c13[i],nnx*nnz*sizeof(float));
  cudaMalloc(&d_c33[i],nnx*nnz*sizeof(float));
  cudaMemcpy(d_c11[i],c11,nnx*nnz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_c13[i],c13,nnx*nnz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_c33[i],c33,nnx*nnz*sizeof(float),cudaMemcpyHostToDevice);
  
  cudaMalloc(&d_taper[i],nnx*nnz*sizeof(float));
  cudaMemcpy(d_taper[i],taper,nnx*nnz*sizeof(float),cudaMemcpyHostToDevice);
 
  cudaMalloc(&p0[i],nnx*nnz*sizeof(float)); 
  cudaMalloc(&p1[i],nnx*nnz*sizeof(float)); 
  cudaMalloc(&q0[i],nnx*nnz*sizeof(float)); 
  cudaMalloc(&q1[i],nnx*nnz*sizeof(float)); 
  cudaMalloc(&tp[i],nnx*nnz*sizeof(float)); cudaMemset(tp[i],0,nnx*nnz*sizeof(float));
  cudaMalloc(&tq[i],nnx*nnz*sizeof(float)); cudaMemset(tq[i],0,nnx*nnz*sizeof(float));
  
  cudaMalloc(&d_gwavelet[i],nt*sizeof(float));
  
  tgwavelet[i]=new float[nt]();
 }

 int npasses=(ns+nGPUs-1)/nGPUs;
 int shotLeft=ns;

 for(int pass=0;pass<npasses;++pass){
  int nGPUsNeed=min(shotLeft,nGPUs);
  //fprintf(stderr,"Pass %d, # GPUs %d\n",pass,nGPUsNeed);
  
  #pragma omp parallel for num_threads(nGPUsNeed)
  for(int i=0;i<nGPUsNeed;++i){
   cudaSetDevice(i);

   int is=pass*nGPUs+i;
   int slocxz=sloc[0+is*4]+sloc[1+is*4]*nnx;
   int nri=sloc[2+is*4];

   cudaMalloc(&d_rloc[i],2*nri*sizeof(int));
   cudaMemcpy(d_rloc[i],rloc+2*sloc[3+is*4],2*nri*sizeof(int),cudaMemcpyHostToDevice);
   
   d[i]=new float[nri*nnt]();
   
   cudaMalloc(&d_d[i],nri*sizeof(float));

   dim3 block(BLOCK_DIM_X,BLOCK_DIM_Y);
   dim3 grid((nnx-2*RADIUS+BLOCK_DIM_X-1)/BLOCK_DIM_X,(nnz-2*RADIUS+BLOCK_DIM_Y-1)/BLOCK_DIM_Y);

   cudaMemset(p0[i],0,nnx*nnz*sizeof(float));
   cudaMemset(q0[i],0,nnx*nnz*sizeof(float));
   cudaMemset(p1[i],0,nnx*nnz*sizeof(float));
   cudaMemset(q1[i],0,nnx*nnz*sizeof(float));
 
   cudaMemset(d_gwavelet[i],0,nt*sizeof(float));
   
   cudaMalloc(&d_d0[i],nri*sizeof(float));
   cudaMalloc(&d_d1[i],nri*sizeof(float));
  
   cudaMemcpy(d_d0[i],d0+(nnt-1)*nr+sloc[3+is*4],nri*sizeof(float),cudaMemcpyHostToDevice);
   
   injectData<<<(nri+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X>>>(p0[i],q0[i],d_d0[i],d_d1[i],0.f,d_rloc[i],nri,nnx,dt2);
//   injectDipoleData<<<(nri+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X>>>(p0[i],q0[i],d_d0[i],d_d1[i],0.f,d_rloc[i],nri,nnx,dt2);
  
   abc<<<grid,block>>>(p0[i],q0[i],d_taper[i],nnx,nnz);
  
   extractWavelet<<<1,1>>>(d_gwavelet[i]+nt-2,p0[i],q0[i],slocxz,nnx,dt2);
   
   for(int it=nt-3;it>=0;--it){
    //fprintf(stderr,"Time step it=%d\n",it);
    
//    backwardDC<<<grid,block,0,compStream[i]>>>(p1[i],q1[i],p0[i],q0[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
    forwardCD<<<grid,block>>>(p1[i],q1[i],p0[i],q0[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
  
    float f=(it+1.)/ratio;
    int i1=f;
    if((it+2)%ratio==0){
     cudaMemcpy(d_d1[i],d0+i1*nr+sloc[3+is*4],nri*sizeof(float),cudaMemcpyHostToDevice);
     float *pt=d_d0[i]; 
     d_d0[i]=d_d1[i];
     d_d1[i]=pt;
    }
    f=f-i1;
    injectData<<<(nri+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X>>>(p1[i],q1[i],d_d0[i],d_d1[i],f,d_rloc[i],nri,nnx,dt2);
  
    abc<<<grid,block>>>(p0[i],q0[i],d_taper[i],nnx,nnz);
    abc<<<grid,block>>>(p1[i],q1[i],d_taper[i],nnx,nnz);
  
    extractWavelet<<<1,1>>>(d_gwavelet[i]+it,p1[i],q1[i],slocxz,nnx,dt2);
  
    float *pt=p0[i]; 
    p0[i]=p1[i];
    p1[i]=pt;
    pt=q0[i];
    q0[i]=q1[i];
    q1[i]=pt;
   }
   cudaMemcpy(tgwavelet[i],d_gwavelet[i],nt*sizeof(float),cudaMemcpyDeviceToHost);
  
   cudaFree(d_d[i]);cudaFree(d_rloc[i]);
   cudaFree(d_d0[i]);cudaFree(d_d1[i]);
  }
  
  for(int i=0;i<nGPUsNeed;++i){
   #pragma omp parallel for num_threads(16) shared(i)
   for(int it=0;it<nt;++it){
    gwavelet[it]+=tgwavelet[i][it];
   }
  }

  shotLeft-=nGPUsNeed;
 }

 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(i);
  cudaFree(d_c11[i]);cudaFree(d_c13[i]);cudaFree(d_c33[i]); 
  cudaFree(d_taper[i]);
  cudaFree(p0[i]);cudaFree(p1[i]);cudaFree(q0[i]);cudaFree(q1[i]);cudaFree(tp[i]);cudaFree(tq[i]);
  cudaFree(d_gwavelet[i]);
  delete []tgwavelet[i];
 }
 
 delete []d_rloc;
 delete []d_c11;delete []d_c13;delete []d_c33;
 delete []d_taper;
 delete []d_d;
 delete []p0;delete []p1;delete []q0;delete []q1;delete []tp;delete []tq;
 delete []d_gwavelet;
 delete []d_d0;delete []d_d1;
 delete []tgwavelet;

 return;
}
