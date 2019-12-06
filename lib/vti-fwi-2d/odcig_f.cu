#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include "myio.h"
#include "mylib.h"
#include "init.h"
#include "wave.h"
#include "kernels.h"

#include <vector>

void odcig_f(float *image,const float *data,const float *c11,const float *c13,const float *c33,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nhx,int nt,int npad,float dx,float dz,float dt,float rate,float ot){

 fprintf(stderr,"Starting migration...\n");

 int ratio=rate/dt+0.5f;
 int ntNeg=std::round(abs(ot)/dt);
 int nnt=(nt-1)/ratio+1;
 int nnt_data=(nt-ntNeg-1)/ratio+1;
 int nnx=nx+2*npad,nnz=nz+2*npad;
 float dx2=dx*dx,dz2=dz*dz,dt2=dt*dt;
 
 memset(image,0,nnx*nnz*(2*nhx+1)*sizeof(float));
 
 std::vector<int> GPUs;
 get_array("gpu",GPUs);
 int nGPUs=GPUs.size();
 fprintf(stderr,"Total # GPUs = %d\n",nGPUs);
 fprintf(stderr,"GPUs used are:\n");
 for(int i=0;i<nGPUs;i++) fprintf(stderr,"%d",GPUs[i]);
 fprintf(stderr,"\n");

 float **sourceWavefield=new float*[nGPUs]();
 int **d_rloc=new int*[nGPUs]();
 float **d_c11=new float*[nGPUs]();
 float **d_c13=new float*[nGPUs]();
 float **d_c33=new float*[nGPUs]();
 float **d_taper=new float*[nGPUs]();
 float **d_sourceWavefield=new float*[nGPUs]();
 float **p0=new float*[nGPUs]();
 float **q0=new float*[nGPUs]();
 float **p1=new float*[nGPUs]();
 float **q1=new float*[nGPUs]();
 float **d_image=new float*[nGPUs]();
 float **d_sourceWavefield0=new float*[nGPUs]();
 float **d_sourceWavefield1=new float*[nGPUs]();
 float **d_data0=new float*[nGPUs]();
 float **d_data1=new float*[nGPUs]();
 float **images=new float*[nGPUs]();
 
 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(GPUs[i]);
  
  sourceWavefield[i]=new float[nnx*nnz*nnt]();
  
  cudaMalloc(&d_c11[i],nnx*nnz*sizeof(float));
  cudaMalloc(&d_c13[i],nnx*nnz*sizeof(float));
  cudaMalloc(&d_c33[i],nnx*nnz*sizeof(float));
  cudaMemcpy(d_c11[i],c11,nnx*nnz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_c13[i],c13,nnx*nnz*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_c33[i],c33,nnx*nnz*sizeof(float),cudaMemcpyHostToDevice);
  
  cudaMalloc(&d_taper[i],nnx*nnz*sizeof(float));
  cudaMemcpy(d_taper[i],taper,nnx*nnz*sizeof(float),cudaMemcpyHostToDevice);
 
  cudaMalloc(&d_sourceWavefield[i],nnx*nnz*sizeof(float));
  cudaMemset(d_sourceWavefield[i],0,nnx*nnz*sizeof(float));
 
  cudaMalloc(&p0[i],nnx*nnz*sizeof(float)); 
  cudaMalloc(&p1[i],nnx*nnz*sizeof(float)); 
  cudaMalloc(&q0[i],nnx*nnz*sizeof(float)); 
  cudaMalloc(&q1[i],nnx*nnz*sizeof(float)); 
  
  cudaMalloc(&d_image[i],nnx*nnz*(2*nhx+1)*sizeof(float));
  cudaMemset(d_image[i],0,nnx*nnz*(2*nhx+1)*sizeof(float));
  
  cudaMalloc(&d_sourceWavefield0[i],nnx*nnz*sizeof(float));
  cudaMalloc(&d_sourceWavefield1[i],nnx*nnz*sizeof(float));
  cudaMemset(d_sourceWavefield0[i],0,nnx*nnz*sizeof(float));
  cudaMemset(d_sourceWavefield1[i],0,nnx*nnz*sizeof(float));
 
  images[i]=new float[nnx*nnz*(2*nhx+1)]();
 }

 int npasses=(ns+nGPUs-1)/nGPUs;
 int shotLeft=ns;

 for(int pass=0;pass<npasses;++pass){
  int nGPUsNeed=min(shotLeft,nGPUs);
  fprintf(stderr,"Pass %d, # GPUs = %d\n",pass,nGPUsNeed);
  
  #pragma omp parallel for num_threads(nGPUsNeed)
  for(int i=0;i<nGPUsNeed;++i){
  cudaSetDevice(GPUs[i]);

   int is=pass*nGPUs+i;
   int slocxz=sloc[0+is*4]+sloc[1+is*4]*nnx;

   cudaMalloc(&d_rloc[i],2*sloc[2+is*4]*sizeof(int));
   cudaMemcpy(d_rloc[i],rloc+2*sloc[3+is*4],2*sloc[2+is*4]*sizeof(int),cudaMemcpyHostToDevice);

   dim3 block(BLOCK_DIM_X,BLOCK_DIM_Y);
   dim3 grid((nnx-2*RADIUS+BLOCK_DIM_X-1)/BLOCK_DIM_X,(nnz-2*RADIUS+BLOCK_DIM_Y-1)/BLOCK_DIM_Y);

   cudaMemset(p0[i],0,nnx*nnz*sizeof(float));
   cudaMemset(q0[i],0,nnx*nnz*sizeof(float));
   cudaMemset(p1[i],0,nnx*nnz*sizeof(float));
   cudaMemset(q1[i],0,nnx*nnz*sizeof(float));
 
   injectDipoleSource<<<1,1>>>(p1[i],q1[i],dt2*wavelet[0],slocxz,nnx);
  
   abc<<<grid,block>>>(p1[i],q1[i],d_taper[i],nnx,nnz);
  
   if(ratio==1){
    recordWavefieldSlice<<<grid,block>>>(d_sourceWavefield[i],p1[i],q1[i],nnx,nnz);
    cudaMemcpy(sourceWavefield[i]+nnx*nnz,d_sourceWavefield[i],nnx*nnz*sizeof(float),cudaMemcpyDeviceToHost);
   }
   
   for(int it=2;it<nt;++it){
    //fprintf(stderr,"Time step it=%d\n",it);
  
    forwardCD<<<grid,block>>>(p0[i],q0[i],p1[i],q1[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
  
    injectDipoleSource<<<1,1>>>(p0[i],q0[i],wavelet[it-1],slocxz,nnx);
  
    abc<<<grid,block>>>(p1[i],q1[i],p0[i],q0[i],d_taper[i],nnx,nnz);
    
    if(it%ratio==0){
     recordWavefieldSlice<<<grid,block>>>(d_sourceWavefield[i],p0[i],q0[i],nnx,nnz);
     cudaMemcpy(sourceWavefield[i]+(it/ratio)*nnx*nnz,d_sourceWavefield[i],nnx*nnz*sizeof(float),cudaMemcpyDeviceToHost);
    }
  
    float *pt=p0[i]; 
    p0[i]=p1[i];
    p1[i]=pt;
    pt=q0[i];
    q0[i]=q1[i];
    q1[i]=pt;
   }

   cudaMemset(p0[i],0,nnx*nnz*sizeof(float));
   cudaMemset(q0[i],0,nnx*nnz*sizeof(float));
   cudaMemset(p1[i],0,nnx*nnz*sizeof(float));
   cudaMemset(q1[i],0,nnx*nnz*sizeof(float));
  
   cudaMalloc(&d_data0[i],sloc[2+is*4]*sizeof(float));
   cudaMalloc(&d_data1[i],sloc[2+is*4]*sizeof(float));
   cudaMemset(d_data0[i],0,sloc[2+is*4]*sizeof(float));
   cudaMemset(d_data1[i],0,sloc[2+is*4]*sizeof(float));
  
   cudaMemcpy(d_data0[i],data+(nnt_data-1)*nr+sloc[3+is*4],sloc[2+is*4]*sizeof(float),cudaMemcpyHostToDevice);
   injectDipoleData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X>>>(p0[i],q0[i],d_data0[i],d_data1[i],0.f,d_rloc[i],sloc[2+is*4],nnx,dt2);
  
   abc<<<grid,block>>>(p0[i],q0[i],d_taper[i],nnx,nnz);
  
   float f=(nt-2.)/ratio;
   int i1=f,i2=i1+1;
   cudaMemcpy(d_sourceWavefield0[i],sourceWavefield[i]+i1*nnx*nnz,nnx*nnz*sizeof(float),cudaMemcpyHostToDevice);
   cudaMemcpy(d_sourceWavefield1[i],sourceWavefield[i]+i2*nnx*nnz,nnx*nnz*sizeof(float),cudaMemcpyHostToDevice);
   f=f-i1;
   extendedImagingCrossCor<<<grid,block>>>(d_image[i],p0[i],q0[i],d_sourceWavefield0[i],d_sourceWavefield1[i],f,nnx,nnz,nhx);
  
   for(int it=nt-3;it>=0;--it){
    //fprintf(stderr,"Time step it=%d\n",it);

//    backwardDC<<<grid,block>>>(p1[i],q1[i],p0[i],q0[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
    forwardCD<<<grid,block>>>(p1[i],q1[i],p0[i],q0[i],d_c11[i],d_c13[i],d_c33[i],dx2,dz2,dt2,nnx,nnz);
	
	if(it>=ntNeg){
     f=(it-ntNeg+1.)/ratio;
     i1=f;
     if((it-ntNeg+2)%ratio==0){
      cudaMemcpy(d_data1[i],data+i1*nr+sloc[3+is*4],sloc[2+is*4]*sizeof(float),cudaMemcpyHostToDevice);     
 	 float *pt=d_data0[i];
      d_data0[i]=d_data1[i];
      d_data1[i]=pt;
     }
     f=f-i1;
     injectDipoleData<<<(sloc[2+is*4]+BLOCK_DIM_X-1)/BLOCK_DIM_X,BLOCK_DIM_X>>>(p1[i],q1[i],d_data0[i],d_data1[i],f,d_rloc[i],sloc[2+is*4],nnx,dt2);
    }

    abc<<<grid,block>>>(p1[i],q1[i],p0[i],q0[i],d_taper[i],nnx,nnz);
  
    f=(float)it/ratio;
    i1=f;
    if((it+1)%ratio==0){
     cudaMemcpy(d_sourceWavefield1[i],sourceWavefield[i]+i1*nnx*nnz,nnx*nnz*sizeof(float),cudaMemcpyHostToDevice);
     float *pt=d_sourceWavefield0[i]; 
     d_sourceWavefield0[i]=d_sourceWavefield1[i];
     d_sourceWavefield1[i]=pt;
    }
    f=f-i1;
    extendedImagingCrossCor<<<grid,block>>>(d_image[i],p1[i],q1[i],d_sourceWavefield0[i],d_sourceWavefield1[i],f,nnx,nnz,nhx);
  
    float *pt=p0[i]; 
    p0[i]=p1[i];
    p1[i]=pt;
    pt=q0[i];
    q0[i]=q1[i];
    q1[i]=pt;
   }
   cudaFree(d_rloc[i]);cudaFree(d_data0[i]);cudaFree(d_data1[i]);
  }
  
  shotLeft-=nGPUsNeed;
 }

 #pragma omp parallel for num_threads(nGPUs)
 for(int i=0;i<nGPUs;i++){
   cudaSetDevice(GPUs[i]);
   cudaMemcpy(images[i],d_image[i],nnx*nnz*(2*nhx+1)*sizeof(float),cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();
 }
  
  for(int i=0;i<nGPUs;++i){
   #pragma omp parallel for num_threads(16) shared(i)
   for(size_t ixz=0;ixz<nnx*nnz*(2*nhx+1);++ixz){
    image[ixz]+=images[i][ixz];
   }
  }

 for(int i=0;i<nGPUs;++i){
  cudaSetDevice(GPUs[i]);
  delete []sourceWavefield[i];
  cudaFree(d_c11[i]);cudaFree(d_c13[i]);cudaFree(d_c33[i]); 
  cudaFree(d_taper[i]);
  cudaFree(d_sourceWavefield[i]);
  cudaFree(p0[i]);cudaFree(p1[i]);cudaFree(q0[i]);cudaFree(q1[i]);
  cudaFree(d_image[i]);
  cudaFree(d_sourceWavefield0[i]);cudaFree(d_sourceWavefield1[i]);
  delete []images[i];
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"gpu %d error %s\n",GPUs[i],cudaGetErrorString(e));
 }
 
 delete []sourceWavefield;
 delete []d_rloc;
 delete []d_c11;delete []d_c13;delete []d_c33;
 delete []d_taper;
 delete []d_sourceWavefield;
 delete []p0;delete []p1;delete []q0;delete []q1;
 delete []d_image;
 delete []d_sourceWavefield0;delete []d_sourceWavefield1;delete []d_data0;delete []d_data1;
 delete []images;

 return;
}
