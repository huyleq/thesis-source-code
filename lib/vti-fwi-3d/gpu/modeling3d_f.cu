#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"

using namespace std;

void modeling3d_f(float soulocX,float soulocY,float soulocZ,float *wavelet,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt){

 vector<int> GPUs;
 get_array("gpu",GPUs);
 int NGPU=GPUs.size();
 fprintf(stderr,"Total # GPUs = %d\n",NGPU);
 fprintf(stderr,"GPUs used are:\n");
 for(int i=0;i<NGPU;i++) fprintf(stderr,"%d ",GPUs[i]);
 fprintf(stderr,"\n");

 float dx2=dx*dx,dy2=dy*dy,dz2=dz*dz,dt2=dt*dt;
 float dt2dx2=dt2/dx2,dt2dy2=dt2/dy2,dt2dz2=dt2/dz2;
 long long nxyz=nx*ny*nz;
 int nxy=nx*ny;

// float *damping=new float[nxy];
// init_abc(damping,nx,ny,npad);
 float *damping=new float[nxy+nz];
 init_abc(damping,nx,ny,nz,npad);
 float **d_damping=new float*[NGPU]();

 float *prevSigmaX=new float[nxyz];
 float *curSigmaX=new float[nxyz];
 float *prevSigmaZ=new float[nxyz];
 float *curSigmaZ=new float[nxyz];

 size_t nElemBlock=HALF_STENCIL*nxy;
 size_t nByteBlock=nElemBlock*sizeof(float);
 int nb=nz/HALF_STENCIL;

 float *h_c11[2],*h_c13[2],*h_c33[2];
 float *h_prevSigmaX[2],*h_curSigmaX[2],*h_SigmaX4[2],*h_SigmaX5[2];
 float *h_prevSigmaZ[2],*h_curSigmaZ[2],*h_SigmaZ4[2],*h_SigmaZ5[2];

 for(int i=0;i<2;++i){
  cudaHostAlloc(&h_c11[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_c13[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_c33[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_prevSigmaX[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_curSigmaX[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaX4[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaX5[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_prevSigmaZ[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_curSigmaZ[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaZ4[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaZ5[i],nByteBlock,cudaHostAllocDefault);
 }

 const int nbuffSigma=NUPDATE+2;
 
 float ****d_SigmaX=new float ***[NGPU]();
 float ****d_SigmaZ=new float ***[NGPU]();
 
 const int nbuffCij=NUPDATE+4;
 float ***d_c11=new float**[NGPU]();
 float ***d_c13=new float**[NGPU]();
 float ***d_c33=new float**[NGPU]();
 
 cudaStream_t *transfInStream=new cudaStream_t[1]();
 cudaStream_t *transfOutStream=new cudaStream_t[NGPU]();
 cudaStream_t *computeStream=new cudaStream_t[NGPU]();
 
 dim3 block(BLOCK_DIM,BLOCK_DIM);
 dim3 grid((nx-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM,(ny-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM);
 
 for(int gpu=0;gpu<NGPU;gpu++){
  cudaSetDevice(GPUs[gpu]);
  
//  cudaMalloc(&d_damping[gpu],nxy*sizeof(float));
//  cudaMemcpy(d_damping[gpu],damping,nxy*sizeof(float),cudaMemcpyHostToDevice);
  cudaMalloc(&d_damping[gpu],(nxy+nz)*sizeof(float));
  cudaMemcpy(d_damping[gpu],damping,(nxy+nz)*sizeof(float),cudaMemcpyHostToDevice);

  d_SigmaX[gpu]=new float**[nbuffSigma]();
  d_SigmaZ[gpu]=new float**[nbuffSigma]();
  for(int i=0;i<nbuffSigma;++i){
   d_SigmaX[gpu][i]=new float*[4]();
   d_SigmaZ[gpu][i]=new float*[4]();
   for(int j=0;j<4;++j){
    cudaMalloc(&d_SigmaX[gpu][i][j],nByteBlock); 
    cudaMalloc(&d_SigmaZ[gpu][i][j],nByteBlock); 
   }
  }

  d_c11[gpu]=new float*[nbuffCij]();
  d_c13[gpu]=new float*[nbuffCij]();
  d_c33[gpu]=new float*[nbuffCij]();
  for(int i=0;i<nbuffCij;++i){
   cudaMalloc(&d_c11[gpu][i],nByteBlock);
   cudaMalloc(&d_c13[gpu][i],nByteBlock);
   cudaMalloc(&d_c33[gpu][i],nByteBlock);
  }
 
  if(gpu==0) cudaStreamCreate(&transfInStream[gpu]);
  cudaStreamCreate(&computeStream[gpu]);
  cudaStreamCreate(&transfOutStream[gpu]);
  
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"GPU %d alloc error %s\n",gpu,cudaGetErrorString(e));
 }

 vector<thread> threads;
 int pipelen=NGPU*(NUPDATE+3)+3;
 int nround=(nt-2)/(NGPU*NUPDATE);
 int roundlen=max(pipelen,nb);
 int nroundlen=nround*roundlen;;
 int nk=(nround-1)*roundlen+pipelen+nb-1;

 int souIndexX=(soulocX-ox)/dx;
 int souIndexY=(soulocY-oy)/dy;
 int souIndexZ=(soulocZ-oz)/dz;
 int souIndex=souIndexX+souIndexY*nx+souIndexZ*nxy;
 int souIndexBlock=souIndexX+souIndexY*nx+(souIndexZ%HALF_STENCIL)*nxy;
 int souBlock=souIndexZ/HALF_STENCIL;

 memset(prevSigmaX,0,nxyz*sizeof(float));
 memset(curSigmaX,0,nxyz*sizeof(float));
 memset(prevSigmaZ,0,nxyz*sizeof(float));
 memset(curSigmaZ,0,nxyz*sizeof(float));
 
 //injecting source at time 0 to wavefields at time 1
 float temp=dt2*wavelet[0];
 curSigmaX[souIndex]=temp;
 curSigmaZ[souIndex]=temp;
// curSigmaX[souIndex]=1.;
// curSigmaZ[souIndex]=1.;
 
 for(int gpu=0;gpu<NGPU;gpu++){
  cudaSetDevice(GPUs[gpu]);
  
  for(int i=0;i<nbuffSigma;++i){
   for(int j=0;j<4;++j){
    cudaMemset(d_SigmaX[gpu][i][j],0,nByteBlock);
    cudaMemset(d_SigmaZ[gpu][i][j],0,nByteBlock);
   }
  }

  for(int i=0;i<nbuffCij;++i){
   cudaMemset(d_c11[gpu][i],0,nByteBlock);
   cudaMemset(d_c13[gpu][i],0,nByteBlock);
   cudaMemset(d_c33[gpu][i],0,nByteBlock);
  }
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"GPU %d alloc error %s\n",gpu,cudaGetErrorString(e));
 }
 
 for(int k=0;k<nk;k++){
   if(k<nroundlen){
    int ib=k%roundlen;
    if(ib<nb){
        size_t ibn=ib*nElemBlock; 
        int k2=k%2;
	    threads.push_back(thread(memcpyCpuToCpu3,h_c11[k2],c11+ibn,h_c13[k2],c13+ibn,h_c33[k2],c33+ibn,nByteBlock));
		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaX[k2],prevSigmaX+ibn,h_curSigmaX[k2],curSigmaX+ibn,nByteBlock));
		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaZ[k2],prevSigmaZ+ibn,h_curSigmaZ[k2],curSigmaZ+ibn,nByteBlock));
    }
   }
   
   if(k>0 && k<=nroundlen){
    int ib=(k-1)%roundlen;
    if(ib<nb){
      int k12=(k-1)%2,kn=k%nbuffCij,k4=k%4;
      cudaSetDevice(GPUs[0]);
      memcpyCpuToGpu2(d_SigmaX[0][0][k4],h_prevSigmaX[k12],d_SigmaX[0][1][k4],h_curSigmaX[k12],nByteBlock,transfInStream);
      memcpyCpuToGpu2(d_SigmaZ[0][0][k4],h_prevSigmaZ[k12],d_SigmaZ[0][1][k4],h_curSigmaZ[k12],nByteBlock,transfInStream);
      memcpyCpuToGpu3(d_c11[0][kn],h_c11[k12],d_c13[0][kn],h_c13[k12],d_c33[0][kn],h_c33[k12],nByteBlock,transfInStream);
    }
   }
  
   for(int gpu=0;gpu<NGPU;gpu++){
    int kgpu=k-gpu*(NUPDATE+3);
    cudaSetDevice(GPUs[gpu]);

    if(kgpu>2 && kgpu<=NUPDATE+1+nroundlen){
     for(int i=0;i<NUPDATE;i++){
      int ib=(kgpu-3-i)%roundlen;
      int iround=(kgpu-3-i)/roundlen;
      if(ib>=0 && ib<nb && iround>=0 && iround<nround){
       int it=iround*NGPU*NUPDATE+gpu*NUPDATE+2+i;
       int ki=kgpu-i,ki14=(ki-1)%4,ki24=(ki-2)%4,ki34=(ki-3)%4,ki2n=(ki-2)%nbuffCij;

       if(ib==0){
        forwardKernelTopBlock<<<grid,block,0,computeStream[gpu]>>>(d_SigmaX[gpu][i+2][ki24],d_SigmaX[gpu][i+1][ki24],d_SigmaX[gpu][i][ki24],d_SigmaZ[gpu][i+2][ki24],d_SigmaZ[gpu][i+1][ki34],d_SigmaZ[gpu][i+1][ki24],d_SigmaZ[gpu][i+1][ki14],d_SigmaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
       }
       else if(ib==nb-1){
        forwardKernelBottomBlock<<<grid,block,0,computeStream[gpu]>>>(d_SigmaX[gpu][i+2][ki24],d_SigmaX[gpu][i+1][ki24],d_SigmaX[gpu][i][ki24],d_SigmaZ[gpu][i+2][ki24],d_SigmaZ[gpu][i+1][ki34],d_SigmaZ[gpu][i+1][ki24],d_SigmaZ[gpu][i+1][ki14],d_SigmaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
       }
       else{
        forwardKernel<<<grid,block,0,computeStream[gpu]>>>(d_SigmaX[gpu][i+2][ki24],d_SigmaX[gpu][i+1][ki24],d_SigmaX[gpu][i][ki24],d_SigmaZ[gpu][i+2][ki24],d_SigmaZ[gpu][i+1][ki34],d_SigmaZ[gpu][i+1][ki24],d_SigmaZ[gpu][i+1][ki14],d_SigmaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
       }
       
       if(ib==souBlock){
        float source=dt2*wavelet[it-1];
        injectSource<<<1,1,0,computeStream[gpu]>>>(d_SigmaX[gpu][i+2][ki24],d_SigmaZ[gpu][i+2][ki24],source,souIndexBlock);
        }

        int iz=ib*HALF_STENCIL;
        if(iz<npad || iz+HALF_STENCIL-1>nz-npad) abcXYZ<<<grid,block,0,computeStream[gpu]>>>(ib,nx,ny,nz,npad,d_SigmaX[gpu][i+2][ki24],d_SigmaX[gpu][i+1][ki24],d_SigmaZ[gpu][i+2][ki24],d_SigmaZ[gpu][i+1][ki24],d_damping[gpu]);
        else abcXY<<<grid,block,0,computeStream[gpu]>>>(ib,nx,ny,nz,npad,d_SigmaX[gpu][i+2][ki24],d_SigmaX[gpu][i+1][ki24],d_SigmaZ[gpu][i+2][ki24],d_SigmaZ[gpu][i+1][ki24],d_damping[gpu]);
     	}
      }
     }
    
    if(kgpu>NUPDATE+3 && kgpu<=NUPDATE+3+nroundlen){
     int ib=(kgpu-NUPDATE-4)%roundlen;
     if(ib<nb){
	     if(NGPU>1 && gpu<NGPU-1){
          int n2=nbuffSigma-2,n1=nbuffSigma-1,kn3=kgpu-NUPDATE-3,kn34=kn3%4,kn3n=kn3%nbuffCij;
	      memcpyGpuToGpu2(d_SigmaX[gpu+1][0][kn34],d_SigmaX[gpu][n2][kn34],d_SigmaX[gpu+1][1][kn34],d_SigmaX[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToGpu2(d_SigmaZ[gpu+1][0][kn34],d_SigmaZ[gpu][n2][kn34],d_SigmaZ[gpu+1][1][kn34],d_SigmaZ[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToGpu3(d_c11[gpu+1][kn3n],d_c11[gpu][kn3n],d_c13[gpu+1][kn3n],d_c13[gpu][kn3n],d_c33[gpu+1][kn3n],d_c33[gpu][kn3n],nByteBlock,transfOutStream+gpu);
	     }
	     else{
          int n2=nbuffSigma-2,n1=nbuffSigma-1,k2=k%2,kn34=(kgpu-NUPDATE-3)%4;
	      memcpyGpuToCpu2(h_SigmaX4[k2],d_SigmaX[gpu][n2][kn34],h_SigmaX5[k2],d_SigmaX[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToCpu2(h_SigmaZ4[k2],d_SigmaZ[gpu][n2][kn34],h_SigmaZ5[k2],d_SigmaZ[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
	     }
     }
    }
    
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) fprintf(stderr,"GPU %d prop error %s\n",gpu,cudaGetErrorString(e));
   }
   
   if(k>pipelen-2 and k<=pipelen-2+nroundlen){
    int ib=(k-pipelen+1)%roundlen;
    if(ib<nb){
        size_t ibn=ib*nElemBlock; 
        int k12=(k-1)%2;
	    memcpyCpuToCpu2(prevSigmaX+ibn,h_SigmaX4[k12],curSigmaX+ibn,h_SigmaX5[k12],nByteBlock);
	    memcpyCpuToCpu2(prevSigmaZ+ibn,h_SigmaZ4[k12],curSigmaZ+ibn,h_SigmaZ5[k12],nByteBlock);
    }
   }
  
   for(int gpu=0;gpu<NGPU;gpu++){
    cudaSetDevice(GPUs[gpu]);
    cudaDeviceSynchronize();
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"GPU %d synch error %s\n",gpu,cudaGetErrorString(e));
   }
   
   for(int i=0;i<threads.size();++i) threads[i].join();
   threads.erase(threads.begin(),threads.end());
 }

 for(int gpu=0;gpu<NGPU;gpu++){
  cudaSetDevice(GPUs[gpu]);
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"GPU %d dealloc error %s\n",gpu,cudaGetErrorString(e));
 }

 write("wavefield",curSigmaX,nxyz);
 to_header("wavefield","n1",nx,"o1",ox,"d1",dx);
 to_header("wavefield","n2",ny,"o2",oy,"d2",dy);
 to_header("wavefield","n3",nz,"o3",oz,"d3",dz);

 delete []prevSigmaX;delete []curSigmaX;
 delete []prevSigmaZ;delete []curSigmaZ;

 for(int i=0;i<2;++i){
  cudaFreeHost(h_c11[i]);
  cudaFreeHost(h_c13[i]);
  cudaFreeHost(h_c33[i]);
  cudaFreeHost(h_prevSigmaX[i]);
  cudaFreeHost(h_curSigmaX[i]);
  cudaFreeHost(h_SigmaX4[i]);
  cudaFreeHost(h_SigmaX5[i]);
  cudaFreeHost(h_prevSigmaZ[i]);
  cudaFreeHost(h_curSigmaZ[i]);
  cudaFreeHost(h_SigmaZ4[i]);
  cudaFreeHost(h_SigmaZ5[i]);
 }
 
 for(int gpu=0;gpu<NGPU;gpu++){
  cudaSetDevice(GPUs[gpu]);

  cudaFree(d_damping[gpu]);
  
  for(int i=0;i<nbuffSigma;++i){
   for(int j=0;j<4;++j){
    cudaFree(d_SigmaX[gpu][i][j]); 
    cudaFree(d_SigmaZ[gpu][i][j]); 
   }
   delete []d_SigmaX[gpu][i];
   delete []d_SigmaZ[gpu][i];
  }
  delete []d_SigmaX[gpu];
  delete []d_SigmaZ[gpu];

  for(int i=0;i<nbuffCij;++i){
   cudaFree(d_c11[gpu][i]);
   cudaFree(d_c13[gpu][i]);
   cudaFree(d_c33[gpu][i]);
  }
  delete []d_c11[gpu];
  delete []d_c13[gpu];
  delete []d_c33[gpu];
  
  if(gpu==0) cudaStreamDestroy(transfInStream[gpu]);
  cudaStreamDestroy(computeStream[gpu]);
  cudaStreamDestroy(transfOutStream[gpu]);
 
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"GPU %d dealloc error %s\n",gpu,cudaGetErrorString(e));
 }

 delete []d_SigmaX;
 delete []d_SigmaZ;
 delete []d_c11;
 delete []d_c13;
 delete []d_c33;
 delete []transfInStream;
 delete []computeStream;
 delete []transfOutStream;
 delete []damping;
 delete []d_damping;
 
 return;
}

