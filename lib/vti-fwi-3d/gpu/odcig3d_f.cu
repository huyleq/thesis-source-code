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

void odcig3d_f(float *image,float *souloc,int ns,float *recloc,float *wavelet,float *v,float *eps,float *del,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate){
 
 vector<int> GPUs;
 get_array("gpu",GPUs);
 int NGPU=GPUs.size();
 fprintf(stderr,"Total # GPUs = %d\n",NGPU);
 fprintf(stderr,"GPUs used are:\n");
 for(int i=0;i<NGPU;i++) fprintf(stderr,"%d",GPUs[i]);
 fprintf(stderr,"\n");

 float dx2=dx*dx,dy2=dy*dy,dz2=dz*dz,dt2=dt*dt;
 int nxy=nx*ny;
 long long nxyz=nx*ny*nz;
 
// float *damping=new float[nxy];
// init_abc(damping,nx,ny,npad);
 float *damping=new float[nxy+nz];
 init_abc(damping,nx,ny,nz,npad);
 float **d_damping=new float*[NGPU]();

 int samplingTimeStep=std::round(samplingRate/dt);
 int nnt=(nt-1)/samplingTimeStep+1;

 size_t nElemBlock=HALF_STENCIL*nxy;
 size_t nByteBlock=nElemBlock*sizeof(float);
 int nb=nz/HALF_STENCIL;

 float *prevSigmaX=new float[nxyz];
 float *curSigmaX=new float[nxyz];
 float *prevSigmaZ=new float[nxyz];
 float *curSigmaZ=new float[nxyz];

 float *h_v[2],*h_eps[2],*h_del[2];
 float *h_prevSigmaX[2],*h_curSigmaX[2],*h_SigmaX4[2],*h_SigmaX5[2];
 float *h_prevSigmaZ[2],*h_curSigmaZ[2],*h_SigmaZ4[2],*h_SigmaZ5[2];
 float *h_data[2];

 for(int i=0;i<2;++i){
  cudaHostAlloc(&h_v[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_eps[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_del[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_prevSigmaX[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_curSigmaX[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaX4[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaX5[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_prevSigmaZ[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_curSigmaZ[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaZ4[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaZ5[i],nByteBlock,cudaHostAllocDefault);
 }

 const int nd_Sigma=NUPDATE+2;
 int nbuffSigma[nd_Sigma];
 
 int **d_recIndex=new int*[NGPU]();
 float ***d_data=new float**[NGPU]();
 
 for(int i=0;i<nd_Sigma;++i) nbuffSigma[i]=3;
 nbuffSigma[1]=4;nbuffSigma[nd_Sigma-2]=4;

 float ****d_SigmaX=new float ***[NGPU]();
 float ****d_SigmaZ=new float ***[NGPU]();
 
 const int nbuffVEpsDel=NUPDATE+4;
 float ***d_v=new float**[NGPU]();
 float ***d_eps=new float**[NGPU]();
 float ***d_del=new float**[NGPU]();
 
 cudaStream_t *transfInStream=new cudaStream_t[NGPU]();
 cudaStream_t *transfOutStream=new cudaStream_t[NGPU]();
 cudaStream_t *computeStream=new cudaStream_t[NGPU]();
 
 for(int gpu=0;gpu<NGPU;gpu++){
  cudaSetDevice(GPUs[gpu]);
  
//  cudaMalloc(&d_damping[gpu],nxy*sizeof(float));
//  cudaMemcpy(d_damping[gpu],damping,nxy*sizeof(float),cudaMemcpyHostToDevice);
  cudaMalloc(&d_damping[gpu],(nxy+nz)*sizeof(float));
  cudaMemcpy(d_damping[gpu],damping,(nxy+nz)*sizeof(float),cudaMemcpyHostToDevice);

  d_data[gpu]=new float*[2]();
  
  d_SigmaX[gpu]=new float**[nd_Sigma]();
  d_SigmaZ[gpu]=new float**[nd_Sigma]();
  for(int i=0;i<nd_Sigma;++i){
   d_SigmaX[gpu][i]=new float*[nbuffSigma[i]]();
   d_SigmaZ[gpu][i]=new float*[nbuffSigma[i]]();
   for(int j=0;j<nbuffSigma[i];++j){
    cudaMalloc(&d_SigmaX[gpu][i][j],nByteBlock); 
    cudaMalloc(&d_SigmaZ[gpu][i][j],nByteBlock); 
   }
  }

  d_v[gpu]=new float*[nbuffVEpsDel]();
  d_eps[gpu]=new float*[nbuffVEpsDel]();
  d_del[gpu]=new float*[nbuffVEpsDel]();
  for(int i=0;i<nbuffVEpsDel;++i){
   cudaMalloc(&d_v[gpu][i],nByteBlock);
   cudaMalloc(&d_eps[gpu][i],nByteBlock);
   cudaMalloc(&d_del[gpu][i],nByteBlock);
  }

  cudaStreamCreate(&transfInStream[gpu]);
  cudaStreamCreate(&computeStream[gpu]);
  cudaStreamCreate(&transfOutStream[gpu]);
 }

 float *prevSigmaXa=new float[nxyz];
 float *curSigmaXa=new float[nxyz];
 float *prevSigmaZa=new float[nxyz];
 float *curSigmaZa=new float[nxyz];
 
 float *h_prevSigmaXa[2],*h_curSigmaXa[2],*h_SigmaXa4[2],*h_SigmaXa5[2];
 float *h_prevSigmaZa[2],*h_curSigmaZa[2],*h_SigmaZa4[2],*h_SigmaZa5[2];

 for(int i=0;i<2;++i){
  cudaHostAlloc(&h_prevSigmaXa[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_curSigmaXa[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaXa4[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaXa5[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_prevSigmaZa[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_curSigmaZa[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaZa4[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_SigmaZa5[i],nByteBlock,cudaHostAllocDefault);
 }
 
 float ****d_SigmaXa=new float ***[NGPU]();
 float ****d_SigmaZa=new float ***[NGPU]();
 
 const int ndg=NUPDATE+2;
 float ***d_image=new float**[NGPU]();

 float ***h_image=new float**[NGPU]();
 
 for(int gpu=0;gpu<NGPU;gpu++){
  cudaSetDevice(GPUs[gpu]);
  
  d_SigmaXa[gpu]=new float**[nd_Sigma]();
  d_SigmaZa[gpu]=new float**[nd_Sigma]();
  for(int i=0;i<nd_Sigma;++i){
   d_SigmaXa[gpu][i]=new float*[nbuffSigma[i]]();
   d_SigmaZa[gpu][i]=new float*[nbuffSigma[i]]();
   for(int j=0;j<nbuffSigma[i];++j){
    cudaMalloc(&d_SigmaXa[gpu][i][j],nByteBlock); 
    cudaMalloc(&d_SigmaZa[gpu][i][j],nByteBlock); 
   }
  }
 
  d_image[gpu]=new float*[ndg]();
  for(int i=0;i<ndg;i++) cudaMalloc(&d_image[gpu][i],nByteBlock*(2*NLAG+1));

  h_image[gpu]=new float*[2]();
  for(int i=0;i<2;i++) cudaHostAlloc(&h_image[gpu][i],nByteBlock*(2*NLAG+1),cudaHostAllocDefault);
 }
 
 vector<thread> threads;
 int pipelen=NGPU*(NUPDATE+3)+3;
 int nround=(nt-2)/(NGPU*NUPDATE);
 int roundlen=max(pipelen,nb);
 int nroundlen=nround*roundlen;;
 int nk=(nround-1)*roundlen+pipelen+nb-1;
 
 int recBlock=(recloc[2]-oz)/dz/HALF_STENCIL; //assume all receivers are at same depth
 
 for(int is=0;is<ns;is++){
	 fprintf(stderr,"shot # %d\n",is);

	 int nr=souloc[5*is+3];
	 int irbegin=souloc[5*is+4];

	 int *recIndex=new int[nr];
	 int *recIndexBlock=new int[nr];
     float *observedData=new float[nnt*nr];
     read("data",observedData,nnt*nr,nnt*irbegin);
	
	 #pragma omp parallel for num_threads(16)
	 for(int ir=0;ir<nr;ir++){
	  int ir1=ir+irbegin;
	  int ix=(recloc[3*ir1]-ox)/dx;
	  int iy=(recloc[3*ir1+1]-oy)/dy;
	  int iz=(recloc[3*ir1+2]-oz)/dz;
	  int ixy=ix+iy*nx;
	  recIndex[ir]=ixy+iz*nxy;
	  recIndexBlock[ir]=ixy+(iz%HALF_STENCIL)*nxy;
	 }
	 
	 int souIndexX=(souloc[5*is]-ox)/dx;
	 int souIndexY=(souloc[5*is+1]-oy)/dy;
	 int souIndexZ=(souloc[5*is+2]-oz)/dz;
	 int souIndex=souIndexX+souIndexY*nx+souIndexZ*nxy;
	 int souIndexBlock=souIndexX+souIndexY*nx+(souIndexZ%HALF_STENCIL)*nxy;
	 int souBlock=souIndexZ/HALF_STENCIL;

	 memset(prevSigmaX,0,nxyz*sizeof(float));
	 memset(curSigmaX,0,nxyz*sizeof(float));
	 memset(prevSigmaZ,0,nxyz*sizeof(float));
	 memset(curSigmaZ,0,nxyz*sizeof(float));
	
	 curSigmaX[souIndex]=dt2*wavelet[0];
	 curSigmaZ[souIndex]=dt2*wavelet[0];
	 
	 for(int i=0;i<2;i++) cudaHostAlloc(&h_data[i],nr*sizeof(float),cudaHostAllocDefault);
	 
	 for(int gpu=0;gpu<NGPU;gpu++){
      cudaSetDevice(GPUs[gpu]);
	  
	  cudaMalloc(&d_recIndex[gpu],nr*sizeof(int));
	  cudaMemcpy(d_recIndex[gpu],recIndexBlock,nr*sizeof(int),cudaMemcpyHostToDevice);
	  
	  for(int i=0;i<2;i++) cudaMalloc(&d_data[gpu][i],nr*sizeof(float));
	  
	  for(int i=0;i<nd_Sigma;++i){
	   for(int j=0;j<nbuffSigma[i];++j){
	    cudaMemset(d_SigmaX[gpu][i][j],0,nByteBlock);
	    cudaMemset(d_SigmaZ[gpu][i][j],0,nByteBlock);
	   }
	  }
	
	  for(int i=0;i<nbuffVEpsDel;++i){
	   cudaMemset(d_v[gpu][i],0,nByteBlock);
	   cudaMemset(d_eps[gpu][i],0,nByteBlock);
	   cudaMemset(d_del[gpu][i],0,nByteBlock);
	  }
	 }
	
	 fprintf(stderr,"propagate source wavefield\n");

	 for(int k=0;k<nk;k++){
	  if(k<nroundlen){
	   int ib=k%roundlen;
       if(ib<nb){
	    threads.push_back(thread(memcpyCpuToCpu3,h_v[k%2],v+ib*nElemBlock,h_eps[k%2],eps+ib*nElemBlock,h_del[k%2],del+ib*nElemBlock,nByteBlock));
		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaX[k%2],prevSigmaX+ib*nElemBlock,h_curSigmaX[k%2],curSigmaX+ib*nElemBlock,nByteBlock));
		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaZ[k%2],prevSigmaZ+ib*nElemBlock,h_curSigmaZ[k%2],curSigmaZ+ib*nElemBlock,nByteBlock));
	   }
      }
	   
	   if(k>0 && k<=nroundlen){
	    int ib=(k-1)%roundlen;
        if(ib<nb){
	    cudaSetDevice(GPUs[0]);
	    memcpyCpuToGpu3(d_v[0][k%nbuffVEpsDel],h_v[(k+1)%2],d_eps[0][k%nbuffVEpsDel],h_eps[(k+1)%2],d_del[0][k%nbuffVEpsDel],h_del[(k+1)%2],nByteBlock,transfInStream);
	    memcpyCpuToGpu2(d_SigmaX[0][0][(k-1)%nbuffSigma[0]],h_prevSigmaX[(k+1)%2],d_SigmaX[0][1][k%nbuffSigma[1]],h_curSigmaX[(k+1)%2],nByteBlock,transfInStream);
	    memcpyCpuToGpu2(d_SigmaZ[0][0][(k-1)%nbuffSigma[0]],h_prevSigmaZ[(k+1)%2],d_SigmaZ[0][1][k%nbuffSigma[1]],h_curSigmaZ[(k+1)%2],nByteBlock,transfInStream);
	   }
       }
	  
	   for(int gpu=0;gpu<NGPU;gpu++){
	    int kgpu=k-gpu*(NUPDATE+3);
	
	    cudaSetDevice(GPUs[gpu]);
	
	    if(kgpu>2 && kgpu<=NUPDATE+1+nroundlen){
	     forwardRandom(0,NUPDATE,kgpu,d_SigmaX[gpu],d_SigmaZ[gpu],nbuffSigma,d_v[gpu],d_eps[gpu],d_del[gpu],nbuffVEpsDel,wavelet,souIndexBlock,souBlock,nx,ny,nz,nt,npad,dx2,dy2,dz2,dt2,computeStream+gpu,gpu,NGPU);
		}
	    
	    if(kgpu>NUPDATE+3 && kgpu<=NUPDATE+3+nroundlen){
         int ib=(kgpu-NUPDATE-4)%roundlen;
         if(ib<nb){
	     if(NGPU>1 && gpu<NGPU-1){
	      memcpyGpuToGpu3(d_v[gpu+1][(kgpu-NUPDATE-3)%nbuffVEpsDel],d_v[gpu][(kgpu-NUPDATE-3)%nbuffVEpsDel],d_eps[gpu+1][(kgpu-NUPDATE-3)%nbuffVEpsDel],d_eps[gpu][(kgpu-NUPDATE-3)%nbuffVEpsDel],d_del[gpu+1][(kgpu-NUPDATE-3)%nbuffVEpsDel],d_del[gpu][(kgpu-NUPDATE-3)%nbuffVEpsDel],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToGpu2(d_SigmaX[gpu+1][0][(kgpu-NUPDATE-4)%nbuffSigma[0]],d_SigmaX[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],d_SigmaX[gpu+1][1][(kgpu-NUPDATE-3)%nbuffSigma[1]],d_SigmaX[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToGpu2(d_SigmaZ[gpu+1][0][(kgpu-NUPDATE-4)%nbuffSigma[0]],d_SigmaZ[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],d_SigmaZ[gpu+1][1][(kgpu-NUPDATE-3)%nbuffSigma[1]],d_SigmaZ[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	     }
	     else{
	      memcpyGpuToCpu2(h_SigmaX4[kgpu%2],d_SigmaX[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],h_SigmaX5[kgpu%2],d_SigmaX[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToCpu2(h_SigmaZ4[kgpu%2],d_SigmaZ[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],h_SigmaZ5[kgpu%2],d_SigmaZ[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	     }
	    }
	   }
       }
	   
       if(k>pipelen-2 and k<=pipelen-2+nroundlen){
        int ib=(k-pipelen+1)%roundlen;
        if(ib<nb){
            int kgpu=k-(NGPU-1)*(NUPDATE+3);
	    memcpyCpuToCpu2(prevSigmaX+ib*nElemBlock,h_SigmaX4[(kgpu+1)%2],curSigmaX+ib*nElemBlock,h_SigmaX5[(kgpu+1)%2],nByteBlock);
	    memcpyCpuToCpu2(prevSigmaZ+ib*nElemBlock,h_SigmaZ4[(kgpu+1)%2],curSigmaZ+ib*nElemBlock,h_SigmaZ5[(kgpu+1)%2],nByteBlock);
	   }
       }
	  
	   for(int gpu=0;gpu<NGPU;gpu++){
	    cudaSetDevice(GPUs[gpu]);
	    cudaDeviceSynchronize();
	   }
	   
	   for(int i=0;i<threads.size();++i) threads[i].join();
	   threads.erase(threads.begin(),threads.end());
	 }
	
	 for(int gpu=0;gpu<NGPU;gpu++){
	  cudaSetDevice(GPUs[gpu]);
	  cudaError_t e=cudaGetLastError();
	  if(e!=cudaSuccess) fprintf(stderr,"error in forward random %s gpu %d\n",cudaGetErrorString(e),gpu);
	 }
	
	 float *pt;
	 pt=curSigmaX;curSigmaX=prevSigmaX;prevSigmaX=pt;
	 pt=curSigmaZ;curSigmaZ=prevSigmaZ;prevSigmaZ=pt;
	
	 memset(prevSigmaXa,0,nxyz*sizeof(float));
	 memset(curSigmaXa,0,nxyz*sizeof(float));
	 memset(prevSigmaZa,0,nxyz*sizeof(float));
	 memset(curSigmaZa,0,nxyz*sizeof(float));
	
	 fprintf(stderr,"inject data\n");

	 #pragma omp parallel for num_threads(16)
	 for(int ir=0;ir<nr;ir++){
      float temp=dt2*observedData[(nnt-1)+ir*nnt];
	  curSigmaXa[recIndex[ir]]=TWOTHIRD*temp;
	  curSigmaZa[recIndex[ir]]=ONETHIRD*temp;
	 }
	
	 for(int k=0;k<nk;k++){
	   if(k<nroundlen){
	    int ib=k%roundlen;
        if(ib<nb){
	    threads.push_back(thread(memcpyCpuToCpu3,h_v[k%2],v+ib*nElemBlock,h_eps[k%2],eps+ib*nElemBlock,h_del[k%2],del+ib*nElemBlock,nByteBlock));
		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaX[k%2],prevSigmaX+ib*nElemBlock,h_curSigmaX[k%2],curSigmaX+ib*nElemBlock,nByteBlock));
		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaZ[k%2],prevSigmaZ+ib*nElemBlock,h_curSigmaZ[k%2],curSigmaZ+ib*nElemBlock,nByteBlock));
		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaXa[k%2],prevSigmaXa+ib*nElemBlock,h_curSigmaXa[k%2],curSigmaXa+ib*nElemBlock,nByteBlock));
		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaZa[k%2],prevSigmaZa+ib*nElemBlock,h_curSigmaZa[k%2],curSigmaZa+ib*nElemBlock,nByteBlock));
	   }
       }
	   
	   if(k>0 && k<=nroundlen){
	    int ib=(k-1)%roundlen;
        if(ib<nb){
	    cudaSetDevice(GPUs[0]);
	    memcpyCpuToGpu3(d_v[0][k%nbuffVEpsDel],h_v[(k+1)%2],d_eps[0][k%nbuffVEpsDel],h_eps[(k+1)%2],d_del[0][k%nbuffVEpsDel],h_del[(k+1)%2],nByteBlock,transfInStream);
	    memcpyCpuToGpu2(d_SigmaX[0][0][(k-1)%nbuffSigma[0]],h_prevSigmaX[(k+1)%2],d_SigmaX[0][1][k%nbuffSigma[1]],h_curSigmaX[(k+1)%2],nByteBlock,transfInStream);
	    memcpyCpuToGpu2(d_SigmaZ[0][0][(k-1)%nbuffSigma[0]],h_prevSigmaZ[(k+1)%2],d_SigmaZ[0][1][k%nbuffSigma[1]],h_curSigmaZ[(k+1)%2],nByteBlock,transfInStream);
	    memcpyCpuToGpu2(d_SigmaXa[0][0][(k-1)%nbuffSigma[0]],h_prevSigmaXa[(k+1)%2],d_SigmaXa[0][1][k%nbuffSigma[1]],h_curSigmaXa[(k+1)%2],nByteBlock,transfInStream);
	    memcpyCpuToGpu2(d_SigmaZa[0][0][(k-1)%nbuffSigma[0]],h_prevSigmaZa[(k+1)%2],d_SigmaZa[0][1][k%nbuffSigma[1]],h_curSigmaZa[(k+1)%2],nByteBlock,transfInStream);
	   }
       }
	   
	   for(int gpu=0;gpu<NGPU;gpu++){
	    int kgpu=k-gpu*(NUPDATE+3);
		int ibgpu=kgpu%nb,itgpu=kgpu/nb;
	    cudaSetDevice(GPUs[gpu]);
	
	    if(ibgpu>recBlock && ibgpu<recBlock+NUPDATE+1) threads.push_back(thread(interpolateResidual,h_data[kgpu%2],observedData,nt-(itgpu*NGPU+gpu)*NUPDATE-ibgpu-1+recBlock,nnt,nr,samplingTimeStep));
	    if(ibgpu>recBlock+1 && ibgpu<recBlock+NUPDATE+2) cudaMemcpyAsync(d_data[gpu][kgpu%2],h_data[(kgpu+1)%2],nr*sizeof(float),cudaMemcpyHostToDevice,transfInStream[gpu]);
	 
	    if(kgpu>2 && kgpu<=NUPDATE+1+nroundlen){
	     extendedImaging(d_image[gpu],ndg,0,NUPDATE,kgpu,d_SigmaX[gpu],d_SigmaZ[gpu],d_SigmaXa[gpu],d_SigmaZa[gpu],nbuffSigma,d_damping[gpu],d_v[gpu],d_eps[gpu],d_del[gpu],nbuffVEpsDel,wavelet,souIndexBlock,souBlock,d_data[gpu][(kgpu+1)%2],nr,d_recIndex[gpu],recBlock,nx,ny,nz,npad,nt,dx2,dy2,dz2,dt2,computeStream+gpu,gpu,NGPU);
	    }
	    
	    if(kgpu>NUPDATE+3 && kgpu<=NUPDATE+3+nroundlen){
         int ib=(kgpu-NUPDATE-4)%roundlen;
         if(ib<nb){
	     if(NGPU>1 && gpu<NGPU-1){
	      memcpyGpuToGpu3(d_v[gpu+1][(kgpu-NUPDATE-3)%nbuffVEpsDel],d_v[gpu][(kgpu-NUPDATE-3)%nbuffVEpsDel],d_eps[gpu+1][(kgpu-NUPDATE-3)%nbuffVEpsDel],d_eps[gpu][(kgpu-NUPDATE-3)%nbuffVEpsDel],d_del[gpu+1][(kgpu-NUPDATE-3)%nbuffVEpsDel],d_del[gpu][(kgpu-NUPDATE-3)%nbuffVEpsDel],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToGpu2(d_SigmaX[gpu+1][0][(kgpu-NUPDATE-4)%nbuffSigma[0]],d_SigmaX[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],d_SigmaX[gpu+1][1][(kgpu-NUPDATE-3)%nbuffSigma[1]],d_SigmaX[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToGpu2(d_SigmaZ[gpu+1][0][(kgpu-NUPDATE-4)%nbuffSigma[0]],d_SigmaZ[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],d_SigmaZ[gpu+1][1][(kgpu-NUPDATE-3)%nbuffSigma[1]],d_SigmaZ[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToGpu2(d_SigmaXa[gpu+1][0][(kgpu-NUPDATE-4)%nbuffSigma[0]],d_SigmaXa[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],d_SigmaXa[gpu+1][1][(kgpu-NUPDATE-3)%nbuffSigma[1]],d_SigmaXa[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToGpu2(d_SigmaZa[gpu+1][0][(kgpu-NUPDATE-4)%nbuffSigma[0]],d_SigmaZa[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],d_SigmaZa[gpu+1][1][(kgpu-NUPDATE-3)%nbuffSigma[1]],d_SigmaZa[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	     }
	     else{
	      memcpyGpuToCpu2(h_SigmaX4[kgpu%2],d_SigmaX[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],h_SigmaX5[kgpu%2],d_SigmaX[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToCpu2(h_SigmaZ4[kgpu%2],d_SigmaZ[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],h_SigmaZ5[kgpu%2],d_SigmaZ[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToCpu2(h_SigmaXa4[kgpu%2],d_SigmaXa[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],h_SigmaXa5[kgpu%2],d_SigmaXa[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	      memcpyGpuToCpu2(h_SigmaZa4[kgpu%2],d_SigmaZa[gpu][nd_Sigma-2][(kgpu-4)%nbuffSigma[nd_Sigma-2]],h_SigmaZa5[kgpu%2],d_SigmaZa[gpu][nd_Sigma-1][(kgpu-3)%nbuffSigma[nd_Sigma-1]],nByteBlock,transfOutStream+gpu);
	     }
	    
	    cudaMemcpyAsync(h_image[gpu][kgpu%2],d_image[gpu][(kgpu-NUPDATE-1)%ndg],nByteBlock*(2*NLAG+1),cudaMemcpyDeviceToHost,transfOutStream[gpu]);
		cudaMemsetAsync(d_image[gpu][(kgpu-NUPDATE-1)%ndg],0,nByteBlock*(2*NLAG+1),transfOutStream[gpu]);
	   }
	  }
      }
	  
       if(k>pipelen-2 and k<=pipelen-2+nroundlen){
        int ib=(k-pipelen+1)%roundlen;
        if(ib<nb){
            int kgpu=k-(NGPU-1)*(NUPDATE+3);
	    threads.push_back(thread(memcpyCpuToCpu2,prevSigmaX+ib*nElemBlock,h_SigmaX4[(kgpu+1)%2],curSigmaX+ib*nElemBlock,h_SigmaX5[(kgpu+1)%2],nByteBlock));
	    threads.push_back(thread(memcpyCpuToCpu2,prevSigmaZ+ib*nElemBlock,h_SigmaZ4[(kgpu+1)%2],curSigmaZ+ib*nElemBlock,h_SigmaZ5[(kgpu+1)%2],nByteBlock));
	    threads.push_back(thread(memcpyCpuToCpu2,prevSigmaXa+ib*nElemBlock,h_SigmaXa4[(kgpu+1)%2],curSigmaXa+ib*nElemBlock,h_SigmaXa5[(kgpu+1)%2],nByteBlock));
	    threads.push_back(thread(memcpyCpuToCpu2,prevSigmaZa+ib*nElemBlock,h_SigmaZa4[(kgpu+1)%2],curSigmaZa+ib*nElemBlock,h_SigmaZa5[(kgpu+1)%2],nByteBlock));
	   }
       }
	  
	   for(int gpu=0;gpu<NGPU;gpu++){
	    int kgpu=k-gpu*(NUPDATE+3);
	    if(kgpu>NUPDATE+4 && kgpu<=NUPDATE+4+nroundlen){
         int ib=(kgpu-NUPDATE-5)%roundlen;
	    if(ib<nb){ 
            for(int lag=0;lag<(2*NLAG+1);lag++){
             sumImageTime(image+ib*nElemBlock+lag*nxyz,h_image[gpu][(kgpu+1)%2]+lag*nElemBlock,nElemBlock);
	        }
        }
       }
       }
	   
	   for(int gpu=0;gpu<NGPU;gpu++){
	    cudaSetDevice(GPUs[gpu]);
	    cudaDeviceSynchronize();
	   }
	   
	   for(int i=0;i<threads.size();++i) threads[i].join();
	   threads.erase(threads.begin(),threads.end());
	 }
	 
	 for(int i=0;i<2;i++) cudaFreeHost(h_data[i]);
	
	 delete []recIndexBlock;
     delete []observedData;

	 for(int gpu=0;gpu<NGPU;gpu++){
      cudaSetDevice(GPUs[gpu]);
	  cudaFree(d_recIndex[gpu]);
	  for(int i=0;i<2;i++) cudaFree(d_data[gpu][i]);
	  cudaError_t e=cudaGetLastError();
	  if(e!=cudaSuccess) fprintf(stderr,"error in adjoint and imaging %s gpu %d\n",cudaGetErrorString(e),gpu);
	 }
 }

 //delete arrays used in forward
 delete []prevSigmaX;delete []curSigmaX;
 delete []prevSigmaZ;delete []curSigmaZ;

 for(int i=0;i<2;++i){
  cudaFreeHost(h_v[i]);
  cudaFreeHost(h_eps[i]);
  cudaFreeHost(h_del[i]);
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
  delete []d_data[gpu];
  
  for(int i=0;i<nd_Sigma;++i){
   for(int j=0;j<nbuffSigma[i];++j){
    cudaFree(d_SigmaX[gpu][i][j]); 
    cudaFree(d_SigmaZ[gpu][i][j]); 
   }
   delete []d_SigmaX[gpu][i];
   delete []d_SigmaZ[gpu][i];
  }
  delete []d_SigmaX[gpu];
  delete []d_SigmaZ[gpu];

  for(int i=0;i<nbuffVEpsDel;++i){
   cudaFree(d_v[gpu][i]);
   cudaFree(d_eps[gpu][i]);
   cudaFree(d_del[gpu][i]);
  }
  delete []d_v[gpu];
  delete []d_eps[gpu];
  delete []d_del[gpu];

  cudaStreamDestroy(transfInStream[gpu]);
  cudaStreamDestroy(computeStream[gpu]);
  cudaStreamDestroy(transfOutStream[gpu]);
 }

 delete []d_recIndex;
 delete []d_data;
 delete []d_SigmaX;
 delete []d_SigmaZ;
 delete []d_v;
 delete []d_eps;
 delete []d_del;
 delete []transfInStream;
 delete []computeStream;
 delete []transfOutStream;
 
 delete []damping;
 delete []d_damping;
 
 //delete arrays used in adjoint
 delete []prevSigmaXa;delete []curSigmaXa;
 delete []prevSigmaZa;delete []curSigmaZa;

 for(int i=0;i<2;++i){
  cudaFreeHost(h_prevSigmaXa[i]);
  cudaFreeHost(h_curSigmaXa[i]);
  cudaFreeHost(h_SigmaXa4[i]);
  cudaFreeHost(h_SigmaXa5[i]);
  cudaFreeHost(h_prevSigmaZa[i]);
  cudaFreeHost(h_curSigmaZa[i]);
  cudaFreeHost(h_SigmaZa4[i]);
  cudaFreeHost(h_SigmaZa5[i]);
 }
 
 for(int gpu=0;gpu<NGPU;gpu++){
  cudaSetDevice(GPUs[gpu]);
  
  for(int i=0;i<nd_Sigma;++i){
   for(int j=0;j<nbuffSigma[i];++j){
    cudaFree(d_SigmaXa[gpu][i][j]); 
    cudaFree(d_SigmaZa[gpu][i][j]); 
   }
   delete []d_SigmaXa[gpu][i];
   delete []d_SigmaZa[gpu][i];
  }
  delete []d_SigmaXa[gpu];
  delete []d_SigmaZa[gpu];
 
  for(int i=0;i<ndg;i++) cudaFree(d_image[gpu][i]);
  delete []d_image[gpu];
  
  for(int i=0;i<2;i++) cudaFreeHost(h_image[gpu][i]);
  delete []h_image[gpu];
  
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"gpu %d error %s\n",gpu,cudaGetErrorString(e));
 }
 
 delete []d_SigmaXa;
 delete []d_SigmaZa;
 
 delete []d_image;
 delete []h_image;

 return;
}
