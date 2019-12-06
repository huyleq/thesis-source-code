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

void modelData3d_f(float *souloc,int ns,float *recloc,float *wavelet,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate){
 
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
 int samplingTimeStep=std::round(samplingRate/dt);
 int nnt=(nt-1)/samplingTimeStep+1;

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
 float *h_data;

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
 
 int **d_recIndex=new int*[NGPU]();
 float **d_data=new float*[NGPU]();
 
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
  
//  cudaError_t e=cudaGetLastError();
//  if(e!=cudaSuccess) fprintf(stderr,"GPU %d alloc error %s\n",gpu,cudaGetErrorString(e));
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

	 int *recIndexBlock=new int[nr];
     float *data=new float[nnt*nr]();
     int itdata=1,krecord=-2,gpurecord=-2,ktransf=-2;
	
	 #pragma omp parallel for num_threads(16)
	 for(int ir=0;ir<nr;ir++){
	  int ir1=ir+irbegin;
	  int ix=(recloc[4*ir1]-ox)/dx;
	  int iy=(recloc[4*ir1+1]-oy)/dy;
	  int iz=(recloc[4*ir1+2]-oz)/dz;
	  int ixy=ix+iy*nx;
	  recIndexBlock[ir]=ixy+(iz%HALF_STENCIL)*nxy;
	 }
	 
	 int souIndexX=(souloc[5*is]-ox)/dx;
	 int souIndexY=(souloc[5*is+1]-oy)/dy;
	 int souIndexZ=(souloc[5*is+2]-oz)/dz;
	 int souIndex=souIndexX+souIndexY*nx+souIndexZ*nxy;
	 int souIndexBlock=souIndexX+souIndexY*nx+(souIndexZ%HALF_STENCIL)*nxy;
	 int souBlock=souIndexZ/HALF_STENCIL;

	 cudaHostAlloc(&h_data,nr*sizeof(float),cudaHostAllocDefault);
	 
	 for(int gpu=0;gpu<NGPU;gpu++){
      cudaSetDevice(GPUs[gpu]);
	  cudaMalloc(&d_recIndex[gpu],nr*sizeof(int));
	  cudaMemcpy(d_recIndex[gpu],recIndexBlock,nr*sizeof(int),cudaMemcpyHostToDevice);
	  cudaMalloc(&d_data[gpu],nr*sizeof(float));
//      cudaError_t e=cudaGetLastError();
//      if(e!=cudaSuccess) fprintf(stderr,"shot %d GPU %d alloc error %s\n",is,gpu,cudaGetErrorString(e));
	 }
	 
	 memset(prevSigmaX,0,nxyz*sizeof(float));
	 memset(curSigmaX,0,nxyz*sizeof(float));
	 memset(prevSigmaZ,0,nxyz*sizeof(float));
	 memset(curSigmaZ,0,nxyz*sizeof(float));
	 
	 //injecting source at time 0 to wavefields at time 1
     float temp=dt2*wavelet[0];
	 curSigmaX[souIndex]=temp;
	 curSigmaZ[souIndex]=temp;
     
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

            if(ib==recBlock && it==samplingTimeStep*itdata && itdata<nnt){
             recordData<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,computeStream[gpu]>>>(d_data[gpu],d_SigmaX[gpu][i+2][ki24],d_SigmaZ[gpu][i+2][ki24],nr,d_recIndex[gpu]);
             krecord=k;
             gpurecord=gpu;
            }
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
        
//        cudaError_t e=cudaGetLastError();
//        if(e!=cudaSuccess) fprintf(stderr,"GPU %d prop error %s\n",gpu,cudaGetErrorString(e));
       }

       if(k-1==krecord && itdata<nnt){
        cudaMemcpyAsync(h_data,d_data[gpurecord],nr*sizeof(float),cudaMemcpyDeviceToHost,transfOutStream[gpurecord]);
        krecord=-2;
        gpurecord=-2;
        ktransf=k;
       }

       if(k-1==ktransf && itdata<nnt){
         #pragma omp parallel for num_threads(16)
         for(int ir=0;ir<nr;ir++) data[itdata+ir*nnt]=h_data[ir];
         itdata++;
         ktransf=-2;
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
//        cudaError_t e=cudaGetLastError();
//        if(e!=cudaSuccess) fprintf(stderr,"GPU %d synch error %s\n",gpu,cudaGetErrorString(e));
       }
       
       for(int i=0;i<threads.size();++i) threads[i].join();
       threads.erase(threads.begin(),threads.end());
     }
    
	 #pragma omp parallel for num_threads(16)
	 for(int ir=0;ir<nr;ir++){
	  int ir1=ir+irbegin;
      if(recloc[4*ir1+3]==0.f) memset(data+ir*nnt,0,nnt*sizeof(float));
     }

     write("data",data,nnt*nr,ios_base::app);
	 
	 cudaFreeHost(h_data);
	
	 delete []recIndexBlock;
     delete []data;

	 for(int gpu=0;gpu<NGPU;gpu++){
      cudaSetDevice(GPUs[gpu]);
	  cudaFree(d_recIndex[gpu]);
	  cudaFree(d_data[gpu]);
//      cudaError_t e=cudaGetLastError();
//      if(e!=cudaSuccess) fprintf(stderr,"shot %d GPU %d dealloc error %s\n",is,gpu,cudaGetErrorString(e));
	 }
 }

 int nrtotal=souloc[5*(ns-1)+3]+souloc[5*(ns-1)+4];
 to_header("data","n1",nnt,"o1",ot,"d1",samplingRate);
 to_header("data","n2",nrtotal,"o2",0.,"d2",1);

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

 delete []d_recIndex;
 delete []d_data;
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

void modelData3d_f(float *ddata,float *souloc,int ns,vector<int> &shotid,float *recloc,const float *wavelet,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,vector<int> &GPUs,int ngpugroup){
    int ngpu1=GPUs.size()/ngpugroup;
    vector<vector<int>> gpugroups;
    
    int nshot1=shotid.size()/ngpugroup;
    int r=shotid.size()%ngpugroup;
    int m=(nshot1+1)*r;
    vector<vector<int>> shotidgroups;
    
    for(int i=0;i<ngpugroup;i++){
        vector<int> gpugroup(GPUs.begin()+i*ngpu1,GPUs.begin()+(i+1)*ngpu1);
        gpugroups.push_back(gpugroup);
        
        if(i<r){
            vector<int> shotidgroup(shotid.begin()+i*(nshot1+1),shotid.begin()+(i+1)*(nshot1+1));
            shotidgroups.push_back(shotidgroup);
        }
        else{
            vector<int> shotidgroup(shotid.begin()+m+(i-r)*nshot1,shotid.begin()+m+(i-r+1)*nshot1);
            shotidgroups.push_back(shotidgroup);
        }
//        fprintf(stderr,"group %d\n",i);
//        for(vector<int>::iterator it=gpugroups[i].begin();it!=gpugroups[i].end();it++) fprintf(stderr," gpu %d\n",*it);
//        for(vector<int>::iterator it=shotidgroups[i].begin();it!=shotidgroups[i].end();it++) fprintf(stderr," shotid %d\n",*it);
    }
    
    vector<thread> threads;
    for(int i=0;i<ngpugroup;i++) threads.push_back(thread(modelDataCij3d_f,ddata,souloc,ns,std::ref(shotidgroups[i]),recloc,wavelet,c11,c13,c33,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,std::ref(gpugroups[i])));
    for(int i=ngpugroup-1;i>=0;i--) threads[i].join();   
    return;
}

void modelDataCij3d_f(float *ddata,float *souloc,int ns,vector<int> &shotid,float *recloc,const float *wavelet,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,vector<int> &GPUs){
 
// vector<int> GPUs;
// get_array("gpu",GPUs);
 int NGPU=GPUs.size();
// fprintf(stderr,"Total # GPUs = %d\n",NGPU);
// fprintf(stderr,"GPUs used are:\n");
// for(int i=0;i<NGPU;i++) fprintf(stderr,"%d ",GPUs[i]);
// fprintf(stderr,"\n");

 float dx2=dx*dx,dy2=dy*dy,dz2=dz*dz,dt2=dt*dt;
 float dt2dx2=dt2/dx2,dt2dy2=dt2/dy2,dt2dz2=dt2/dz2;
 long long nxyz=nx*ny*nz;
 int nxy=nx*ny;
 int samplingTimeStep=std::round(samplingRate/dt);
 int nnt=(nt-1)/samplingTimeStep+1;

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
 float *h_data;

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
 
 int **d_recIndex=new int*[NGPU]();
 float **d_data=new float*[NGPU]();
 
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
  if(e!=cudaSuccess) fprintf(stderr,"GPU %d alloc error %s\n",GPUs[gpu],cudaGetErrorString(e));
 }

 vector<thread> threads;
 int pipelen=NGPU*(NUPDATE+3)+3;
 int nround=(nt-2)/(NGPU*NUPDATE);
 int roundlen=max(pipelen,nb);
 int nroundlen=nround*roundlen;;
 int nk=(nround-1)*roundlen+pipelen+nb-1;
 
 int recBlock=(recloc[2]-oz)/dz/HALF_STENCIL; //assume all receivers are at same depth

 for(vector<int>::iterator id=shotid.begin();id!=shotid.end();id++){
     int is=*id;
	 fprintf(stderr,"shot # %d\n",is);

	 int nr=souloc[5*is+3];
	 int irbegin=souloc[5*is+4];

	 int *recIndexBlock=new int[nr];
     float *data=new float[nnt*nr]();
     int itdata=1,krecord=-2,gpurecord=-2,ktransf=-2;
	
	 #pragma omp parallel for num_threads(16)
	 for(int ir=0;ir<nr;ir++){
	  int ir1=ir+irbegin;
	  int ix=(recloc[4*ir1]-ox)/dx;
	  int iy=(recloc[4*ir1+1]-oy)/dy;
	  int iz=(recloc[4*ir1+2]-oz)/dz;
	  int ixy=ix+iy*nx;
	  recIndexBlock[ir]=ixy+(iz%HALF_STENCIL)*nxy;
	 }
	 
	 int souIndexX=(souloc[5*is]-ox)/dx;
	 int souIndexY=(souloc[5*is+1]-oy)/dy;
	 int souIndexZ=(souloc[5*is+2]-oz)/dz;
	 int souIndex=souIndexX+souIndexY*nx+souIndexZ*nxy;
	 int souIndexBlock=souIndexX+souIndexY*nx+(souIndexZ%HALF_STENCIL)*nxy;
	 int souBlock=souIndexZ/HALF_STENCIL;

	 cudaHostAlloc(&h_data,nr*sizeof(float),cudaHostAllocDefault);
	 
	 for(int gpu=0;gpu<NGPU;gpu++){
      cudaSetDevice(GPUs[gpu]);
	  cudaMalloc(&d_recIndex[gpu],nr*sizeof(int));
	  cudaMemcpy(d_recIndex[gpu],recIndexBlock,nr*sizeof(int),cudaMemcpyHostToDevice);
	  cudaMalloc(&d_data[gpu],nr*sizeof(float));
//      cudaError_t e=cudaGetLastError();
//      if(e!=cudaSuccess) fprintf(stderr,"shot %d GPU %d alloc error %s\n",is,gpu,cudaGetErrorString(e));
	 }
	 
	 memset(prevSigmaX,0,nxyz*sizeof(float));
	 memset(curSigmaX,0,nxyz*sizeof(float));
	 memset(prevSigmaZ,0,nxyz*sizeof(float));
	 memset(curSigmaZ,0,nxyz*sizeof(float));
	 
	 //injecting source at time 0 to wavefields at time 1
     float temp=dt2*wavelet[0];
	 curSigmaX[souIndex]=temp;
	 curSigmaZ[souIndex]=temp;
     
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

            if(ib==recBlock && it==samplingTimeStep*itdata && itdata<nnt){
             recordData<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,computeStream[gpu]>>>(d_data[gpu],d_SigmaX[gpu][i+2][ki24],d_SigmaZ[gpu][i+2][ki24],nr,d_recIndex[gpu]);
             krecord=k;
             gpurecord=gpu;
            }
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
        
//        cudaError_t e=cudaGetLastError();
//        if(e!=cudaSuccess) fprintf(stderr,"GPU %d prop error %s\n",gpu,cudaGetErrorString(e));
       }

       if(k-1==krecord && itdata<nnt){
        cudaMemcpyAsync(h_data,d_data[gpurecord],nr*sizeof(float),cudaMemcpyDeviceToHost,transfOutStream[gpurecord]);
        krecord=-2;
        gpurecord=-2;
        ktransf=k;
       }

       if(k-1==ktransf && itdata<nnt){
         #pragma omp parallel for num_threads(16)
         for(int ir=0;ir<nr;ir++) data[itdata+ir*nnt]=h_data[ir];
         itdata++;
         ktransf=-2;
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
//        cudaError_t e=cudaGetLastError();
//        if(e!=cudaSuccess) fprintf(stderr,"GPU %d synch error %s\n",GPUs[gpu],cudaGetErrorString(e));
       }
       
       for(int i=0;i<threads.size();++i) threads[i].join();
       threads.erase(threads.begin(),threads.end());
     }
	 
     #pragma omp parallel for num_threads(16)
	 for(int ir=0;ir<nr;ir++){
	  int ir1=ir+irbegin;
      if(recloc[4*ir1+3]==0.f) memset(data+ir*nnt,0,nnt*sizeof(float));
     }
    
//     write("data",data,nnt*nr,ios_base::app);
     size_t pos=(long long)nnt*(long long)irbegin;
     memcpy(ddata+pos,data,nnt*nr*sizeof(float));
     	 
	 cudaFreeHost(h_data);
	
	 delete []recIndexBlock;
     delete []data;

	 for(int gpu=0;gpu<NGPU;gpu++){
      cudaSetDevice(GPUs[gpu]);
	  cudaFree(d_recIndex[gpu]);
	  cudaFree(d_data[gpu]);
      cudaError_t e=cudaGetLastError();
      if(e!=cudaSuccess) fprintf(stderr,"shot %d GPU %d dealloc error %s\n",is,GPUs[gpu],cudaGetErrorString(e));
	 }
 }

// int nrtotal=souloc[5*(ns-1)+3]+souloc[5*(ns-1)+4];
// to_header("data","n1",nnt,"o1",ot,"d1",samplingRate);
// to_header("data","n2",nrtotal,"o2",0.,"d2",1);
//
// write("wavefield",curSigmaX,nxyz);
// to_header("wavefield","n1",nx,"o1",ox,"d1",dx);
// to_header("wavefield","n2",ny,"o2",oy,"d2",dy);
// to_header("wavefield","n3",nz,"o3",oz,"d3",dz);

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

 delete []d_recIndex;
 delete []d_data;
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

void waveletGradient3d_f(float *gwavelet,const float *ddata,float *souloc,int ns,vector<int> &shotid,float *recloc,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,vector<int> &GPUs,int ngpugroup){
    memset(gwavelet,0,nt*sizeof(float));

    //GPUs is array of all gpus
    //assume nshot >= ngpugroup
    float **gw=new float*[ngpugroup]();
    
    int ngpu1=GPUs.size()/ngpugroup;
    vector<vector<int>> gpugroups;
    
    int nshot1=shotid.size()/ngpugroup;
    int r=shotid.size()%ngpugroup;
    int m=(nshot1+1)*r;
    vector<vector<int>> shotidgroups;
    
    for(int i=0;i<ngpugroup;i++){
        gw[i]=new float[nt];
        
        vector<int> gpugroup(GPUs.begin()+i*ngpu1,GPUs.begin()+(i+1)*ngpu1);
        gpugroups.push_back(gpugroup);
        
        if(i<r){
            vector<int> shotidgroup(shotid.begin()+i*(nshot1+1),shotid.begin()+(i+1)*(nshot1+1));
            shotidgroups.push_back(shotidgroup);
        }
        else{
            vector<int> shotidgroup(shotid.begin()+m+(i-r)*nshot1,shotid.begin()+m+(i-r+1)*nshot1);
            shotidgroups.push_back(shotidgroup);
        }
//        fprintf(stderr,"group %d\n",i);
//        for(vector<int>::iterator it=gpugroups[i].begin();it!=gpugroups[i].end();it++) fprintf(stderr," gpu %d\n",*it);
//        for(vector<int>::iterator it=shotidgroups[i].begin();it!=shotidgroups[i].end();it++) fprintf(stderr," shotid %d\n",*it);
    }
    
    vector<thread> threads;
    for(int i=0;i<ngpugroup;i++) threads.push_back(thread(waveletGradientCij3d_f,gw[i],ddata,souloc,ns,std::ref(shotidgroups[i]),recloc,c11,c13,c33,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,std::ref(gpugroups[i])));

    for(int i=ngpugroup-1;i>=0;i--){
        threads[i].join();   
        #pragma omp parallel for 
        for(size_t j=0;j<nt;j++) gwavelet[j]+=gw[i][j];  
        delete[]gw[i];
    }

    delete []gw;
    return;
}

void waveletGradientCij3d_f(float *gwavelet,const float *ddata,float *souloc,int ns,vector<int> &shotid,float *recloc,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,vector<int> &GPUs){
 memset(gwavelet,0,nt*sizeof(float));

// vector<int> GPUs;
// get_array("gpu",GPUs);
 int NGPU=GPUs.size();
// fprintf(stderr,"Total # GPUs = %d\n",NGPU);
// fprintf(stderr,"GPUs used are:\n");
// for(int i=0;i<NGPU;i++) fprintf(stderr,"%d ",GPUs[i]);
// fprintf(stderr,"\n");

 float dx2=dx*dx,dy2=dy*dy,dz2=dz*dz,dt2=dt*dt;
 float dt2dx2=dt2/dx2,dt2dy2=dt2/dy2,dt2dz2=dt2/dz2;
 long long nxy=nx*ny,nxyz=nxy*nz;
 int samplingTimeStep=std::round(samplingRate/dt);
 int nnt=(nt-1)/samplingTimeStep+1;

// float *damping=new float[nxy];
// init_abc(damping,nx,ny,npad);
 float *damping=new float[nxy+nz];
 init_abc(damping,nx,ny,nz,npad);
 float **d_damping=new float*[NGPU]();

 float *prevLambdaX=new float[nxyz];
 float *curLambdaX=new float[nxyz];
 float *prevLambdaZ=new float[nxyz];
 float *curLambdaZ=new float[nxyz];

 size_t nElemBlock=HALF_STENCIL*nxy;
 size_t nByteBlock=nElemBlock*sizeof(float);
 int nb=nz/HALF_STENCIL;

 float *h_c11[2],*h_c13[2],*h_c33[2];
 float *h_prevLambdaX[2],*h_curLambdaX[2],*h_LambdaX4[2],*h_LambdaX5[2];
 float *h_prevLambdaZ[2],*h_curLambdaZ[2],*h_LambdaZ4[2],*h_LambdaZ5[2];
 float *h_data,*h_res[2];

 for(int i=0;i<2;++i){
  cudaHostAlloc(&h_c11[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_c13[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_c33[i],nByteBlock,cudaHostAllocDefault);
  
  cudaHostAlloc(&h_prevLambdaX[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_curLambdaX[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_LambdaX4[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_LambdaX5[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_prevLambdaZ[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_curLambdaZ[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_LambdaZ4[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_LambdaZ5[i],nByteBlock,cudaHostAllocDefault);
 }

 const int nbuffSigma=NUPDATE+2;
 
 float **d_gwavelet=new float*[NGPU]();
 int **d_recIndex=new int*[NGPU]();
 float **d_data=new float*[NGPU]();
 float ***d_res=new float**[NGPU]();
 
 float ****d_LambdaX=new float ***[NGPU]();
 float ****d_LambdaZ=new float ***[NGPU]();
 
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
  float mem=0.;
  cudaSetDevice(GPUs[gpu]);
 
  cudaMalloc(&d_gwavelet[gpu],nt*sizeof(float));
  cudaMemset(d_gwavelet[gpu],0,nt*sizeof(float));
  mem+=nt*sizeof(float);

//  cudaMalloc(&d_damping[gpu],nxy*sizeof(float));
//  cudaMemcpy(d_damping[gpu],damping,nxy*sizeof(float),cudaMemcpyHostToDevice);
  cudaMalloc(&d_damping[gpu],(nxy+nz)*sizeof(float));
  mem+=(nxy+nz)*sizeof(float);
  cudaMemcpy(d_damping[gpu],damping,(nxy+nz)*sizeof(float),cudaMemcpyHostToDevice);

  d_LambdaX[gpu]=new float**[nbuffSigma]();
  d_LambdaZ[gpu]=new float**[nbuffSigma]();
  for(int i=0;i<nbuffSigma;++i){
   d_LambdaX[gpu][i]=new float*[4]();
   d_LambdaZ[gpu][i]=new float*[4]();
   for(int j=0;j<4;++j){
    cudaMalloc(&d_LambdaX[gpu][i][j],nByteBlock); 
    cudaMalloc(&d_LambdaZ[gpu][i][j],nByteBlock); 
    mem+=2*nByteBlock;
   }
  }

  d_c11[gpu]=new float*[nbuffCij]();
  d_c13[gpu]=new float*[nbuffCij]();
  d_c33[gpu]=new float*[nbuffCij]();
  for(int i=0;i<nbuffCij;++i){
   cudaMalloc(&d_c11[gpu][i],nByteBlock);
   cudaMalloc(&d_c13[gpu][i],nByteBlock);
   cudaMalloc(&d_c33[gpu][i],nByteBlock);
   mem+=3*nByteBlock;
  }
 
  d_res[gpu]=new float*[2]();
 
  if(gpu==0) cudaStreamCreate(&transfInStream[gpu]);
  cudaStreamCreate(&computeStream[gpu]);
  cudaStreamCreate(&transfOutStream[gpu]);
  
  fprintf(stderr,"gpu %d allocates %f GB\n",GPUs[gpu],mem*1e-9);
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"GPU %d alloc error %s\n",gpu,cudaGetErrorString(e));
 }

 vector<thread> threads;
 int pipelen=NGPU*(NUPDATE+3)+3;
 int nround=(nt-2)/(NGPU*NUPDATE);
 int roundlen=max(pipelen,nb);
 int nroundlen=nround*roundlen;;
 int nk=(nround-1)*roundlen+pipelen+nb-1;
// fprintf(stderr,"pipelen=%d nround=%d roundlen=%d nk=%d\n",pipelen,nround,roundlen,nk);
 
 int recBlock=(recloc[2]-oz)/dz/HALF_STENCIL; //assume all receivers are at same depth
 
 for(vector<int>::iterator id=shotid.begin();id!=shotid.end();id++){
     int is=*id;
	 fprintf(stderr,"shot # %d\n",is);

	 int nr=souloc[5*is+3];
	 int irbegin=souloc[5*is+4];

	 int *recIndex=new int[nr];
	 int *recIndexBlock=new int[nr];
     float *data=new float[nnt*nr]();
	
	 #pragma omp parallel for num_threads(16)
	 for(int ir=0;ir<nr;ir++){
	  int ir1=ir+irbegin;
	  int ix=(recloc[4*ir1]-ox)/dx;
	  int iy=(recloc[4*ir1+1]-oy)/dy;
	  int iz=(recloc[4*ir1+2]-oz)/dz;
	  int ixy=ix+iy*nx;
	  recIndex[ir]=ixy+iz*nxy;
	  recIndexBlock[ir]=ixy+(iz%HALF_STENCIL)*nxy;
	 }
	 
	 int souIndexX=(souloc[5*is]-ox)/dx;
	 int souIndexY=(souloc[5*is+1]-oy)/dy;
	 int souIndexZ=(souloc[5*is+2]-oz)/dz;
	 int souIndexBlock=souIndexX+souIndexY*nx+(souIndexZ%HALF_STENCIL)*nxy;
	 int souBlock=souIndexZ/HALF_STENCIL;

	 cudaHostAlloc(&h_data,nr*sizeof(float),cudaHostAllocDefault);
	 cudaHostAlloc(&h_res[0],nr*sizeof(float),cudaHostAllocDefault);
     h_res[1]=h_data;
	 
	 for(int gpu=0;gpu<NGPU;gpu++){
      cudaSetDevice(GPUs[gpu]);
	  cudaMalloc(&d_recIndex[gpu],nr*sizeof(int));
	  cudaMemcpy(d_recIndex[gpu],recIndexBlock,nr*sizeof(int),cudaMemcpyHostToDevice);
	  cudaMalloc(&d_data[gpu],nr*sizeof(float));
	  cudaMalloc(&d_res[gpu][0],nr*sizeof(float));
      d_res[gpu][1]=d_data[gpu];
//      cudaError_t e=cudaGetLastError();
//      if(e!=cudaSuccess) fprintf(stderr,"shot %d GPU %d alloc error %s\n",is,gpu,cudaGetErrorString(e));
	 }
	
     //handle data
     size_t pos=(long long)nnt*(long long)irbegin;
     memcpy(data,ddata+pos,nnt*nr*sizeof(float));
	 
     //propagate adjoint wavefields
     memset(prevLambdaX,0,nxyz*sizeof(float));
	 memset(curLambdaX,0,nxyz*sizeof(float));
	 memset(prevLambdaZ,0,nxyz*sizeof(float));
	 memset(curLambdaZ,0,nxyz*sizeof(float));
	
//     fprintf(stderr,"inject residual to adjoint wavefields\n");
	 #pragma omp parallel for num_threads(16)
	 for(int ir=0;ir<nr;ir++){
      float temp=dt2*data[(nnt-1)+ir*nnt];
	  curLambdaX[recIndex[ir]]=TWOTHIRD*temp;
	  curLambdaZ[recIndex[ir]]=ONETHIRD*temp;
	 }
     
     for(int k=0;k<nk;k++){
       if(k<nroundlen){
        int ib=k%roundlen;
        if(ib<nb){
            size_t ibn=ib*nElemBlock; 
            int k2=k%2;
    	    threads.push_back(thread(memcpyCpuToCpu3,h_c11[k2],c11+ibn,h_c13[k2],c13+ibn,h_c33[k2],c33+ibn,nByteBlock));
    		threads.push_back(thread(memcpyCpuToCpu2,h_prevLambdaX[k2],prevLambdaX+ibn,h_curLambdaX[k2],curLambdaX+ibn,nByteBlock));
    		threads.push_back(thread(memcpyCpuToCpu2,h_prevLambdaZ[k2],prevLambdaZ+ibn,h_curLambdaZ[k2],curLambdaZ+ibn,nByteBlock));
        }
       }
       
       if(k>0 && k<=nroundlen){
        int ib=(k-1)%roundlen;
        if(ib<nb){
          int k12=(k-1)%2,kn=k%nbuffCij,k4=k%4;
          cudaSetDevice(GPUs[0]);
          memcpyCpuToGpu2(d_LambdaX[0][0][k4],h_prevLambdaX[k12],d_LambdaX[0][1][k4],h_curLambdaX[k12],nByteBlock,transfInStream);
          memcpyCpuToGpu2(d_LambdaZ[0][0][k4],h_prevLambdaZ[k12],d_LambdaZ[0][1][k4],h_curLambdaZ[k12],nByteBlock,transfInStream);
          memcpyCpuToGpu3(d_c11[0][kn],h_c11[k12],d_c13[0][kn],h_c13[k12],d_c33[0][kn],h_c33[k12],nByteBlock,transfInStream);
        }
       }
      
       for(int gpu=0;gpu<NGPU;gpu++){
           int kgpu=k+2-gpu*(NUPDATE+3);
           if(kgpu>2 && kgpu<=NUPDATE+1+nroundlen){
               for(int i=0;i<NUPDATE;i++){
                   int ib=(kgpu-3-i)%roundlen;
                   int iround=(kgpu-3-i)/roundlen;
                   if(ib==recBlock && iround>=0 && iround<nround){
                       int it=iround*NGPU*NUPDATE+gpu*NUPDATE+2+i;
                       it=nt-1-it;
                       threads.push_back(thread(interpolateResidual,h_res[k%2],data,it+1,nnt,nr,samplingTimeStep));
                   }
               }
           }
           
           kgpu=k+1-gpu*(NUPDATE+3);
           if(kgpu>2 && kgpu<=NUPDATE+1+nroundlen){
               for(int i=0;i<NUPDATE;i++){
                   int ib=(kgpu-3-i)%roundlen;
                   int iround=(kgpu-3-i)/roundlen;
                   if(ib==recBlock && iround>=0 && iround<nround){
                       cudaSetDevice(GPUs[gpu]);
                       int it=iround*NGPU*NUPDATE+gpu*NUPDATE+2+i;
                       it=nt-1-it;
                       cudaMemcpyAsync(d_res[gpu][k%2],h_res[(k-1)%2],nr*sizeof(float),cudaMemcpyHostToDevice,transfOutStream[gpu]); 
                   }
               }
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
           it=nt-1-it;
           int ki=kgpu-i,ki14=(ki-1)%4,ki24=(ki-2)%4,ki34=(ki-3)%4,ki2n=(ki-2)%nbuffCij;
    
           if(ib==0){
            forwardKernelTopBlock<<<grid,block,0,computeStream[gpu]>>>(d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaX[gpu][i][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki34],d_LambdaZ[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki14],d_LambdaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
           }
           else if(ib==nb-1){
            forwardKernelBottomBlock<<<grid,block,0,computeStream[gpu]>>>(d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaX[gpu][i][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki34],d_LambdaZ[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki14],d_LambdaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
           }
           else{
            forwardKernel<<<grid,block,0,computeStream[gpu]>>>(d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaX[gpu][i][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki34],d_LambdaZ[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki14],d_LambdaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
           }
           
           if(ib==souBlock){
            extractAdjWfldAtSouLoc<<<1,1,0,computeStream[gpu]>>>(d_gwavelet[gpu],d_LambdaX[gpu][i+2][ki24],d_LambdaZ[gpu][i+2][ki24],souIndexBlock,it);
            }

            if(ib==recBlock){
             injectResidual<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,computeStream[gpu]>>>(d_res[gpu][(k-1)%2],d_LambdaX[gpu][i+2][ki24],d_LambdaZ[gpu][i+2][ki24],nr,d_recIndex[gpu],dt2);
            }
            
            int iz=ib*HALF_STENCIL;
            if(iz<npad || iz+HALF_STENCIL-1>nz-npad) abcXYZ<<<grid,block,0,computeStream[gpu]>>>(ib,nx,ny,nz,npad,d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki24],d_damping[gpu]);
            else abcXY<<<grid,block,0,computeStream[gpu]>>>(ib,nx,ny,nz,npad,d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki24],d_damping[gpu]);
           }
          }
         }
        
        if(kgpu>NUPDATE+3 && kgpu<=NUPDATE+3+nroundlen){
         int ib=(kgpu-NUPDATE-4)%roundlen;
         if(ib<nb){
    	     if(NGPU>1 && gpu<NGPU-1){
              int n2=nbuffSigma-2,n1=nbuffSigma-1,kn3=kgpu-NUPDATE-3,kn34=kn3%4,kn3n=kn3%nbuffCij;
    	      memcpyGpuToGpu2(d_LambdaX[gpu+1][0][kn34],d_LambdaX[gpu][n2][kn34],d_LambdaX[gpu+1][1][kn34],d_LambdaX[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToGpu2(d_LambdaZ[gpu+1][0][kn34],d_LambdaZ[gpu][n2][kn34],d_LambdaZ[gpu+1][1][kn34],d_LambdaZ[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToGpu3(d_c11[gpu+1][kn3n],d_c11[gpu][kn3n],d_c13[gpu+1][kn3n],d_c13[gpu][kn3n],d_c33[gpu+1][kn3n],d_c33[gpu][kn3n],nByteBlock,transfOutStream+gpu);
    	     }
    	     else{
              int n2=nbuffSigma-2,n1=nbuffSigma-1,k2=k%2,kn3=kgpu-NUPDATE-3,kn34=kn3%4;
    	      memcpyGpuToCpu2(h_LambdaX4[k2],d_LambdaX[gpu][n2][kn34],h_LambdaX5[k2],d_LambdaX[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToCpu2(h_LambdaZ4[k2],d_LambdaZ[gpu][n2][kn34],h_LambdaZ5[k2],d_LambdaZ[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	     }
          }
        }
        
//        cudaError_t e=cudaGetLastError();
//        if(e!=cudaSuccess) fprintf(stderr,"GPU %d prop error %s\n",gpu,cudaGetErrorString(e));
       }

       if(k>pipelen-2 and k<=pipelen-2+nroundlen){
        int ib=(k-pipelen+1)%roundlen;
        if(ib<nb){
            size_t ibn=ib*nElemBlock; 
            int k12=(k-1)%2;
    	    memcpyCpuToCpu2(prevLambdaX+ibn,h_LambdaX4[k12],curLambdaX+ibn,h_LambdaX5[k12],nByteBlock);
    	    memcpyCpuToCpu2(prevLambdaZ+ibn,h_LambdaZ4[k12],curLambdaZ+ibn,h_LambdaZ5[k12],nByteBlock);
        }
       }
      
       for(int gpu=0;gpu<NGPU;gpu++){
        cudaSetDevice(GPUs[gpu]);
        cudaDeviceSynchronize();
//        cudaError_t e=cudaGetLastError();
//        if(e!=cudaSuccess) fprintf(stderr,"GPU %d synch error %s\n",gpu,cudaGetErrorString(e));
       }
       
       for(int i=0;i<threads.size();++i) threads[i].join();
       threads.erase(threads.begin(),threads.end());
     }
    
//     if(is==0){
//      write("adjwfld",curLambdaX,nxyz);
//      to_header("adjwfld","n1",nx,"o1",ox,"d1",dx);
//      to_header("adjwfld","n2",ny,"o2",oy,"d2",dy);
//      to_header("adjwfld","n3",nz,"o3",oz,"d3",dz);
//     }

	 cudaFreeHost(h_data);
	 cudaFreeHost(h_res[0]);
	
	 delete []recIndexBlock;
	 delete []recIndex;
     delete []data;

	 for(int gpu=0;gpu<NGPU;gpu++){
      cudaSetDevice(GPUs[gpu]);
	  cudaFree(d_recIndex[gpu]);
	  cudaFree(d_data[gpu]);
	  cudaFree(d_res[gpu][0]);
//      cudaError_t e=cudaGetLastError();
//      if(e!=cudaSuccess) fprintf(stderr,"shot %d GPU %d dealloc error %s\n",is,gpu,cudaGetErrorString(e));
	 }
 }

// int nrtotal=souloc[5*(ns-1)+3]+souloc[5*(ns-1)+4];
// to_header("modeleddata","n1",nnt,"o1",ot,"d1",samplingRate);
// to_header("modeleddata","n2",nrtotal,"o2",0.,"d2",1);
// to_header("residual","n1",nnt,"o1",ot,"d1",samplingRate);
// to_header("residual","n2",nrtotal,"o2",0.,"d2",1);

 delete []prevLambdaX;delete []curLambdaX;
 delete []prevLambdaZ;delete []curLambdaZ;

 for(int i=0;i<2;++i){
  cudaFreeHost(h_c11[i]);
  cudaFreeHost(h_c13[i]);
  cudaFreeHost(h_c33[i]);
  
  cudaFreeHost(h_prevLambdaX[i]);
  cudaFreeHost(h_curLambdaX[i]);
  cudaFreeHost(h_LambdaX4[i]);
  cudaFreeHost(h_LambdaX5[i]);
  cudaFreeHost(h_prevLambdaZ[i]);
  cudaFreeHost(h_curLambdaZ[i]);
  cudaFreeHost(h_LambdaZ4[i]);
  cudaFreeHost(h_LambdaZ5[i]);
 }

 float *h_gwavelet;
 cudaHostAlloc(&h_gwavelet,nt*sizeof(float),cudaHostAllocDefault);

 for(int gpu=0;gpu<NGPU;gpu++){
  cudaSetDevice(GPUs[gpu]);
  
  cudaMemcpyAsync(h_gwavelet,d_gwavelet[gpu],nt*sizeof(float),cudaMemcpyDeviceToHost,transfOutStream[gpu]);
  cudaStreamSynchronize(transfOutStream[gpu]);
  
  #pragma omp parallel for
  for(int i=0;i<nt;i++) gwavelet[i]+=h_gwavelet[i];

  cudaFree(d_gwavelet[gpu]);
  
  cudaFree(d_damping[gpu]);
  
  for(int i=0;i<nbuffSigma;++i){
   for(int j=0;j<4;++j){
    cudaFree(d_LambdaX[gpu][i][j]); 
    cudaFree(d_LambdaZ[gpu][i][j]); 
   }
   delete []d_LambdaX[gpu][i];
   delete []d_LambdaZ[gpu][i];
  }
  delete []d_LambdaX[gpu];
  delete []d_LambdaZ[gpu];

  for(int i=0;i<nbuffCij;++i){
   cudaFree(d_c11[gpu][i]);
   cudaFree(d_c13[gpu][i]);
   cudaFree(d_c33[gpu][i]);
  }
  delete []d_c11[gpu];
  delete []d_c13[gpu];
  delete []d_c33[gpu];
  
  delete []d_res[gpu];
  
  if(gpu==0) cudaStreamDestroy(transfInStream[gpu]);
  cudaStreamDestroy(computeStream[gpu]);
  cudaStreamDestroy(transfOutStream[gpu]);
 
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"GPU %d dealloc error %s\n",gpu,cudaGetErrorString(e));
 }

 cudaFreeHost(h_gwavelet);

 delete []d_recIndex;
 delete []d_data;
 delete []d_res;
 delete []d_gwavelet;
 delete []d_LambdaX;
 delete []d_LambdaZ;
 delete []d_c11;
 delete []d_c13;
 delete []d_c33;
 delete []transfInStream;
 delete []computeStream;
 delete []transfOutStream;
 delete []damping;
 delete []d_damping;

 scale(gwavelet,dt2,nt); 
 return;
}
