#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"
#include "boundary.h"
#include "check.h"
#include "conversions.h"
#include "cluster.h"

using namespace std;

void rtm3d(float *image,int nx,int ny,int nz,int npad,float oz,float dz,float wbottom,vector<int> &shotid,int max_shot_per_job,int icall,const string &command){
    random_shuffle(shotid.begin(),shotid.end());    

    long long nxy=nx*ny,nxyz=nxy*nz;

    vector<Job> jobs;
    vector<int> nfail;

    int nshotleft=shotid.size(),njob=0,njobtotal=(shotid.size()+max_shot_per_job-1)/max_shot_per_job;
    while(nshotleft>0){
        int nshot1job=min(nshotleft,max_shot_per_job);
        string shotname="icall_"+to_string(icall)+"_shot_";
        string shotlist;
        for(int i=0;i<nshot1job;i++){
            int id=i+njob*max_shot_per_job;
            if(i<nshot1job-1){
                shotname+=to_string(shotid[id])+"_";
                shotlist+=to_string(shotid[id])+",";
            }
            else{
                shotname+=to_string(shotid[id]);
                shotlist+=to_string(shotid[id]);
            }
        }
        string scriptfile="./scripts/submit_"+shotname+".sh";
        string jobname=shotname;
        string outfile="./output/"+shotname+".log";
        string gradfile="./grads/image_"+shotname+".H";
        string command1=command+" image="+gradfile+" shotid="+shotlist;
        genScript(scriptfile,jobname,outfile,command1);
        string id=submitScript(scriptfile);
//        string id=to_string(njob);
        string state;
        int nerror=0;
        while(id.compare("error")==0 && nerror<MAX_FAIL){
            this_thread::sleep_for(chrono::seconds(5));
            id=submitPBSScript(scriptfile);
            nerror++;
        }
        if(id.compare("error")!=0){
            state="SUBMITTED";
            int idx=njob;
            Job job(idx,id,scriptfile,outfile,gradfile,state);
            jobs.push_back(job);
            nfail.push_back(0);
        }
        else fprintf(stderr,"job %s reaches MAX_FAIL %d\n",jobname.c_str(),MAX_FAIL);
        njob++;
        nshotleft-=nshot1job;
    }
    njob=jobs.size();
    
    this_thread::sleep_for(chrono::seconds(15));

    cout<<"submitted "<<njob<<" jobs"<<endl;
//    for(int i=0;i<jobs.size();i++) jobs[i].printJob();

    int ncompleted=0;
    
    float *image0=new float[nxyz]();

    while(ncompleted<njob){
        for(int i=0;i<jobs.size();i++){
            string id=jobs[i]._jobId;
            int idx=jobs[i]._jobIdx;
            string jobstate=jobs[i]._jobState;
            if(jobstate.compare("COMPLETED")!=0 && jobstate.compare("FAILED")!=0 && jobstate.compare("TIMEOUT")!=0){
                string state=getJobState(id);
//                string state="COMPLETED";
                if(state.compare("COMPLETED")==0){
                    cout<<"job "<<idx<<" id "<<id<<" state "<<state<<endl;
                    ncompleted++; 
                    readFromHeader(jobs[i]._gradFile,image,nxyz);
                    cout<<"summing image from file "<<jobs[i]._gradFile<<endl;
                    #pragma omp parallel for
                    for(int i=0;i<nxyz;i++) image0[i]+=image[i];
                }
                else if(state.compare("FAILED")==0 || state.compare("TIMEOUT")==0){
                    cout<<"job "<<idx<<" id "<<id<<" state "<<state<<endl;
                    nfail[idx]++;
                    if(nfail[idx]>MAX_FAIL){
                        cout<<"job "<<idx<<" reached MAX_FAIL "<<MAX_FAIL<<endl;
                        ncompleted++;
                        continue;
                    }
                    cout<<" resubmitting"<<endl;
                    Job newjob=jobs[i];
                    string newid=submitScript(newjob._scriptFile);
//                    string newid=to_string(idx);
                    if(newid.compare("error")!=0) newjob.setJobState("SUBMITTED");
                    newjob.setJobId(newid);
                    jobs.push_back(newjob);
                }
                jobs[i].setJobState(state);
            }
        }
    }

    zeroBoundary(image0,nx,ny,nz,npad);
    int nwbottom=(wbottom-oz)/dz+1-npad;
    memset(image0+npad*nxy,0,nwbottom*nxy*sizeof(float));
    
    #pragma omp parallel for num_threads(16)
    for(int iz=1;iz<nz-1;iz++){
        for(int iy=1;iy<ny-1;iy++){
            #pragma omp simd
            for(int ix=1;ix<nx-1;ix++){
                size_t i=ix+iy*nx+iz*nxy;
                image[i]=image0[i+1]+image0[i-1]+image0[i+nx]+image0[i-nx]+image0[i+nxy]+image0[i-nxy]-6.f*image0[i];
            }
        }
    }
    
    delete []image0;
    
//    for(int i=0;i<jobs.size();i++) jobs[i].printJob();
    return;
}

void rtmVEpsDel(float *image,float *souloc,int ns,vector<int> &shotid,float *recloc,float *wavelet,float *v,float *eps,float *del,float *randboundaryV,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,float wbottom){
 
 long long nxy=nx*ny;
 long long nxyz=nxy*nz;
 
 float *c11=new float[nxyz];
 float *c13=new float[nxyz];
 float *c33=new float[nxyz];
 
// putBoundary(randboundaryV,v,nx,ny,nz,npad);
 VEpsDel2Cij(c11,c13,c33,v,eps,del,1.,1.,1.,nxyz);
 
 rtmCij3d_f(image,souloc,ns,shotid,recloc,wavelet,c11,c13,c33,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate);
 
 zeroBoundary(image,nx,ny,nz,npad);
 int nwbottom=(wbottom-oz)/dz+1-npad;
 memset(image+npad*nxy,0,nwbottom*nxy*sizeof(float));
 
 delete []c11;delete []c13;delete []c33;
 return;
}
 
void rtmCij3d_f(float *image,float *souloc,int ns,vector<int> &shotid,float *recloc,float *wavelet,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate){
 
 vector<int> GPUs;
 get_array("gpu",GPUs);
 int NGPU=GPUs.size();
// fprintf(stderr,"Total # GPUs = %d\n",NGPU);
// fprintf(stderr,"GPUs used are:\n");
// for(int i=0;i<NGPU;i++) fprintf(stderr,"%d ",GPUs[i]);
// fprintf(stderr,"\n");

 float dx2=dx*dx,dy2=dy*dy,dz2=dz*dz,dt2=dt*dt;
 float dt2dx2=dt2/dx2,dt2dy2=dt2/dy2,dt2dz2=dt2/dz2;
 int nxy=nx*ny;
 long long nxyz=nxy*nz;
 int samplingTimeStep=std::round(samplingRate/dt);
 int nnt=(nt-1)/samplingTimeStep+1;

 memset(image,0,nxyz*sizeof(float));

 //this is to store multiple images
// float *image1=new float[nxyz]();
// float *image2=new float[nxyz]();
// float *image3=new float[nxyz]();

// float *damping=new float[nxy];
// init_abc(damping,nx,ny,npad);
 float *damping=new float[nxy+nz];
 init_abc(damping,nx,ny,nz,npad);
 float **d_damping=new float*[NGPU]();

 float *prevSigmaX=new float[nxyz];
 float *curSigmaX=new float[nxyz];
 float *prevSigmaZ=new float[nxyz];
 float *curSigmaZ=new float[nxyz];

 float *prevLambdaX=new float[nxyz];
 float *curLambdaX=new float[nxyz];
 float *prevLambdaZ=new float[nxyz];
 float *curLambdaZ=new float[nxyz];

 size_t nElemBlock=HALF_STENCIL*nxy;
 size_t nByteBlock=nElemBlock*sizeof(float);
 int nb=nz/HALF_STENCIL;

 float *h_c11[2],*h_c13[2],*h_c33[2];
 float *h_prevSigmaX[2],*h_curSigmaX[2],*h_SigmaX4[2],*h_SigmaX5[2];
 float *h_prevSigmaZ[2],*h_curSigmaZ[2],*h_SigmaZ4[2],*h_SigmaZ5[2];
 float *h_prevLambdaX[2],*h_curLambdaX[2],*h_LambdaX4[2],*h_LambdaX5[2];
 float *h_prevLambdaZ[2],*h_curLambdaZ[2],*h_LambdaZ4[2],*h_LambdaZ5[2];
 float *h_data[2];
 float *h_imagei[2],*h_imageo[2];

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
  
  cudaHostAlloc(&h_prevLambdaX[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_curLambdaX[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_LambdaX4[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_LambdaX5[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_prevLambdaZ[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_curLambdaZ[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_LambdaZ4[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_LambdaZ5[i],nByteBlock,cudaHostAllocDefault);
  
  cudaHostAlloc(&h_imagei[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_imageo[i],nByteBlock,cudaHostAllocDefault);
 }

 const int nbuffSigma=NUPDATE+2;
 
 int **d_recIndex=new int*[NGPU]();
 float ***d_data=new float**[NGPU]();
 
 float ****d_SigmaX=new float ***[NGPU]();
 float ****d_SigmaZ=new float ***[NGPU]();
 float ****d_LambdaX=new float ***[NGPU]();
 float ****d_LambdaZ=new float ***[NGPU]();
 
 const int nbuffCij=NUPDATE+4;
 float ***d_c11=new float**[NGPU]();
 float ***d_c13=new float**[NGPU]();
 float ***d_c33=new float**[NGPU]();
 float ***d_image=new float**[NGPU]();
 
 cudaStream_t *transfInStream=new cudaStream_t[1]();
 cudaStream_t *transfOutStream=new cudaStream_t[NGPU]();
 cudaStream_t *computeStream=new cudaStream_t[NGPU]();
 
 dim3 block(BLOCK_DIM,BLOCK_DIM);
 dim3 grid((nx-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM,(ny-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM);
 
 for(int gpu=0;gpu<NGPU;gpu++){
  float mem=0.;
  cudaSetDevice(GPUs[gpu]);
  
//  cudaMalloc(&d_damping[gpu],nxy*sizeof(float));
//  cudaMemcpy(d_damping[gpu],damping,nxy*sizeof(float),cudaMemcpyHostToDevice);
  cudaMalloc(&d_damping[gpu],(nxy+nz)*sizeof(float));
  mem+=(nxy+nz)*sizeof(float);
  cudaMemcpy(d_damping[gpu],damping,(nxy+nz)*sizeof(float),cudaMemcpyHostToDevice);

  d_SigmaX[gpu]=new float**[nbuffSigma]();
  d_SigmaZ[gpu]=new float**[nbuffSigma]();
  for(int i=0;i<nbuffSigma;++i){
   d_SigmaX[gpu][i]=new float*[4]();
   d_SigmaZ[gpu][i]=new float*[4]();
   for(int j=0;j<4;++j){
    cudaMalloc(&d_SigmaX[gpu][i][j],nByteBlock); 
    cudaMalloc(&d_SigmaZ[gpu][i][j],nByteBlock); 
    mem+=2*nByteBlock;
   }
  }

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
 
  d_image[gpu]=new float*[nbuffCij]();
  for(int i=0;i<nbuffCij;++i){
   cudaMalloc(&d_image[gpu][i],nByteBlock);
   mem+=nByteBlock;
  }

  d_data[gpu]=new float*[2]();
 
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

	 cudaHostAlloc(&h_data[0],nr*sizeof(float),cudaHostAllocDefault);
	 cudaHostAlloc(&h_data[1],nr*sizeof(float),cudaHostAllocDefault);
	 
	 for(int gpu=0;gpu<NGPU;gpu++){
      cudaSetDevice(GPUs[gpu]);
	  cudaMalloc(&d_recIndex[gpu],nr*sizeof(int));
	  cudaMemcpy(d_recIndex[gpu],recIndexBlock,nr*sizeof(int),cudaMemcpyHostToDevice);
	  cudaMalloc(&d_data[gpu][0],nr*sizeof(float));
	  cudaMalloc(&d_data[gpu][1],nr*sizeof(float));
//      cudaError_t e=cudaGetLastError();
//      if(e!=cudaSuccess) fprintf(stderr,"shot %d GPU %d alloc error %s\n",is,gpu,cudaGetErrorString(e));
	 }
	 
//     fprintf(stderr,"fowrard propagation with random boundary\n");

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
    
//     if(is==0){
//      write("randforwardwfld",curSigmaX,nxyz);
//      to_header("randforwardwfld","n1",nx,"o1",ox,"d1",dx);
//      to_header("randforwardwfld","n2",ny,"o2",oy,"d2",dy);
//      to_header("randforwardwfld","n3",nz,"o3",oz,"d3",dz);
//     }

//     fprintf(stderr,"backward propagations\n");
     //flip forward wavefields
	 float *pt;
	 pt=curSigmaX;curSigmaX=prevSigmaX;prevSigmaX=pt;
	 pt=curSigmaZ;curSigmaZ=prevSigmaZ;prevSigmaZ=pt;

     fprintf(stderr,"forward wavefield min %.10f max %.10f\n",min(curSigmaX,nxyz),max(curSigmaX,nxyz));
	 
	 memset(prevLambdaX,0,nxyz*sizeof(float));
	 memset(curLambdaX,0,nxyz*sizeof(float));
	 memset(prevLambdaZ,0,nxyz*sizeof(float));
	 memset(curLambdaZ,0,nxyz*sizeof(float));
	
     read("data",data,nnt*nr,(long long)nnt*(long long)irbegin);
     fprintf(stderr,"data min %.10f max %.10f\n",min(data,nnt*nr),max(data,nnt*nr));
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
    		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaX[k2],prevSigmaX+ibn,h_curSigmaX[k2],curSigmaX+ibn,nByteBlock));
    		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaZ[k2],prevSigmaZ+ibn,h_curSigmaZ[k2],curSigmaZ+ibn,nByteBlock));
    		threads.push_back(thread(memcpyCpuToCpu2,h_prevLambdaX[k2],prevLambdaX+ibn,h_curLambdaX[k2],curLambdaX+ibn,nByteBlock));
    		threads.push_back(thread(memcpyCpuToCpu2,h_prevLambdaZ[k2],prevLambdaZ+ibn,h_curLambdaZ[k2],curLambdaZ+ibn,nByteBlock));
    	    threads.push_back(thread(memcpy,h_imagei[k2],image+ibn,nByteBlock));
        }
       }
       
       if(k>0 && k<=nroundlen){
        int ib=(k-1)%roundlen;
        if(ib<nb){
          int k12=(k-1)%2,kn=k%nbuffCij,k4=k%4;
          cudaSetDevice(GPUs[0]);
          memcpyCpuToGpu2(d_SigmaX[0][0][k4],h_prevSigmaX[k12],d_SigmaX[0][1][k4],h_curSigmaX[k12],nByteBlock,transfInStream);
          memcpyCpuToGpu2(d_SigmaZ[0][0][k4],h_prevSigmaZ[k12],d_SigmaZ[0][1][k4],h_curSigmaZ[k12],nByteBlock,transfInStream);
          memcpyCpuToGpu2(d_LambdaX[0][0][k4],h_prevLambdaX[k12],d_LambdaX[0][1][k4],h_curLambdaX[k12],nByteBlock,transfInStream);
          memcpyCpuToGpu2(d_LambdaZ[0][0][k4],h_prevLambdaZ[k12],d_LambdaZ[0][1][k4],h_curLambdaZ[k12],nByteBlock,transfInStream);
          memcpyCpuToGpu3(d_c11[0][kn],h_c11[k12],d_c13[0][kn],h_c13[k12],d_c33[0][kn],h_c33[k12],nByteBlock,transfInStream);
          cudaMemcpyAsync(d_image[0][kn],h_imagei[k12],nByteBlock,cudaMemcpyHostToDevice,*transfInStream);
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
                       threads.push_back(thread(interpolateResidual,h_data[k%2],data,it+1,nnt,nr,samplingTimeStep));
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
                       cudaMemcpyAsync(d_data[gpu][k%2],h_data[(k-1)%2],nr*sizeof(float),cudaMemcpyHostToDevice,transfOutStream[gpu]); 
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
            forwardKernelTopBlock<<<grid,block,0,computeStream[gpu]>>>(d_SigmaX[gpu][i+2][ki24],d_SigmaX[gpu][i+1][ki24],d_SigmaX[gpu][i][ki24],d_SigmaZ[gpu][i+2][ki24],d_SigmaZ[gpu][i+1][ki34],d_SigmaZ[gpu][i+1][ki24],d_SigmaZ[gpu][i+1][ki14],d_SigmaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
            forwardKernelTopBlock<<<grid,block,0,computeStream[gpu]>>>(d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaX[gpu][i][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki34],d_LambdaZ[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki14],d_LambdaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
           }
           else if(ib==nb-1){
            forwardKernelBottomBlock<<<grid,block,0,computeStream[gpu]>>>(d_SigmaX[gpu][i+2][ki24],d_SigmaX[gpu][i+1][ki24],d_SigmaX[gpu][i][ki24],d_SigmaZ[gpu][i+2][ki24],d_SigmaZ[gpu][i+1][ki34],d_SigmaZ[gpu][i+1][ki24],d_SigmaZ[gpu][i+1][ki14],d_SigmaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
            forwardKernelBottomBlock<<<grid,block,0,computeStream[gpu]>>>(d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaX[gpu][i][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki34],d_LambdaZ[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki14],d_LambdaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
           }
           else{
            forwardKernel<<<grid,block,0,computeStream[gpu]>>>(d_SigmaX[gpu][i+2][ki24],d_SigmaX[gpu][i+1][ki24],d_SigmaX[gpu][i][ki24],d_SigmaZ[gpu][i+2][ki24],d_SigmaZ[gpu][i+1][ki34],d_SigmaZ[gpu][i+1][ki24],d_SigmaZ[gpu][i+1][ki14],d_SigmaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
            forwardKernel<<<grid,block,0,computeStream[gpu]>>>(d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaX[gpu][i][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki34],d_LambdaZ[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki14],d_LambdaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
           }
           
           if(ib==souBlock){
            float source=dt2*wavelet[it+1];
            injectSource<<<1,1,0,computeStream[gpu]>>>(d_SigmaX[gpu][i+2][ki24],d_SigmaZ[gpu][i+2][ki24],source,souIndexBlock);
            }

            if(ib==recBlock){
             injectResidual<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,computeStream[gpu]>>>(d_data[gpu][(k-1)%2],d_LambdaX[gpu][i+2][ki24],d_LambdaZ[gpu][i+2][ki24],nr,d_recIndex[gpu],dt2);
            }
            
            int iz=ib*HALF_STENCIL;
            if(iz<npad || iz+HALF_STENCIL-1>nz-npad) abcXYZ<<<grid,block,0,computeStream[gpu]>>>(ib,nx,ny,nz,npad,d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki24],d_damping[gpu]);
            else abcXY<<<grid,block,0,computeStream[gpu]>>>(ib,nx,ny,nz,npad,d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki24],d_damping[gpu]);
            
            imagingKernel<<<grid,block,0,computeStream[gpu]>>>(d_image[gpu][ki2n],d_SigmaX[gpu][i+1][ki24],d_SigmaZ[gpu][i+1][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki24],nx,ny);
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
    	      memcpyGpuToGpu2(d_LambdaX[gpu+1][0][kn34],d_LambdaX[gpu][n2][kn34],d_LambdaX[gpu+1][1][kn34],d_LambdaX[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToGpu2(d_LambdaZ[gpu+1][0][kn34],d_LambdaZ[gpu][n2][kn34],d_LambdaZ[gpu+1][1][kn34],d_LambdaZ[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToGpu3(d_c11[gpu+1][kn3n],d_c11[gpu][kn3n],d_c13[gpu+1][kn3n],d_c13[gpu][kn3n],d_c33[gpu+1][kn3n],d_c33[gpu][kn3n],nByteBlock,transfOutStream+gpu);
    	      cudaMemcpyAsync(d_image[gpu+1][kn3n],d_image[gpu][kn3n],nByteBlock,cudaMemcpyDefault,transfOutStream[gpu]);
    	     }
    	     else{
              int n2=nbuffSigma-2,n1=nbuffSigma-1,k2=k%2,kn3=kgpu-NUPDATE-3,kn34=kn3%4,kn3n=kn3%nbuffCij;
    	      memcpyGpuToCpu2(h_SigmaX4[k2],d_SigmaX[gpu][n2][kn34],h_SigmaX5[k2],d_SigmaX[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToCpu2(h_SigmaZ4[k2],d_SigmaZ[gpu][n2][kn34],h_SigmaZ5[k2],d_SigmaZ[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToCpu2(h_LambdaX4[k2],d_LambdaX[gpu][n2][kn34],h_LambdaX5[k2],d_LambdaX[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToCpu2(h_LambdaZ4[k2],d_LambdaZ[gpu][n2][kn34],h_LambdaZ5[k2],d_LambdaZ[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      cudaMemcpyAsync(h_imageo[k2],d_image[gpu][kn3n],nByteBlock,cudaMemcpyDeviceToHost,transfOutStream[gpu]);
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
    	    memcpyCpuToCpu2(prevSigmaX+ibn,h_SigmaX4[k12],curSigmaX+ibn,h_SigmaX5[k12],nByteBlock);
    	    memcpyCpuToCpu2(prevSigmaZ+ibn,h_SigmaZ4[k12],curSigmaZ+ibn,h_SigmaZ5[k12],nByteBlock);
    	    memcpyCpuToCpu2(prevLambdaX+ibn,h_LambdaX4[k12],curLambdaX+ibn,h_LambdaX5[k12],nByteBlock);
    	    memcpyCpuToCpu2(prevLambdaZ+ibn,h_LambdaZ4[k12],curLambdaZ+ibn,h_LambdaZ5[k12],nByteBlock);
    	    memcpy(image+ibn,h_imageo[k12],nByteBlock);
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
    
     fprintf(stderr,"adj wavefield min %.10f max %.10f\n",min(curLambdaX,nxyz),max(curLambdaX,nxyz));
//     if(is==0){
//      write("adjwfld",curLambdaX,nxyz);
//      to_header("adjwfld","n1",nx,"o1",ox,"d1",dx);
//      to_header("adjwfld","n2",ny,"o2",oy,"d2",dy);
//      to_header("adjwfld","n3",nz,"o3",oz,"d3",dz);
//     }

	 cudaFreeHost(h_data[0]);
	 cudaFreeHost(h_data[1]);
	
	 delete []recIndexBlock;
	 delete []recIndex;
     delete []data;

	 for(int gpu=0;gpu<NGPU;gpu++){
      cudaSetDevice(GPUs[gpu]);
	  cudaFree(d_recIndex[gpu]);
	  cudaFree(d_data[gpu][0]);
	  cudaFree(d_data[gpu][1]);
//      cudaError_t e=cudaGetLastError();
//      if(e!=cudaSuccess) fprintf(stderr,"shot %d GPU %d dealloc error %s\n",is,gpu,cudaGetErrorString(e));
	 }
     
//     #pragma omp parallel for num_threads(16)
//     for(size_t i=0;i<nxyz;i++) image1[i]+=image[i];
//
//     if(is%2==0){
//      #pragma omp parallel for num_threads(16)
//      for(size_t i=0;i<nxyz;i++) image2[i]+=image[i];
//     }
//     
//     if(is%4==0){
//      #pragma omp parallel for num_threads(16)
//      for(size_t i=0;i<nxyz;i++) image3[i]+=image[i];
//     }
     
//     memset(image,0,nxyz*sizeof(float));
 }

// write("image1",image1,nxyz);
// to_header("image1","n1",nx,"o1",ox,"d1",dx);
// to_header("image1","n2",ny,"o2",oy,"d2",dy);
// to_header("image1","n3",nz,"o3",oz,"d3",dz);
//
// write("image2",image2,nxyz);
// to_header("image2","n1",nx,"o1",ox,"d1",dx);
// to_header("image2","n2",ny,"o2",oy,"d2",dy);
// to_header("image2","n3",nz,"o3",oz,"d3",dz);
//
// write("image3",image3,nxyz);
// to_header("image3","n1",nx,"o1",ox,"d1",dx);
// to_header("image3","n2",ny,"o2",oy,"d2",dy);
// to_header("image3","n3",nz,"o3",oz,"d3",dz);

// delete []image1;delete []image2;delete []image3;

// int nrtotal=souloc[5*(ns-1)+3]+souloc[5*(ns-1)+4];
// to_header("modeleddata","n1",nnt,"o1",ot,"d1",samplingRate);
// to_header("modeleddata","n2",nrtotal,"o2",0.,"d2",1);
// to_header("residual","n1",nnt,"o1",ot,"d1",samplingRate);
// to_header("residual","n2",nrtotal,"o2",0.,"d2",1);

 delete []prevSigmaX;delete []curSigmaX;
 delete []prevSigmaZ;delete []curSigmaZ;
 delete []prevLambdaX;delete []curLambdaX;
 delete []prevLambdaZ;delete []curLambdaZ;

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
  
  cudaFreeHost(h_prevLambdaX[i]);
  cudaFreeHost(h_curLambdaX[i]);
  cudaFreeHost(h_LambdaX4[i]);
  cudaFreeHost(h_LambdaX5[i]);
  cudaFreeHost(h_prevLambdaZ[i]);
  cudaFreeHost(h_curLambdaZ[i]);
  cudaFreeHost(h_LambdaZ4[i]);
  cudaFreeHost(h_LambdaZ5[i]);
  
  cudaFreeHost(h_imagei[i]);
  cudaFreeHost(h_imageo[i]);
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
  
  for(int i=0;i<nbuffCij;++i){
   cudaFree(d_image[gpu][i]);
  }
  delete []d_image[gpu];

  delete []d_data[gpu];
  
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
 delete []d_LambdaX;
 delete []d_LambdaZ;
 delete []d_c11;
 delete []d_c13;
 delete []d_c33;
 delete []d_image;
 delete []transfInStream;
 delete []computeStream;
 delete []transfOutStream;
 delete []damping;
 delete []d_damping;
 
 return;
}

