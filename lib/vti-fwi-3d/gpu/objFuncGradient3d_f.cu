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
#include "agc.h"

using namespace std;

void objFuncGradientCij_cluster(float *fgcij,int nx,int ny,int nz,vector<int> &shotid,float pct,int max_shot_per_job,int icall,const string &command,float &time_in_min){
    chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
    
    long long nxy=nx*ny,nxyz=nxy*nz,nn=3*nxyz;
    memset(fgcij,0,(nn+1)*sizeof(float));
//    random_shuffle(shotid.begin(),shotid.end());    

    vector<Job> jobs;
    vector<int> nfail;

    int nshotleft=shotid.size(),njob=0,njobtotal=(shotid.size()+max_shot_per_job-1)/max_shot_per_job;
    while(nshotleft>0){
        this_thread::sleep_for(chrono::seconds(1));
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
        string gradfile="./grads/fgcij_"+shotname+".H";
        string wantgrad="n";
        if((float)njob/njobtotal<=pct) wantgrad="y"; 
        string command1=command+" wantgrad="+wantgrad+" fgcij="+gradfile+" shotid="+shotlist;
        genScript(scriptfile,jobname,outfile,command1);
        string id=submitScript(scriptfile);
//        string id=to_string(njob);
        string state;
        int nerror=0;
        while(id.compare("error")==0 && nerror<MAX_FAIL){
            this_thread::sleep_for(chrono::seconds(5));
            id=submitScript(scriptfile);
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
    
    float *fg=new float[nn+1];

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
                    if((float)idx/njobtotal<=pct){
                     readFromHeader(jobs[i]._gradFile,fg,nn+1);
                     cout<<"summing gradient from file "<<jobs[i]._gradFile<<endl;
                     #pragma omp parallel for
                     for(size_t j=0;j<nn+1;j++) fgcij[j]+=fg[j];
                    }
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

    delete []fg;
    
    chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
    chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
    time_in_min=time.count()/60.f;
    
//    for(int i=0;i<jobs.size();i++) jobs[i].printJob();
    return;
}

double objFuncGradientVEpsDel_cluster(float *gvepsdel,float *vepsdel,int nx,int ny,int nz,int npad,float oz,float dz,float wbottom,float v0,float eps0,vector<int> &shotid,float pct,int max_shot_per_job,int icall,const string &command){
    double objfunc=0.;
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
        string gradfile="./grads/gcij_"+shotname+".H";
        string wantgrad="n";
        if((float)njob/njobtotal<=pct) wantgrad="y"; 
        string command1=command+" wantgrad="+wantgrad+" gcij="+gradfile+" shotid="+shotlist;
        genScript(scriptfile,jobname,outfile,command1);
        string id=submitScript(scriptfile);
//        string id=to_string(njob);
        string state;
        if(id.compare("error")!=0) state="SUBMITTED";
        int idx=njob;
        Job job(idx,id,scriptfile,outfile,gradfile,state);
        jobs.push_back(job);
        nfail.push_back(0);
        njob++;
        nshotleft-=nshot1job;
    }

    this_thread::sleep_for(chrono::seconds(15));

    cout<<"submitted "<<njob<<" jobs"<<endl;
//    for(int i=0;i<jobs.size();i++) jobs[i].printJob();

    int ncompleted=0;
    
    float *gcij=new float[3*nxyz]();
    float *gc11=gcij,*gc13=gcij+nxyz,*gc33=gcij+2*nxyz;

    while(ncompleted<njob){
        for(int i=0;i<jobs.size();i++){
            string id=jobs[i]._jobId;
            int idx=jobs[i]._jobIdx;
            string jobstate=jobs[i]._jobState;
            if(jobstate.compare("COMPLETED")!=0 && jobstate.compare("FAILED")!=0){
                string state=getJobState(id);
//                string state="COMPLETED";
                if(state.compare("COMPLETED")==0){
                    cout<<"job "<<idx<<" id "<<id<<" state "<<state<<endl;
                    ncompleted++; 
                    objfunc+=stod(get_s(jobs[i]._outFile,"objfunc"));
                    if((float)idx/njobtotal<=pct){
                     readFromHeader(jobs[i]._gradFile,gvepsdel,3*nxyz);
                     cout<<"summing gradient from file "<<jobs[i]._gradFile<<endl;
                     #pragma omp parallel for
                     for(int i=0;i<3*nxyz;i++) gcij[i]+=gvepsdel[i];
                    }
                }
                else if(state.compare("FAILED")==0){
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

    zeroBoundary(gc11,nx,ny,nz,npad);
    zeroBoundary(gc13,nx,ny,nz,npad);
    zeroBoundary(gc33,nx,ny,nz,npad);
    int nwbottom=(wbottom-oz)/dz+1-npad;
    memset(gc11+npad*nxy,0,nwbottom*nxy*sizeof(float));
    memset(gc13+npad*nxy,0,nwbottom*nxy*sizeof(float));
    memset(gc33+npad*nxy,0,nwbottom*nxy*sizeof(float));
    
    float *gv=gvepsdel,*geps=gvepsdel+nxyz,*gdel=gvepsdel+2*nxyz;
    float *v=vepsdel,*eps=vepsdel+nxyz,*del=vepsdel+2*nxyz;
    GradCij2GradVEpsDel(gv,geps,gdel,gc11,gc13,gc33,v,eps,del,v0,eps0,1.,nxyz);
    
    delete []gcij;
    
//    for(int i=0;i<jobs.size();i++) jobs[i].printJob();
    return objfunc;
}

double objFuncGradientVEpsDel(float *gv,float *geps,float *gdel,float *souloc,int ns,vector<int> &shotid,float *recloc,float *wavelet,float *v,float *eps,float *del,float *padboundaryV,float *randboundaryV,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,float v0,float eps0,float wbottom,vector<int> &GPUs,int ngpugroup){
 
 long long nxy=nx*ny,nxyz=nxy*nz,nn=3*nxyz;
 long long nboundary=nxyz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);
 
 float *cij=new float[nn];
 float *c11=cij,*c13=cij+nxyz,*c33=cij+2*nxyz;
 
 float *fgcij=new float[nn+1];
 float *gcij=fgcij+1;
 float *gc11=gcij,*gc13=gcij+nxyz,*gc33=gcij+2*nxyz;
 
 float *boundaryEps=new float[nboundary]; getBoundary(boundaryEps,eps,nx,ny,nz,npad);
 float *boundaryDel=new float[nboundary]; getBoundary(boundaryDel,del,nx,ny,nz,npad);
 
 float *padboundaryCij=new float[3*nboundary];
 float *randboundaryCij=new float[3*nboundary];
 
 VEpsDel2Cij(c11,c13,c33,v,eps,del,v0,eps0,1.,nxyz);
 VEpsDel2Cij(padboundaryCij,padboundaryCij+nboundary,padboundaryCij+2*nboundary,padboundaryV,boundaryEps,boundaryDel,v0,eps0,1.,nboundary);
 VEpsDel2Cij(randboundaryCij,randboundaryCij+nboundary,randboundaryCij+2*nboundary,randboundaryV,boundaryEps,boundaryDel,v0,eps0,1.,nboundary);
 
 if(ngpugroup>1) objFuncGradientCij3d(fgcij,souloc,ns,shotid,recloc,wavelet,cij,padboundaryCij,randboundaryCij,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,GPUs,ngpugroup);
 else objFuncGradientCij3d_f(fgcij,souloc,ns,shotid,recloc,wavelet,cij,padboundaryCij,randboundaryCij,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,GPUs);
 
 double objfunc=fgcij[0];
 
 zeroBoundary(gc11,nx,ny,nz,npad);
 zeroBoundary(gc13,nx,ny,nz,npad);
 zeroBoundary(gc33,nx,ny,nz,npad);
 int nwbottom=(wbottom-oz)/dz+1-npad;
 memset(gc11+npad*nxy,0,nwbottom*nxy*sizeof(float));
 memset(gc13+npad*nxy,0,nwbottom*nxy*sizeof(float));
 memset(gc33+npad*nxy,0,nwbottom*nxy*sizeof(float));
 
 GradCij2GradVEpsDel(gv,geps,gdel,gc11,gc13,gc33,v,eps,del,v0,eps0,1.,nxyz);
 
// write("gc11",gc11,nxyz);
// to_header("gc11","n1",nx,"o1",ox,"d1",dx);
// to_header("gc11","n2",ny,"o2",oy,"d2",dy);
// to_header("gc11","n3",nz,"o3",oz,"d3",dz);
//
// write("gc13",gc13,nxyz);
// to_header("gc13","n1",nx,"o1",ox,"d1",dx);
// to_header("gc13","n2",ny,"o2",oy,"d2",dy);
// to_header("gc13","n3",nz,"o3",oz,"d3",dz);
//
// write("gc33",gc33,nxyz);
// to_header("gc33","n1",nx,"o1",ox,"d1",dx);
// to_header("gc33","n2",ny,"o2",oy,"d2",dy);
// to_header("gc33","n3",nz,"o3",oz,"d3",dz);

 delete []cij;
 delete []fgcij;
 delete []boundaryEps;delete []boundaryDel;delete []padboundaryCij;delete []randboundaryCij;
 return objfunc;
}
 
void objFuncGradientCij3d(float *fgcij,float *souloc,int ns,vector<int> &shotid,float *recloc,float *wavelet,float *cij,float *padboundaryCij,float *randboundaryCij,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,vector<int> &GPUs,int ngpugroup){
    if(ngpugroup==1) objFuncGradientCij3d_f(fgcij,souloc,ns,shotid,recloc,wavelet,cij,padboundaryCij,randboundaryCij,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,GPUs);
    else{
        //GPUs is array of all gpus
        //assume nshot >= ngpugroup
        long long nxyz=nx*ny*nz,nn=3*nxyz;
        memset(fgcij,0,(nn+1)*sizeof(float));
        
        float **fg=new float*[ngpugroup]();
        float **tcij=new float*[ngpugroup]();
        
        int ngpu1=GPUs.size()/ngpugroup;
        vector<vector<int>> gpugroups;
        
        int nshot1=shotid.size()/ngpugroup;
        int r=shotid.size()%ngpugroup;
        int m=(nshot1+1)*r;
        vector<vector<int>> shotidgroups;
        
        for(int i=0;i<ngpugroup;i++){
            fg[i]=new float[nn+1];
            tcij[i]=new float[nn]; memcpy(tcij[i],cij,nn*sizeof(float));
            
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
//            fprintf(stderr,"group %d\n",i);
//            for(vector<int>::iterator it=gpugroups[i].begin();it!=gpugroups[i].end();it++) fprintf(stderr," gpu %d\n",*it);
//            for(vector<int>::iterator it=shotidgroups[i].begin();it!=shotidgroups[i].end();it++) fprintf(stderr," shotid %d\n",*it);
        }
        
        vector<thread> threads;
        for(int i=0;i<ngpugroup;i++) threads.push_back(thread(objFuncGradientCij3d_f,fg[i],souloc,ns,std::ref(shotidgroups[i]),recloc,wavelet,tcij[i],padboundaryCij,randboundaryCij,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,std::ref(gpugroups[i])));
    
        for(int i=ngpugroup-1;i>=0;i--){
            threads[i].join();   
            #pragma omp parallel for 
            for(size_t j=0;j<nn+1;j++) fgcij[j]+=fg[i][j];  
            delete[]fg[i]; delete []tcij[i];
        }
    
        delete []fg; delete []tcij;
    }
    return;
}

void objFuncGradientCij3d_f(float *fgcij,float *souloc,int ns,vector<int> &shotid,float *recloc,float *wavelet,float *cij,float *padboundaryCij,float *randboundaryCij,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,vector<int> &GPUs){

 double objfunc=0.;

// vector<int> GPUs;
// get_array("gpu",GPUs);
 int NGPU=GPUs.size();
// fprintf(stderr,"Total # GPUs = %d\n",NGPU);
// fprintf(stderr,"GPUs used are:\n");
// for(int i=0;i<NGPU;i++) fprintf(stderr,"%d ",GPUs[i]);
// fprintf(stderr,"\n");
 float dx2=dx*dx,dy2=dy*dy,dz2=dz*dz,dt2=dt*dt;
 float dt2dx2=dt2/dx2,dt2dy2=dt2/dy2,dt2dz2=dt2/dz2;
 int nxy=nx*ny;
 long long nxyz=nxy*nz,nboundary=nxyz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);
 int samplingTimeStep=std::round(samplingRate/dt);
 int nnt=(nt-1)/samplingTimeStep+1;

 float *gcij=fgcij+1;
 float *gc11=gcij,*gc13=gcij+nxyz,*gc33=gcij+2*nxyz;
 float *c11=cij,*c13=cij+nxyz,*c33=cij+2*nxyz;

 memset(gc11,0,nxyz*sizeof(float));
 memset(gc13,0,nxyz*sizeof(float));
 memset(gc33,0,nxyz*sizeof(float));

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
 float *h_data,*h_res[2];
 float *h_gc11i[2],*h_gc11o[2],*h_gc13i[2],*h_gc13o[2],*h_gc33i[2],*h_gc33o[2];

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
  
  cudaHostAlloc(&h_gc11i[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_gc11o[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_gc13i[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_gc13o[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_gc33i[i],nByteBlock,cudaHostAllocDefault);
  cudaHostAlloc(&h_gc33o[i],nByteBlock,cudaHostAllocDefault);
 }

 const int nbuffSigma=NUPDATE+2;
 
 int **d_recIndex=new int*[NGPU]();
 float **d_data=new float*[NGPU]();
 float ***d_res=new float**[NGPU]();
 
 float ****d_SigmaX=new float ***[NGPU]();
 float ****d_SigmaZ=new float ***[NGPU]();
 float ****d_LambdaX=new float ***[NGPU]();
 float ****d_LambdaZ=new float ***[NGPU]();
 
 const int nbuffCij=NUPDATE+4;
 float ***d_c11=new float**[NGPU]();
 float ***d_c13=new float**[NGPU]();
 float ***d_c33=new float**[NGPU]();
 float ***d_gc11=new float**[NGPU]();
 float ***d_gc13=new float**[NGPU]();
 float ***d_gc33=new float**[NGPU]();
 
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
 
  d_gc11[gpu]=new float*[nbuffCij]();
  d_gc13[gpu]=new float*[nbuffCij]();
  d_gc33[gpu]=new float*[nbuffCij]();
  for(int i=0;i<nbuffCij;++i){
   cudaMalloc(&d_gc11[gpu][i],nByteBlock);
   cudaMalloc(&d_gc13[gpu][i],nByteBlock);
   cudaMalloc(&d_gc33[gpu][i],nByteBlock);
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
//     fprintf(stderr,"nr=%d irbegin=%d nnt=%d\n",nr,irbegin,nnt);

	 int *recIndex=new int[nr];
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
	  recIndex[ir]=ixy+iz*nxy;
	  recIndexBlock[ir]=ixy+(iz%HALF_STENCIL)*nxy;
	 }
	 
	 int souIndexX=(souloc[5*is]-ox)/dx;
	 int souIndexY=(souloc[5*is+1]-oy)/dy;
	 int souIndexZ=(souloc[5*is+2]-oz)/dz;
	 int souIndex=souIndexX+souIndexY*nx+souIndexZ*nxy;
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
      cudaError_t e=cudaGetLastError();
      if(e!=cudaSuccess) fprintf(stderr,"shot %d GPU %d alloc error %s\n",is,gpu,cudaGetErrorString(e));
	 }
	 
//     fprintf(stderr,"put on absorbing boundary\n");
     putBoundary(padboundaryCij,c11,nx,ny,nz,npad);
     putBoundary(padboundaryCij+nboundary,c13,nx,ny,nz,npad);
     putBoundary(padboundaryCij+2*nboundary,c33,nx,ny,nz,npad);

//     fprintf(stderr,"fowrard propagation with absorbing boundary\n");

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
        
        cudaError_t e=cudaGetLastError();
        if(e!=cudaSuccess) fprintf(stderr,"GPU %d forward abc prop error %s\n",gpu,cudaGetErrorString(e));
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
        cudaError_t e=cudaGetLastError();
        if(e!=cudaSuccess) fprintf(stderr,"GPU %d forward abc prop synch error %s\n",gpu,cudaGetErrorString(e));
       }
       
       for(int i=0;i<threads.size();++i) threads[i].join();
       threads.erase(threads.begin(),threads.end());
     }
    
//     fprintf(stderr,"shot %d forward wavefield min %.10f max %.10f\n",is,min(curSigmaX,nxyz),max(curSigmaX,nxyz));
//     write("forwardAbcWfld",curSigmaX,nxyz);
//     to_header("forwardAbcWfld","n1",nx,"o1",ox,"d1",dx);
//     to_header("forwardAbcWfld","n2",ny,"o2",oy,"d2",dy);
//     to_header("forwardAbcWfld","n3",nz,"o3",oz,"d3",dz);
	 
     #pragma omp parallel for num_threads(16)
	 for(int ir=0;ir<nr;ir++){
	  int ir1=ir+irbegin;
      if(recloc[4*ir1+3]==0.f) memset(data+ir*nnt,0,nnt*sizeof(float));
     }

//     fprintf(stderr,"shot %d modeled data min %.10f max %.10f\n",is,min(data,nnt*nr),max(data,nnt*nr));
     
     float tpow=0.f; get_param("tpow",tpow);
     if(tpow!=0.f) tpower(data,nnt,ot,samplingRate,nr,tpow);

//     write("modeleddata",data,nnt*nr);
//     to_header("modeleddata","n1",nnt,"o1",ot,"d1",samplingRate);
//     to_header("modeleddata","n2",nr,"o2",0.,"d2",1);

//
//     if(is==0){
//      write("abforwardwfld",curSigmaX,nxyz);
//      to_header("abforwardwfld","n1",nx,"o1",ox,"d1",dx);
//      to_header("abforwardwfld","n2",ny,"o2",oy,"d2",dy);
//      to_header("abforwardwfld","n3",nz,"o3",oz,"d3",dz);
//     }

//	 fprintf(stderr,"compute objfunc\n");
     float *observedData=new float[nnt*nr];
     read("data",observedData,nnt*nr,(long long)nnt*(long long)irbegin);

//     fprintf(stderr,"shot %d observerd data min %.10f max %.10f\n",is,min(observedData,nnt*nr),max(observedData,nnt*nr));

     int objtype=0; get_param("objtype",objtype);
     if(objtype==0){
         #pragma omp parallel for reduction(+:objfunc) num_threads(16)
         for(size_t i=0;i<nnt*nr;i++){
             data[i]=data[i]-observedData[i];
             objfunc+=data[i]*data[i];
         }
     }
     else{
         int halfwidth=20; get_param("halfwidth",halfwidth);
         objfunc+=residualAGC(nnt,nr,halfwidth,data,observedData);
     }
     fprintf(stderr,"shot %d accumulated objfunc=%.10f\n",is,objfunc/2);
     delete []observedData;
     
//     write("adjsou",data,nnt*nr);
//     to_header("adjsou","n1",nnt,"o1",ot,"d1",samplingRate);
//     to_header("adjsou","n2",nr,"o2",0.,"d2",1);

//     write("residual",data,nnt*nr,ios_base::app);
	
//     fprintf(stderr,"put on random boundary\n");
     putBoundary(randboundaryCij,c11,nx,ny,nz,npad);
     putBoundary(randboundaryCij+nboundary,c13,nx,ny,nz,npad);
     putBoundary(randboundaryCij+2*nboundary,c33,nx,ny,nz,npad);

//     fprintf(stderr,"fowrard propagation with random boundary\n");

	 memset(prevSigmaX,0,nxyz*sizeof(float));
	 memset(curSigmaX,0,nxyz*sizeof(float));
	 memset(prevSigmaZ,0,nxyz*sizeof(float));
	 memset(curSigmaZ,0,nxyz*sizeof(float));
	 
	 //injecting source at time 0 to wavefields at time 1
     temp=dt2*wavelet[0];
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
        
        cudaError_t e=cudaGetLastError();
        if(e!=cudaSuccess) fprintf(stderr,"GPU %d forward random prop error %s\n",gpu,cudaGetErrorString(e));
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
        if(e!=cudaSuccess) fprintf(stderr,"GPU %d forward random prop synch error %s\n",gpu,cudaGetErrorString(e));
       }
       
       for(int i=0;i<threads.size();++i) threads[i].join();
       threads.erase(threads.begin(),threads.end());
     }
    
//     fprintf(stderr,"shot %d forward wavefield min %.10f max %.10f\n",is,min(curSigmaX,nxyz),max(curSigmaX,nxyz));
//     write("forwardRandWfld",curSigmaX,nxyz);
//     to_header("forwardRandWfld","n1",nx,"o1",ox,"d1",dx);
//     to_header("forwardRandWfld","n2",ny,"o2",oy,"d2",dy);
//     to_header("forwardRandWfld","n3",nz,"o3",oz,"d3",dz);

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
    		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaX[k2],prevSigmaX+ibn,h_curSigmaX[k2],curSigmaX+ibn,nByteBlock));
    		threads.push_back(thread(memcpyCpuToCpu2,h_prevSigmaZ[k2],prevSigmaZ+ibn,h_curSigmaZ[k2],curSigmaZ+ibn,nByteBlock));
    		threads.push_back(thread(memcpyCpuToCpu2,h_prevLambdaX[k2],prevLambdaX+ibn,h_curLambdaX[k2],curLambdaX+ibn,nByteBlock));
    		threads.push_back(thread(memcpyCpuToCpu2,h_prevLambdaZ[k2],prevLambdaZ+ibn,h_curLambdaZ[k2],curLambdaZ+ibn,nByteBlock));
    	    threads.push_back(thread(memcpyCpuToCpu3,h_gc11i[k2],gc11+ibn,h_gc13i[k2],gc13+ibn,h_gc33i[k2],gc33+ibn,nByteBlock));
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
          memcpyCpuToGpu3(d_gc11[0][kn],h_gc11i[k12],d_gc13[0][kn],h_gc13i[k12],d_gc33[0][kn],h_gc33i[k12],nByteBlock,transfInStream);
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
            gradientKernelTopBlock<<<grid,block,0,computeStream[gpu]>>>(d_gc11[gpu][ki2n],d_gc13[gpu][ki2n],d_gc33[gpu][ki2n],d_SigmaX[gpu][i+2][ki24],d_SigmaX[gpu][i+1][ki24],d_SigmaX[gpu][i][ki24],d_SigmaZ[gpu][i+2][ki24],d_SigmaZ[gpu][i+1][ki34],d_SigmaZ[gpu][i+1][ki24],d_SigmaZ[gpu][i+1][ki14],d_SigmaZ[gpu][i][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dx2,dy2,dz2,dt2);
            forwardKernelTopBlock<<<grid,block,0,computeStream[gpu]>>>(d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaX[gpu][i][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki34],d_LambdaZ[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki14],d_LambdaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
           }
           else if(ib==nb-1){
            gradientKernelBottomBlock<<<grid,block,0,computeStream[gpu]>>>(d_gc11[gpu][ki2n],d_gc13[gpu][ki2n],d_gc33[gpu][ki2n],d_SigmaX[gpu][i+2][ki24],d_SigmaX[gpu][i+1][ki24],d_SigmaX[gpu][i][ki24],d_SigmaZ[gpu][i+2][ki24],d_SigmaZ[gpu][i+1][ki34],d_SigmaZ[gpu][i+1][ki24],d_SigmaZ[gpu][i+1][ki14],d_SigmaZ[gpu][i][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dx2,dy2,dz2,dt2);
            forwardKernelBottomBlock<<<grid,block,0,computeStream[gpu]>>>(d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaX[gpu][i][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki34],d_LambdaZ[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki14],d_LambdaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
           }
           else{
            gradientKernel<<<grid,block,0,computeStream[gpu]>>>(d_gc11[gpu][ki2n],d_gc13[gpu][ki2n],d_gc33[gpu][ki2n],d_SigmaX[gpu][i+2][ki24],d_SigmaX[gpu][i+1][ki24],d_SigmaX[gpu][i][ki24],d_SigmaZ[gpu][i+2][ki24],d_SigmaZ[gpu][i+1][ki34],d_SigmaZ[gpu][i+1][ki24],d_SigmaZ[gpu][i+1][ki14],d_SigmaZ[gpu][i][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dx2,dy2,dz2,dt2);
            forwardKernel<<<grid,block,0,computeStream[gpu]>>>(d_LambdaX[gpu][i+2][ki24],d_LambdaX[gpu][i+1][ki24],d_LambdaX[gpu][i][ki24],d_LambdaZ[gpu][i+2][ki24],d_LambdaZ[gpu][i+1][ki34],d_LambdaZ[gpu][i+1][ki24],d_LambdaZ[gpu][i+1][ki14],d_LambdaZ[gpu][i][ki24],d_c11[gpu][ki2n],d_c13[gpu][ki2n],d_c33[gpu][ki2n],nx,ny,dt2dx2,dt2dy2,dt2dz2);
           }
           
           if(ib==souBlock){
            float source=dt2*wavelet[it+1];
            injectSource<<<1,1,0,computeStream[gpu]>>>(d_SigmaX[gpu][i+2][ki24],d_SigmaZ[gpu][i+2][ki24],source,souIndexBlock);
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
    	      memcpyGpuToGpu2(d_SigmaX[gpu+1][0][kn34],d_SigmaX[gpu][n2][kn34],d_SigmaX[gpu+1][1][kn34],d_SigmaX[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToGpu2(d_SigmaZ[gpu+1][0][kn34],d_SigmaZ[gpu][n2][kn34],d_SigmaZ[gpu+1][1][kn34],d_SigmaZ[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToGpu2(d_LambdaX[gpu+1][0][kn34],d_LambdaX[gpu][n2][kn34],d_LambdaX[gpu+1][1][kn34],d_LambdaX[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToGpu2(d_LambdaZ[gpu+1][0][kn34],d_LambdaZ[gpu][n2][kn34],d_LambdaZ[gpu+1][1][kn34],d_LambdaZ[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToGpu3(d_c11[gpu+1][kn3n],d_c11[gpu][kn3n],d_c13[gpu+1][kn3n],d_c13[gpu][kn3n],d_c33[gpu+1][kn3n],d_c33[gpu][kn3n],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToGpu3(d_gc11[gpu+1][kn3n],d_gc11[gpu][kn3n],d_gc13[gpu+1][kn3n],d_gc13[gpu][kn3n],d_gc33[gpu+1][kn3n],d_gc33[gpu][kn3n],nByteBlock,transfOutStream+gpu);
    	     }
    	     else{
              int n2=nbuffSigma-2,n1=nbuffSigma-1,k2=k%2,kn3=kgpu-NUPDATE-3,kn34=kn3%4,kn3n=kn3%nbuffCij;
    	      memcpyGpuToCpu2(h_SigmaX4[k2],d_SigmaX[gpu][n2][kn34],h_SigmaX5[k2],d_SigmaX[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToCpu2(h_SigmaZ4[k2],d_SigmaZ[gpu][n2][kn34],h_SigmaZ5[k2],d_SigmaZ[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToCpu2(h_LambdaX4[k2],d_LambdaX[gpu][n2][kn34],h_LambdaX5[k2],d_LambdaX[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToCpu2(h_LambdaZ4[k2],d_LambdaZ[gpu][n2][kn34],h_LambdaZ5[k2],d_LambdaZ[gpu][n1][kn34],nByteBlock,transfOutStream+gpu);
    	      memcpyGpuToCpu3(h_gc11o[k2],d_gc11[gpu][kn3n],h_gc13o[k2],d_gc13[gpu][kn3n],h_gc33o[k2],d_gc33[gpu][kn3n],nByteBlock,transfOutStream+gpu);
    	     }
          }
        }
        
        cudaError_t e=cudaGetLastError();
        if(e!=cudaSuccess) fprintf(stderr,"GPU %d backward prop error %s\n",gpu,cudaGetErrorString(e));
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
    	    memcpyCpuToCpu3(gc11+ibn,h_gc11o[k12],gc13+ibn,h_gc13o[k12],gc33+ibn,h_gc33o[k12],nByteBlock);
        }
       }
      
       for(int gpu=0;gpu<NGPU;gpu++){
        cudaSetDevice(GPUs[gpu]);
        cudaDeviceSynchronize();
        cudaError_t e=cudaGetLastError();
        if(e!=cudaSuccess) fprintf(stderr,"GPU %d backward prop synch error %s\n",gpu,cudaGetErrorString(e));
       }
       
       for(int i=0;i<threads.size();++i) threads[i].join();
       threads.erase(threads.begin(),threads.end());
     }
     
//     fprintf(stderr,"shot %d backward wavefield min %.10f max %.10f\n",is,min(curSigmaX,nxyz),max(curSigmaX,nxyz));
//     fprintf(stderr,"shot %d adjoint wavefield min %.10f max %.10f\n",is,min(curLambdaX,nxyz),max(curLambdaX,nxyz));
//     write("adjointWfld",curLambdaX,nxyz);
//     to_header("adjointWfld","n1",nx,"o1",ox,"d1",dx);
//     to_header("adjointWfld","n2",ny,"o2",oy,"d2",dy);
//     to_header("adjointWfld","n3",nz,"o3",oz,"d3",dz);
    
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
      cudaError_t e=cudaGetLastError();
      if(e!=cudaSuccess) fprintf(stderr,"shot %d GPU %d dealloc error %s\n",is,gpu,cudaGetErrorString(e));
	 }
 }

 fgcij[0]=objfunc/2;
 fprintf(stderr,"done propagation. objfunc=%.10f\n",fgcij[0]);

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
  
  cudaFreeHost(h_gc11i[i]);
  cudaFreeHost(h_gc11o[i]);
  cudaFreeHost(h_gc13i[i]);
  cudaFreeHost(h_gc13o[i]);
  cudaFreeHost(h_gc33i[i]);
  cudaFreeHost(h_gc33o[i]);
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
   cudaFree(d_gc11[gpu][i]);
   cudaFree(d_gc13[gpu][i]);
   cudaFree(d_gc33[gpu][i]);
  }
  delete []d_gc11[gpu];
  delete []d_gc13[gpu];
  delete []d_gc33[gpu];

  delete []d_res[gpu];
  
  if(gpu==0) cudaStreamDestroy(transfInStream[gpu]);
  cudaStreamDestroy(computeStream[gpu]);
  cudaStreamDestroy(transfOutStream[gpu]);
 
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) fprintf(stderr,"GPU %d dealloc error %s\n",gpu,cudaGetErrorString(e));
 }

 delete []d_recIndex;
 delete []d_data;
 delete []d_res;
 delete []d_SigmaX;
 delete []d_SigmaZ;
 delete []d_LambdaX;
 delete []d_LambdaZ;
 delete []d_c11;
 delete []d_c13;
 delete []d_c33;
 delete []d_gc11;
 delete []d_gc13;
 delete []d_gc33;
 delete []transfInStream;
 delete []computeStream;
 delete []transfOutStream;
 delete []damping;
 delete []d_damping;
 
 return;
}

void objFuncCij3d(float *f,float *souloc,int ns,vector<int> &shotid,float *recloc,float *wavelet,float *cij,float *padboundaryCij,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,vector<int> &GPUs,int ngpugroup){
    //GPUs is array of all gpus
    //assume nshot >= ngpugroup
    float *objf=new float[ngpugroup]();
    
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
    }
    
    vector<thread> threads;
    for(int i=0;i<ngpugroup;i++) threads.push_back(thread(objFuncCij3d_f,objf+i,souloc,ns,std::ref(shotidgroups[i]),recloc,wavelet,cij,padboundaryCij,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,std::ref(gpugroups[i])));

    for(int i=0;i<ngpugroup;i++){
        threads[i].join();   
        *f+=objf[i];
    }

    delete []objf;
    return;
}

void objFuncCij3d_f(float *f,float *souloc,int ns,vector<int> &shotid,float *recloc,float *wavelet,float *cij,float *padboundaryCij,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,vector<int> &GPUs){

 double objfunc=0.;

// vector<int> GPUs;
// get_array("gpu",GPUs);
 int NGPU=GPUs.size();
// fprintf(stderr,"Total # GPUs = %d\n",NGPU);
// fprintf(stderr,"GPUs used are:\n");
// for(int i=0;i<NGPU;i++) fprintf(stderr,"%d ",GPUs[i]);
// fprintf(stderr,"\n");

 float dx2=dx*dx,dy2=dy*dy,dz2=dz*dz,dt2=dt*dt;
 float dt2dx2=dt2/dx2,dt2dy2=dt2/dy2,dt2dz2=dt2/dz2;
 int nxy=nx*ny;
 long long nxyz=nxy*nz,nboundary=nxyz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);
 int samplingTimeStep=std::round(samplingRate/dt);
 int nnt=(nt-1)/samplingTimeStep+1;

 float *c11=cij,*c13=cij+nxyz,*c33=cij+2*nxyz;

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

  d_c11[gpu]=new float*[nbuffCij]();
  d_c13[gpu]=new float*[nbuffCij]();
  d_c33[gpu]=new float*[nbuffCij]();
  for(int i=0;i<nbuffCij;++i){
   cudaMalloc(&d_c11[gpu][i],nByteBlock);
   cudaMalloc(&d_c13[gpu][i],nByteBlock);
   cudaMalloc(&d_c33[gpu][i],nByteBlock);
   mem+=3*nByteBlock;
  }
 
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
     int itdata=1,krecord=-2,gpurecord=-2,ktransf=-2;
	
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
	 
     //put on absorbing boundary
     putBoundary(padboundaryCij,c11,nx,ny,nz,npad);
     putBoundary(padboundaryCij+nboundary,c13,nx,ny,nz,npad);
     putBoundary(padboundaryCij+2*nboundary,c33,nx,ny,nz,npad);

//     fprintf(stderr,"fowrard propagation with absorbing boundary\n");

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

//     write("modeleddata",data,nnt*nr,ios_base::app);
//
//     if(is==0){
//      write("abforwardwfld",curSigmaX,nxyz);
//      to_header("abforwardwfld","n1",nx,"o1",ox,"d1",dx);
//      to_header("abforwardwfld","n2",ny,"o2",oy,"d2",dy);
//      to_header("abforwardwfld","n3",nz,"o3",oz,"d3",dz);
//     }

//	 fprintf(stderr,"compute objfunc\n");
     float *observedData=new float[nnt*nr];
     read("data",observedData,nnt*nr,(long long)nnt*(long long)irbegin);
     #pragma omp parallel for reduction(+:objfunc) num_threads(16)
     for(size_t i=0;i<nnt*nr;i++){
         data[i]=data[i]-observedData[i];
         objfunc+=data[i]*data[i];
     }
     delete []observedData;
     
	 cudaFreeHost(h_data);
	
	 delete []recIndexBlock;
	 delete []recIndex;
     delete []data;

	 for(int gpu=0;gpu<NGPU;gpu++){
      cudaSetDevice(GPUs[gpu]);
	  cudaFree(d_recIndex[gpu]);
	  cudaFree(d_data[gpu]);
//      cudaError_t e=cudaGetLastError();
//      if(e!=cudaSuccess) fprintf(stderr,"shot %d GPU %d dealloc error %s\n",is,gpu,cudaGetErrorString(e));
	 }
 }

// int nrtotal=souloc[5*(ns-1)+3]+souloc[5*(ns-1)+4];
// to_header("modeleddata","n1",nnt,"o1",ot,"d1",samplingRate);
// to_header("modeleddata","n2",nrtotal,"o2",0.,"d2",1);
// to_header("residual","n1",nnt,"o1",ot,"d1",samplingRate);
// to_header("residual","n2",nrtotal,"o2",0.,"d2",1);

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
 
 *f=objfunc/2;
 return;
}

