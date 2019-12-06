#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

#include "myio.h"
#include "cluster.h"

using namespace std;

void rtmCij_cees_cluster(float *image,int nx,int ny,int nz,vector<int> &shotid,float pct,int max_shot_per_job,int icall,const string &command,const string &workdir){
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
        string scriptfile=workdir+"/scripts/submit_"+shotname+".sh";
        string jobname=shotname;
        string outfile=workdir+"/output/"+shotname+".log";
        string gradfile=workdir+"/grads/image_"+shotname+".H";
        string wantgrad="n";
        if((float)njob/njobtotal<=pct) wantgrad="y"; 
        string command1=command+" wantgrad="+wantgrad+" image="+gradfile+" shotid="+shotlist;
        genPBSScript(scriptfile,jobname,outfile,command1);
        string id=submitPBSScript(scriptfile);
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
    
    float *image0=new float[nxyz];

    while(ncompleted<njob){
        for(int i=0;i<jobs.size();i++){
            this_thread::sleep_for(chrono::seconds(5));
            string id=jobs[i]._jobId;
            int idx=jobs[i]._jobIdx;
            string jobstate=jobs[i]._jobState;
            if(jobstate.compare("C")!=0 && jobstate.compare("E")!=0){
                string state=getPBSJobState(jobs[i]);
//                string state="C";
                if(state.compare("C")==0){
                    cout<<"job "<<idx<<" id "<<id<<" state "<<state<<endl;
                    ncompleted++; 
                    if((float)idx/njobtotal<=pct){
                     readFromHeader(jobs[i]._gradFile,image,nxyz);
                     cout<<"summing gradient from file "<<jobs[i]._gradFile<<endl;
                     #pragma omp parallel for
                     for(size_t j=0;j<nxyz;j++) image0[j]+=image[j];
                    }
                }
                else if(state.compare("E")==0){
                    cout<<"job "<<idx<<" id "<<id<<" state "<<state<<endl;
                    nfail[idx]++;
                    if(nfail[idx]>MAX_FAIL){
                        cout<<"job "<<idx<<" reached MAX_FAIL "<<MAX_FAIL<<endl;
                        ncompleted++;
                        continue;
                    }
                    cout<<" resubmitting"<<endl;
                    Job newjob=jobs[i];
                    string newid=submitPBSScript(newjob._scriptFile);
//                    string newid=to_string(idx);
                    if(newid.compare("error")!=0) newjob.setJobState("SUBMITTED");
                    newjob.setJobId(newid);
                    jobs.push_back(newjob);
                }
                jobs[i].setJobState(state);
            }
        }
    }

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

int main(int argc,char **argv){
 myio_init(argc,argv);

 int nx,ny,nz;
 float ox,oy,oz,dx,dy,dz;
 
 from_header("cij","n1",nx,"o1",ox,"d1",dx);
 from_header("cij","n2",ny,"o2",oy,"d2",dy);
 from_header("cij","n3",nz,"o3",oz,"d3",dz);
 
 long long nxy=nx*ny,nxyz=nxy*nz;

 string command=get_s("exec");
 command+=" par="+get_s("par")+" wavelet="+get_s("wavelet");
 command+=" souloc="+get_s("souloc")+" recloc="+get_s("recloc");
 command+=" cij="+get_s("cij")+" data="+get_s("data");
 string workdir=get_s("workdir");

 int ns;
 from_header("souloc","n2",ns);

 vector<int> shotid;
 bool providedShotId=get_array("shotid",shotid);
 if(!providedShotId){
  vector<int> shotrange;
  if(!get_array("shotrange",shotrange)){
    shotrange.push_back(0);
    shotrange.push_back(ns);
  }
  vector<int> badshot;
  get_array("badshot",badshot);
 for(int i=shotrange[0];i<shotrange[1];i++){
   if(find(badshot.begin(),badshot.end(),i)==badshot.end()) shotid.push_back(i);
  }
 }

 int max_shot_per_job=1;
 float pct=1.;
 get_param("max_shot_per_job",max_shot_per_job,"pct",pct);
    
 int icall;
 get_param("icall",icall);

 float *image=new float[nxyz];
 
 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 rtmCij_cees_cluster(image,nx,ny,nz,shotid,pct,max_shot_per_job,icall,command,workdir);
 
 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
  if(write("image",image,nxyz)){
   to_header("image","n1",nx,"o1",ox,"d1",dx);
   to_header("image","n2",ny,"o2",oy,"d2",dy);
   to_header("image","n3",nz,"o3",oz,"d3",dz);
  }

 delete []image;
 
 myio_close();
 return 0;
}
