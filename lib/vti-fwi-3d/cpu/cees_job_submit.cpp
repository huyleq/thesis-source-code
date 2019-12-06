#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

#include "myio.h"
#include "mylib.h"
#include "cluster.h"

#include "cees_job_submit.h"

using namespace std;

void cees_job_init(vector<int> &shotid,int icall,const string &command,const string &workdir,vector<Job> &jobs){
    int nshot=shotid.size();
    for(int i=0;i<nshot;i++){
        string shotname="icall_"+to_string(icall)+"_shot_"+to_string(shotid[i]);
        string scriptfile=workdir+"/scripts/submit_"+shotname+".sh";
        string jobname=shotname;
        string outfile=workdir+"/output/"+shotname+".log";
        string gradfile=workdir+"/grads/fgcij_"+shotname+".H";
        string command1=command+" fgcij="+gradfile+" shotid="+to_string(shotid[i]);
        genPBSScript(scriptfile,jobname,outfile,command1);
        string id,state="INIT";
        Job job(i,id,scriptfile,outfile,gradfile,state);
        jobs.push_back(job);
    }
    return;
}

void cees_job_submit(Job &job){
    string id=submitPBSScript(job._scriptFile);
    int nfail=0;
    while(id.compare("error")==0 && nfail<MAX_FAIL){
        this_thread::sleep_for(chrono::seconds(5));
        id=submitPBSScript(job._scriptFile);
        nfail++;
    }
    string state;
    if(id.compare("error")!=0) state="SUBMITTED";
    else{
        state="FAIL_SUBMIT";
        fprintf(stderr,"script %s fails to submit MAX_FAIL %d times\n",job._scriptFile.c_str(),MAX_FAIL);
    }
    job.setJobId(id);
    job.setJobState(state);
    return;
}

void cees_job_submit(vector<Job> &jobs){
    int njob=jobs.size();
    int nsubmitted=0;
    for(int i=0;i<min(NMAX_JOB,njob);i++){
        cees_job_submit(jobs[i]);
        nsubmitted++;
        this_thread::sleep_for(chrono::seconds(1));
    }
    
    cout<<"submitted "<<nsubmitted<<" jobs"<<endl;
    this_thread::sleep_for(chrono::seconds(15));
    
    while(nsubmitted<njob){
        int nrun=cees_get_num_job("R");
        int nq=cees_get_num_job("Q");
        if(nrun>nq && nrun+nq<NMAX_JOB){
            cout<<nrun<<" jobs running. "<<nq<<" jobs in queue."<<endl;
            cees_job_submit(jobs[nsubmitted]);
            nsubmitted++;
            this_thread::sleep_for(chrono::seconds(1));
            nrun=cees_get_num_job("R");
            nq=cees_get_num_job("Q");
        }
        this_thread::sleep_for(chrono::seconds(5));
    }

    return;
}

void cees_job_collect(float *fg,float *temp_fg,size_t nelem,vector<Job> &jobs){
    int njob=jobs.size();
    vector<int> nfail(njob,0);
    int ncompleted=0;
    while(ncompleted<njob){
        for(int i=0;i<jobs.size();i++){
            string jobstate=jobs[i]._jobState;
            if(jobstate.compare("INIT")!=0 && jobstate.compare("FAIL_SUBMIT")!=0 && jobstate.compare("C")!=0 && jobstate.compare("E")!=0){
                string state=getPBSJobState(jobs[i]);
                if(state.compare("C")==0){
                    cout<<"job id "<<jobs[i]._jobId<<" state "<<state<<endl;
                    ncompleted++; 
                    readFromHeader(jobs[i]._gradFile,temp_fg,nelem);
                    cout<<"summing gradient from file "<<jobs[i]._gradFile<<endl;
                    #pragma omp parallel for
                    for(size_t j=0;j<nelem;j++) fg[j]+=temp_fg[j];
                }
                else if(state.compare("E")==0){
                    cout<<"job id "<<jobs[i]._jobId<<" state "<<state<<endl;
                    nfail[i]++;
                    if(nfail[i]>MAX_FAIL){
                        cout<<"job id "<<jobs[i]._jobId<<" reached MAX_FAIL "<<MAX_FAIL<<endl;
                        ncompleted++;
                        continue;
                    }
                    cout<<"resubmitting"<<endl;
                    cees_job_submit(jobs[i]);
                }
                jobs[i].setJobState(state);
            }
        }
    }

    return;
}

int cees_get_num_job(const string &state){
    string command="qselect -u huyle -s "+state+" | wc -l";
    string result=runCommand(command);
    return stoi(result);
}

void objFuncGradientCij_cees_cluster(float *fgcij,int nx,int ny,int nz,vector<int> &shotid,float pct,int max_shot_per_job,int icall,const string &command,const string &workdir){
    long long nxy=nx*ny,nxyz=nxy*nz,nn=3*nxyz;
    memset(fgcij,0,(nn+1)*sizeof(float));
//    random_shuffle(shotid.begin(),shotid.end());    

    float *fg=new float[nn+1];

    int nleft=shotid.size();
    vector<int>::iterator b=shotid.begin();
    while(b!=shotid.end()){
        int nshot=min(NMAX_JOB,nleft);
        vector<int>::iterator e=b+nshot;
        vector<int> shotid1(b,e);
        fprintf(stderr,"submitting from shot %d to shot %d\n",shotid1[0],shotid1[shotid1.size()-1]);

        vector<Job> jobs;
        vector<int> nfail;
    
        int nshotleft=shotid1.size(),njob=0,njobtotal=(shotid1.size()+max_shot_per_job-1)/max_shot_per_job;
        while(nshotleft>0){
            int nshot1job=min(nshotleft,max_shot_per_job);
            string shotname="icall_"+to_string(icall)+"_shot_";
            string shotlist;
            for(int i=0;i<nshot1job;i++){
                int id=i+njob*max_shot_per_job;
                if(i<nshot1job-1){
                    shotname+=to_string(shotid1[id])+"_";
                    shotlist+=to_string(shotid1[id])+",";
                }
                else{
                    shotname+=to_string(shotid1[id]);
                    shotlist+=to_string(shotid1[id]);
                }
            }
            string scriptfile=workdir+"/scripts/submit_"+shotname+".sh";
            string jobname=shotname;
            string outfile=workdir+"/output/"+shotname+".log";
            string gradfile=workdir+"/grads/fgcij_"+shotname+".H";
            string wantgrad="n";
            if((float)njob/njobtotal<=pct) wantgrad="y"; 
            string command1=command+" wantgrad="+wantgrad+" fgcij="+gradfile+" shotid="+shotlist;
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
                         readFromHeader(jobs[i]._gradFile,fg,nn+1);
                         cout<<"summing gradient from file "<<jobs[i]._gradFile<<endl;
                         #pragma omp parallel for
                         for(size_t j=0;j<nn+1;j++) fgcij[j]+=fg[j];
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

        b=e;
        nleft-=nshot;
    }

    delete []fg;
    
//    for(int i=0;i<jobs.size();i++) jobs[i].printJob();
    return;
}

