#include "cluster.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <array> 
#include <memory>
#include <errno.h>
#include <numeric>

#include "myio.h"

using namespace std;

bool genPBSScript(const string &scriptname,const string &jobname,const string &output,const string &command){
    ofstream ofs;
    if(!open_file(ofs,scriptname,ofstream::out)){
        cout<<"cannot open script file "<<scriptname<<endl;
        close_file(ofs);
        return false;
    }
    else{
        ofs<<"#!/bin/tcsh"<<endl;
        ofs<<"#"<<endl;
        ofs<<"#PBS -N "<<jobname<<endl;
        ofs<<"#PBS -l nodes=1:ppn=24"<<endl;
        ofs<<"#PBS -q default"<<endl;
        ofs<<"#PBS -V"<<endl;
//        ofs<<"#PBS -m e"<<endl;
//        ofs<<"#PBS -M huyle@stanford.edu"<<endl;
        ofs<<"#PBS -e "<<output<<endl;
        ofs<<"#PBS -o "<<output<<endl;
        ofs<<"#"<<endl;
        ofs<<"cd $PBS_O_WORKDIR"<<endl<<endl;
        ofs<<command<<endl;
        close_file(ofs);
        return true;
    }
}

string submitPBSScript(const string &scriptname){
    string command="qsub "+scriptname;
    string result=runCommand(command);
    cout<<"submitting script "<<scriptname<<endl<<result;
    if(result.find("error")!=string::npos){
        cout<<"error submit script "<<scriptname<<endl;
        return "error";
    }
    else{
        size_t pos=result.find(".");
        if(pos!=string::npos){
            string jobid=result.substr(0,pos);
            cout<<"job id "<<jobid<<endl;
            return jobid;
        }
        else{
            cout<<"error submit script "<<scriptname<<endl;
            return "error";
        }
    }
}

string getPBSJobState(Job &job){
//    string command="scontrol show job "+ jobid;
//    string result=runCommand(command);
//    size_t pos=result.find("JobState=");
//    size_t pos1=result.find(" ",pos);
//    size_t len=pos1-pos-9;
//    string state=result.substr(pos+9,len);
//    return state; // PENDING CONFIGURING RUNNING COMPLETING COMPLETED FAILED
    string command="qstat "+ job._jobId+" 2>&1";
    string result=runCommand(command);
//    cout<<"result of running qstat"<<endl<<result<<"end result"<<endl;
    if(result.find("pbs_connect")==0) return "UNKNOWN"; //system busy. try again later
    string s=result.substr(0,5);
    if(s.compare("qstat")==0){
        fprintf(stderr,"job %s not in qstat anymore. checking output file\n",job._jobId.c_str());
        string state=get_s(job._outFile,"jobstate");
        if(state.compare("C")!=0) state="E";
        return state;
    }
    char *c=new char[result.size()+1];
    char *c0=c;
    strcpy(c,result.c_str());
    strtok(c," ");
    int count=0;
    string state;
    while(c!=nullptr && count<17){
        state=c;
        c=strtok(nullptr," ");
        count++;
    }
    delete []c0;
    return state; // PENDING CONFIGURING RUNNING COMPLETING COMPLETED FAILED
}

bool genScript(const string &scriptname,const string &jobname,const string &output,const string &command){
    ofstream ofs;
    if(!open_file(ofs,scriptname,ofstream::out)){
        cout<<"cannot open script file "<<scriptname<<endl;
        close_file(ofs);
        return false;
    }
    else{
        ofs<<"#!/bin/bash"<<endl;
        ofs<<"#"<<endl;
        ofs<<"#SBATCH --job-name="<<jobname<<endl;
        ofs<<"#SBATCH --output="<<output<<endl;
        ofs<<"#SBATCH --time=2:00:00"<<endl;
        ofs<<"#SBATCH --ntasks=1"<<endl;
        ofs<<"#SBATCH --cpus-per-task=10"<<endl;
        ofs<<"#SBATCH --mem-per-cpu=3000"<<endl;
        ofs<<"#SBATCH --gres gpu:8"<<endl;
        ofs<<"#SBATCH --gres-flags=enforce-binding"<<endl;
        ofs<<"#"<<endl;
        ofs<<"srun "<<command<<endl;
        close_file(ofs);
        return true;
    }
}

string submitScript(const string &scriptname){
    string command="sbatch "+scriptname+" 2>&1";
    string result=runCommand(command);
    cout<<"submitting script "<<scriptname<<endl<<result;
    if(result.find("Submitted")!=string::npos){
        size_t pos=result.rfind(" ");
        size_t len=result.size()-pos-2;
        string jobid=result.substr(pos+1,len);
        cout<<"job id "<<jobid<<endl;
        return jobid;
    }
    else{
        cout<<"error submit script "<<scriptname<<endl;
        return "error";
    }
}

string getJobState(const string &jobid){
//    string command="scontrol show job "+ jobid;
//    string result=runCommand(command);
//    size_t pos=result.find("JobState=");
//    size_t pos1=result.find(" ",pos);
//    size_t len=pos1-pos-9;
//    string state=result.substr(pos+9,len);
//    return state; // PENDING CONFIGURING RUNNING COMPLETING COMPLETED FAILED
    string command="sacct -j "+jobid+" 2>&1";
    string result=runCommand(command);
    string state;
    if(!result.empty()){
        char *c=new char[result.size()+1];
        char *c0=c;
        strcpy(c,result.c_str());
        c=strtok(c," ");
        int count=0;
        while(c!=nullptr && count<20){
            state=c;
            c=strtok(nullptr," ");
            count++;
        }
        delete []c0;
    }
    return state; // PENDING CONFIGURING RUNNING COMPLETING COMPLETED FAILED TIMEOUT
}

string runCommand(const string &command){
    array<char, 128> buffer;
    string result;
    unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(),"r"), pclose);
    if(!pipe){
        fprintf(stderr,"popen() failed when running command %s. check later.\n",command.c_str());
        return result;
    }
    while(fgets(buffer.data(),buffer.size(),pipe.get())!= nullptr) result+=buffer.data();
    return result;

//    pre c++11 version    
//    char buffer[128];
//    string result;
//    FILE* pipe=popen(command.c_str(),"r");
//    if(!pipe){
//        fprintf(stderr,"popen() failed when running command %s. error: %s\n",command.c_str(),strerror(errno));
//        pclose(pipe);
//        return result;
//    }
//    while(!feof(pipe)) {
//        if(fgets(buffer,128,pipe)!= NULL) result+=buffer;
//    }
//    pclose(pipe);
}

void Job::setJobId(const string &id){
    _jobId=id;
    return;
}

void Job::setJobState(const string &state){
    _jobState=state;
    return;
}

void Job::printJob(){
    cout<<"Job idx "<<_jobIdx<<" id "<<_jobId<<" scriptfile "<<_scriptFile<<" outfile "<<_outFile<<" gradfile "<<_gradFile<<" state "<<_jobState<<endl;
    return;
}

float server_job_partition(int njob,vector<float> &minute_per_job,vector<int> &partition){
    int nserver=minute_per_job.size();
    float a=0.f;
    for(int i=1;i<nserver;i++) a+=1.f/minute_per_job[i];
    a=1.f+minute_per_job[0]*a;
    int p0=njob/a;
    partition.push_back(p0);
    for(int i=1;i<nserver-1;i++){
        int pi=minute_per_job[0]/minute_per_job[i]*p0;
        partition.push_back(pi);
    }
    int p1=njob-std::accumulate(partition.begin(),partition.end(),0);
    partition.push_back(p1);
    return p1*minute_per_job[nserver-1];
}
