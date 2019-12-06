#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <map>
#include <algorithm>

#include "myio.h"
#include "cluster.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);

    map<string,int> jobid;
    vector<int> completed;
    vector<string> script;
    vector<string> gradfile;

    int nshot=5;
    
    for(int i=0;i<nshot;i++){
        string shotname="Shot"+to_string(i);
        string scriptname="./scripts/submit"+shotname+".sh";
        script.push_back(scriptname);
        string jobname=shotname;
        string output="./output/"+shotname+".log";
        string gradfilename="vnew."+shotname+".H";
        gradfile.push_back(gradfilename);
        string command="./Bin/testfunction.x v=v.H vnew="+gradfilename+" shotid="+to_string(i);
        genScript(scriptname,jobname,output,command);
        string id=submitScript(scriptname);
        jobid[id]=i;
    }

    int nx,ny,nz;
    float ox,oy,oz,dx,dy,dz;
    from_header("v","n1",nx,"o1",ox,"d1",dx);
    from_header("v","n2",ny,"o2",oy,"d2",dy);
    from_header("v","n3",nz,"o3",oz,"d3",dz);
    
    size_t nxyz=nx*ny*nz;
    float *vtotal=new float[nxyz]();
    
    while(completed.size()!=nshot){
        for(map<string,int>::iterator it=jobid.begin();it!=jobid.end();it++){
            string id=it->first;
            int shot=it->second;
            if(find(completed.begin(),completed.end(),shot)==completed.end()){
                string state=getJobState(id);
                if(state.compare("COMPLETED")==0){
                    cout<<"shot "<<shot<<" id "<<id<<" state "<<state<<endl;
                    completed.push_back(shot);
                    sumGrad(vtotal,gradfile[shot],nxyz);
                }
                else if(state.compare("FAILED")==0){
                    string newid=submitScript(script[shot]);
                    jobid[newid]=shot;
                }
            }
        }
    }
    
    cout<<"completed shots"<<endl;
    for(vector<int>::iterator it=completed.begin();it!=completed.end();it++) cout<<*it<<endl;
   
    write("vtotal",vtotal,nxyz);
    to_header("vtotal","n1",nx,"o1",ox,"d1",dx);
    to_header("vtotal","n2",ny,"o2",oy,"d2",dy);
    to_header("vtotal","n3",nz,"o3",oz,"d3",dz);
    
    delete []vtotal;

    myio_close();
    return 0;
}
