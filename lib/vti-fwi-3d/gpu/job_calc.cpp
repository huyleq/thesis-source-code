#include <cstdlib>
#include <cstdio>
#include <vector>
#include <numeric>
#include <string>
#include <cstring>

#include "myio.h"
#include "cluster.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
    vector<string> server; get_sarray("server",server);
    int m=server.size();
    
    vector<float> t; get_array("minute_per_job",t);
    
    int njob; get_param("njob",njob);
    
    vector<int> n;

    float total=server_job_partition(njob,t,n);

//    float a=0.f;
//    for(int i=1;i<m;i++) a+=1.f/t[i];
//    a=1.f+t[0]*a;
//    int n0=njob/a;
//    n.push_back(n0);
//
//    for(int i=1;i<m-1;i++){
//        int ni=t[0]/t[i]*n0;
//        n.push_back(ni);
//    }
//
//    int n1=njob-std::accumulate(n.begin(),n.end(),0);
//    n.push_back(n1);
    
    int start=0,end=start+n[0];
    for(int i=0;i<m;i++){
        fprintf(stderr,"%s should do %d jobs from %d to %d\n",server[i].c_str(),n[i],start,end);
        start=end;
        end+=n[i+1];
    }
    fprintf(stderr,"total time to do %d jobs is %f hours\n",njob,total/60.f);
    
    myio_close();
    return 0;
}
