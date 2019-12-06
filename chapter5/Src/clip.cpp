#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

#include "myio.h"
#include "mylib.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);

    long long ntrace,nt;
    float ot,dt;
    from_header("datain","n1",nt,"o1",ot,"d1",dt);
    from_header("datain","n2",ntrace);
    fprintf(stderr,"ntrace=%lld nt=%d\n",ntrace,nt);

    int nr;
    from_header("recloc","n2",nr);
    float *recloc=new float[3*nr];
    read("recloc",recloc,3*nr);
    fprintf(stderr,"nr=%d should be equal to ntrace\n",nr);
    
    float *recloc1=new float[4*nr];
    #pragma omp parallel for num_threads(16)
    for(size_t i=0;i<nr;i++){
        memcpy(recloc1+i*4,recloc+i*3,3*sizeof(float));
        recloc1[i*4+3]=1.;
    }
    
    size_t n=nt*ntrace;
    cout<<"trying to allocate "<<4*n*1e-9<<" gb"<<endl;
    float *data;
    try{
        data=new float[n];
    }
    catch(std::bad_alloc &ba){
        cout<<"cannot allocate memory for data: "<<ba.what()<<endl;
    }
    read("datain",data,n);

    vector<float> timemark,threshold;
    get_array("timemark",timemark);
    get_array("threshold",threshold);

    #pragma omp parallel for num_threads(16)
    for(size_t i=0;i<ntrace;i++){
        for(int j=0;j<timemark.size();j++){
            int k=timemark[j]/dt;
            if(max_abs(data+i*nt+k,nt-k)>threshold[j]){
                memset(data+i*nt,0,nt*sizeof(float));
                recloc1[i*4+3]=0.;
                break;
            }
        }
    }
    
    write("dataout",data,n);
    to_header("dataout","n1",nt,"o1",ot,"d1",dt);
    to_header("dataout","n2",ntrace,"o2",0,"d2",1);

    write("reclocout",recloc1,4*nr);
    to_header("reclocout","n1",4,"o1",0,"d1",1);
    to_header("reclocout","n2",nr,"o2",0,"d2",1);

    delete []data;delete []recloc;
    myio_close();
    return 0;
}
