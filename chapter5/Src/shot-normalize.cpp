#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <new>

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

    float *data;
    try{
        data=new float[nt*ntrace];
    }
    catch(std::bad_alloc &ba){
        cout<<"cannot allocate memory for data: "<<ba.what()<<endl;
    }
    size_t n=(long long)nt*(long long)ntrace;
    read("datain",data,n);
    
    float normval; get_param("normval",normval);

    #pragma omp parallel for num_threads(16)
    for(size_t i=0;i<ntrace;i++){
        float maxval=max(data+i*nt,nt);
        if(maxval>0.1){
            #pragma omp simd
            for(int j=0;j<nt;j++){
                data[j+i*nt]=data[j+i*nt]/maxval*normval;
            }
        }
    }
    
    write("dataout",data,n);
    to_header("dataout","n1",nt,"o1",ot,"d1",dt);
    to_header("dataout","n2",ntrace,"o2",0,"d2",1);

    myio_close();
    return 0;
}
