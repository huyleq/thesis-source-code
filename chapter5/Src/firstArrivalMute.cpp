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
    float *recloc=new float[4*nr];
    read("recloc",recloc,4*nr);
    fprintf(stderr,"nr=%d should be equal to ntrace\n",nr);

    int ns;
    from_header("souloc","n2",ns);
    float *souloc=new float[5*ns];
    read("souloc",souloc,5*ns);
    
    float *data=new float[nt*ntrace];
    size_t n=(long long)nt*(long long)ntrace;
    read("datain",data,n);

    float waterv=1500.f; get_param("waterv",waterv);

    #pragma omp parallel for num_threads(16)
    for(int i=0;i<ns;i++){
//        fprintf(stderr,"shot %d\n",i);
        float sx=souloc[5*i];
        float sy=souloc[5*i+1];
        float sz=souloc[5*i+2];
        int nr1=souloc[5*i+3];
        int start=souloc[5*i+4];
//        fprintf(stderr,"sx %.10f sy %.10f sz %.10f nr1 %d start %d\n",sx,sy,sz,nr1,start);
        for(int j=0;j<nr1;j++){
            long long ir=start+j;
            float gx=recloc[4*ir]; gx=gx-sx;
            float gy=recloc[4*ir+1]; gy=gy-sy;
            float gz=recloc[4*ir+2]; gz=gz-sz;
            int k=(sqrt(gx*gx+gy*gy+gz*gz)/waterv-ot)/dt;
//            fprintf(stderr," gx %.10f gy %.10f gz %.10f k %d\n",gx,gy,gz,k);
            memset(data+ir*(long long)nt,0,k*sizeof(float));
        }
    }
    
    write("dataout",data,n);
    to_header("dataout","n1",nt,"o1",ot,"d1",dt);
    to_header("dataout","n2",ntrace,"o2",0,"d2",1);

    delete []data;delete []recloc;

    myio_close();
    return 0;
}
