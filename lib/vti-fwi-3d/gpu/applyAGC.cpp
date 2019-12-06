#include "myio.h"
#include "agc.h"

using namespace std;

int main(int argc,char ** argv){
    myio_init(argc,argv);
    
    int nt,ntr;
    float ot,dt;
    from_header("data","n1",nt,"o1",ot,"d1",dt);
    from_header("data","n2",ntr);
    
    int ntntr=nt*ntr;
    float *d=new float[ntntr]; read("data",d,ntntr);
    float *hd=new float[ntntr]();
    float *sd=new float[nt]();
    
    int halfwidth; get_param("halfwidth",halfwidth);
    for(int i=0;i<ntr;i++){
        agc(nt,halfwidth,d+i*nt,hd+i*nt,sd);
    }
    
    write("agcdata",hd,ntntr);
    to_header("agcdata","n1",nt,"o1",ot,"d1",dt);
    to_header("agcdata","n2",ntr,"o1",0,"d2",1);

    delete []d;
    delete []hd;delete []sd;

    myio_close();
    return 0;
}
