#include "myio.h"
#include "agc.h"

using namespace std;

int main(int argc,char ** argv){
    myio_init(argc,argv);
    
    int nt,ntr;
    float ot,dt;
    from_header("bgdata","n1",nt,"o1",ot,"d1",dt);
    from_header("bgdata","n2",ntr);
    
    int ntntr=nt*ntr;
    float *d=new float[ntntr]; read("bgdata",d,ntntr);
    float *d0=new float[ntntr]; read("data",d0,ntntr);
    float *hd=new float[ntntr]();
    float *hd0=new float[ntntr]();
    float *sd=new float[nt]();
    float *res=new float[ntntr]();
    float *adjsou=new float[ntntr]();
    
    int halfwidth; get_param("halfwidth",halfwidth);
    for(int i=0;i<ntr;i++){
        agc(nt,halfwidth,d+i*nt,hd+i*nt,sd);
        agc(nt,halfwidth,d0+i*nt,hd0+i*nt,sd);
    }
    
//    float objfunc=residualAGC(nt,ntr,halfwidth,d,d0,res,adjsou);
    float objfunc=residualAGC(nt,ntr,halfwidth,d,d0);
    fprintf(stderr,"objfunc=%.10f\n",objfunc);

    write("hd",hd,ntntr);
    to_header("hd","n1",nt,"o1",ot,"d1",dt);
    to_header("hd","n2",ntr,"o1",0,"d2",1);

    write("hd0",hd0,ntntr);
    to_header("hd0","n1",nt,"o1",ot,"d1",dt);
    to_header("hd0","n2",ntr,"o1",0,"d2",1);

    write("res",res,ntntr);
    to_header("res","n1",nt,"o1",ot,"d1",dt);
    to_header("res","n2",ntr,"o1",0,"d2",1);

    write("adjsou",d,ntntr);
    to_header("adjsou","n1",nt,"o1",ot,"d1",dt);
    to_header("adjsou","n2",ntr,"o1",0,"d2",1);

    delete []d;delete []d0;
    delete []hd;delete []hd0;delete []sd;delete []res;delete []adjsou;

    myio_close();
    return 0;
}
