#include <iostream>
#include "myio.h"
#include "mylib.h"
#include "agc.h"

using namespace std;

int main(int argc,char ** argv){
    myio_init(argc,argv);
    
    int nt,ntr;
    float ot,dt;
    from_header("data","n1",nt,"o1",ot,"d1",dt);
    
    float *d=new float[nt]; read("data",d,nt);
    float *d0=new float[nt]; read("data0",d0,nt);
    float *res=new float[nt]();
    float *adjsou=new float[nt]();
    float *hd=new float[nt]();
    float *hd0=new float[nt]();
    float *sd=new float[nt]();
    
    int halfwidth; get_param("halfwidth",halfwidth);
    residualAGC(nt,1,halfwidth,d,d0,res,adjsou);
    float obj0=dot_product(res,res,nt)/2;
    
    write("adjsou",adjsou,nt);
    to_header("adjsou","n1",nt,"o1",ot,"d1",dt);

    float delta; get_param("delta",delta);
    agc(nt,halfwidth,d0,hd0,sd);
    int m=3;
    float *adjsou1=new float[nt*m]();
    for(int k=0;k<m;k++){
        for(int i=0;i<nt;i++){
            d[i]+=delta;
            agc(nt,halfwidth,d,hd,sd);
            subtract(res,hd,hd0,nt);
            float obj=dot_product(res,res,nt)/2;
            adjsou1[i+k*nt]=(obj-obj0)/delta;
            d[i]-=delta;
        }
        delta=delta/2;
    }

    write("adjsou1",adjsou1,nt*m);
    to_header("adjsou1","n1",nt,"o1",ot,"d1",dt);
    to_header("adjsou1","n2",m,"o1",0,"d1",1);

    delete []d; delete []d0;
    delete []res;delete []adjsou;
    delete []hd;delete []hd0;delete []sd;
    delete []adjsou1;

    myio_close();
    return 0;
}
