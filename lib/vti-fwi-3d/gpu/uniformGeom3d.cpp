#include <cstring>
#include "myio.h"

using namespace std;

int main(int argc,char ** argv){
 myio_init(argc,argv);

 int nsx,nsy,nrx,nry;
 float osx,osy,dsx,dsy,zs,orx,ory,drx,dry,zr;
 get_param("nsoux",nsx,"osoux",osx,"dsoux",dsx);
 get_param("nsouy",nsy,"osouy",osy,"dsouy",dsy);
 get_param("nrecx",nrx,"orecx",orx,"drecx",drx);
 get_param("nrecy",nry,"orecy",ory,"drecy",dry);
 get_param("zsou",zs,"zrec",zr);

 int ns=nsx*nsy;
 float *souloc=new float[5*ns]();
 #pragma omp parallel for num_threads(16)
 for(int i=0;i<nsy;i++){
  for(int j=0;j<nsx;j++){
   int k=j+i*nsx;
   souloc[5*k]=osx+j*dsx;
   souloc[5*k+1]=osy+i*dsy;
   souloc[5*k+2]=zs;
   souloc[5*k+3]=nrx*nry;
   souloc[5*k+4]=k*nrx*nry;
  }
 }

 int nr=nrx*nry*ns;
 float *recloc=new float[3*nr]();
 #pragma omp parallel for num_threads(16)
 for(int i=0;i<nry;++i){
  for(int j=0;j<nrx;j++){
   int k=j+i*nrx;
   recloc[3*k]=orx+j*drx;
   recloc[3*k+1]=ory+i*dry;
   recloc[3*k+2]=zr;
  }
 }
 
 for(int i=1;i<ns;++i) memcpy(recloc+i*3*nrx*nry,recloc,3*nrx*nry*sizeof(float));

 to_header("souloc","n1",5,"o1",1.,"d1",1.);
 to_header("souloc","n2",ns,"o2",1.,"d2",1.);
 write("souloc",souloc,5*ns);

 to_header("recloc","n1",3,"o1",1.,"d1",1.);
 to_header("recloc","n2",nr,"o2",1.,"d2",1.);
 write("recloc",recloc,3*nr);

 delete []souloc; delete []recloc;

 myio_close();

 return 0;
}
