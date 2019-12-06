#include <cstring>
#include "myio.h"

using namespace std;

int main(int argc,char ** argv){
 myio_init(argc,argv);

 int ns,nr;
 float os,ds,zs,o_r,dr,zrtop,zrbott;
 get_param("nsou",ns,"osou",os,"dsou",ds);
 get_param("nrec",nr,"orec",o_r,"drec",dr);
 get_param("zsou",zs,"zrectop",zrtop,"zrecbott",zrbott);
 int nr2=2*nr;

 float *souloc=new float[ns*4]();
 #pragma omp parallel for num_threads(16)
 for(int i=0;i<ns;i++){
  souloc[4*i]=os+i*ds;
  souloc[4*i+1]=zs;
  souloc[4*i+2]=nr2;
  souloc[4*i+3]=i*nr2;
 }

 float *recloc=new float[ns*nr2*2]();
 #pragma omp parallel for num_threads(16)
 for(int i=0;i<nr;++i){
  recloc[2*i]=o_r+i*dr;
  recloc[2*i+1]=zrtop;
  recloc[2*(i+nr)]=o_r+i*dr;
  recloc[2*(i+nr)+1]=zrbott;
 }
 
 for(int i=1;i<ns;++i) memcpy(recloc+i*nr2*2,recloc,nr2*2*sizeof(float));

 to_header("souloc","n1",4,"o1",1.,"d1",1.);
 to_header("souloc","n2",ns,"o2",1.,"d1",1.);
 write("souloc",souloc,ns*4);

 to_header("recloc","n1",2,"o1",1.,"d1",1.);
 to_header("recloc","n2",ns*nr2,"o2",1.,"d1",1.);
 write("recloc",recloc,ns*nr2*2);

 delete []souloc; delete []recloc;

 myio_close();

 return 0;
}
