#include "myio.h"

using namespace std;

int main(int argc,char ** argv){
 myio_init(argc,argv);

 int ns,nr;
 float os,ds,zs,minoffset,dr,zr;
 get_param("nsou",ns,"osou",os,"dsou",ds);
 get_param("nrec",nr,"minoffset",minoffset,"drec",dr);
 get_param("zsou",zs,"zrec",zr);

 float *souloc=new float[ns*5]();
 for(int i=0;i<ns;i++){
  souloc[4*i]=os+i*ds;
  souloc[4*i+1]=os+i*ds;
  souloc[4*i+2]=zs;
  souloc[4*i+3]=nr;
  souloc[4*i+4]=i*nr;
 }

 float *recloc=new float[ns*nr*3]();
 for(int is=0;is<ns;is++){
  for(int ir=0;ir<nr;ir++){
   recloc[2*(is*nr+ir)]=os+is*ds+minoffset+ir*dr;
   recloc[2*(is*nr+ir)]=os+is*ds+minoffset+ir*dr;
   recloc[2*(is*nr+ir)+1]=zr;
  }
 }
 
 to_header("souloc","n1",4,"o1",1.,"d1",1.);
 to_header("souloc","n2",ns,"o2",1.,"d1",1.);
 write("souloc",souloc,ns*4);

 to_header("recloc","n1",2,"o1",1.,"d1",1.);
 to_header("recloc","n2",ns*nr,"o2",1.,"d1",1.);
 write("recloc",recloc,ns*nr*2);

 delete []souloc; delete []recloc;

 myio_close();

 return 0;
}
