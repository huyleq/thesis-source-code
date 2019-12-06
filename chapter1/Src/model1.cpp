#include "myio.h"
#include "mylib.h"

int main(int argc,char ** argv){
 myio_init(argc,argv);
 
 int nx,nz;
 float ox,oz,ot,dx,dz; 
 get_param("nx",nx,"dx",dx,"ox",ox);
 get_param("nz",nz,"dz",dz,"oz",oz);
 
 float *v=new float[nx*nz]();
 
 float v1,v2,v3,d1,d2;
 get_param("v1",v1,"v2",v2,"v3",v3);
 get_param("d1",d1,"d2",d2);

 int nz1=d1/dz+1,nz2=d2/dz+1;
 set(v,v1,nz1*nx);
 set(v+nz1*nx,v2,(nz2-nz1)*nx);
 set(v+nz2*nx,v3,(nz-nz2)*nx);
 
 write("v",v,nx*nz);
 to_header("v","n1",nx,"o1",ox,"d1",dx);
 to_header("v","n2",nz,"o2",oz,"d2",dz);
 
 delete []v;
 
 myio_close();
 
 return 0;
}
