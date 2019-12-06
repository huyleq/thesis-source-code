#include <cmath>
#include <omp.h>
#include <cstdio>
#include <string>

#include "myio.h"
#include "init.h"
#include "wave.h"
#include "conversions.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);

 int nnx,nnz,nx,nz,nt,npad;
 float ox,oz,ot,dx,dz,dt;
 init2d(nnx,nnz,nx,nz,nt,dx,dz,dt,ox,oz,ot,npad);

 float *wavelet=new float[nt];
 read("wavelet",wavelet,nt);
 
 int ns;
 from_header("souloc","n2",ns);
 float *souloc=new float[ns*4]();
 read("souloc",souloc,ns*4);
 int *sloc=new int[ns*4]();
 #pragma omp parallel for num_threads(16)
 for(int is=0;is<ns;is++){
  sloc[is*4+0]=(souloc[is*4+0]-ox)/dx+0.5+npad;
  sloc[is*4+1]=(souloc[is*4+1]-oz)/dz+0.5+npad;
  sloc[is*4+2]=souloc[is*4+2];
  sloc[is*4+3]=souloc[is*4+3];
 }

 int nr;
 from_header("recloc","n2",nr);
 float *recloc=new float[nr*2]();
 read("recloc",recloc,nr*2);
 int *rloc=new int[nr*2]();
 #pragma omp parallel for num_threads(16)
 for(int ir=0;ir<nr;ir++){
  rloc[ir*2+0]=(recloc[ir*2+0]-ox)/dx+0.5+npad;
  rloc[ir*2+1]=(recloc[ir*2+1]-oz)/dz+0.5+npad;
 }

 float rate;
 get_param("rate",rate);
 int ratio=rate/dt+0.5;
 int ntNeg=std::round(abs(ot)/dt);
 int nnt_data=(nt-ntNeg-1)/ratio+1;
 float *data=new float[nr*nnt_data]();
 
 float *taper=new float[nnx*nnz]();
 init_abc(taper,nx,nz,npad);
 
 float *c11=new float[nnx*nnz]();
 float *c13=new float[nnx*nnz]();
 float *c33=new float[nnx*nnz]();

 string parameter=get_s("parameter");

 if(parameter.compare("cij")==0){
  init_model("c11",c11,nx,nz,npad); 
  init_model("c13",c13,nx,nz,npad);
  init_model("c33",c33,nx,nz,npad);
 }
 else if(parameter.compare("vepsdel")==0){
  float *v=new float[nnx*nnz]();
  init_model("v",v,nx,nz,npad); 
  float *eps=new float[nnx*nnz]();
  init_model("eps",eps,nx,nz,npad); 
  float *del=new float[nnx*nnz]();
  init_model("del",del,nx,nz,npad); 
 
  VEpsDel2Cij(c11,c13,c33,v,eps,del,1.,1.,1.,nnx*nnz);
  delete []v;delete []eps;delete []del;
 }
 else if(parameter.compare("vvnvh")==0){
  float *v=new float[nnx*nnz]();
  init_model("v",v,nx,nz,npad); 
  float *vn=new float[nnx*nnz]();
  init_model("vn",vn,nx,nz,npad); 
  float *vh=new float[nnx*nnz]();
  init_model("vh",vh,nx,nz,npad); 
 
  VVnVh2Cij(c11,c13,c33,v,vn,vh,1.,1.,1.,nnx*nnz);
  delete []v;delete []vn;delete []vh;
 }
 else fprintf(stderr,"please specify parameter=cij or parameter=vepsdel\n");
 
 synthetic_f(data,c11,c13,c33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);

 write("data",data,nr*nnt_data);
 to_header("data","n1",nr);
 to_header("data","n2",nnt_data,"o2",0.,"d2",rate);
 
 delete []wavelet;
 delete []c11;
 delete []c13;
 delete []c33;
 delete []data;
 delete []sloc; delete []souloc;
 delete []rloc; delete []recloc;
 delete []taper;

 myio_close();
 return 0;
}
