#include <cmath>
#include <omp.h>
#include <cstdio>
#include "myio.h"
#include "mylib.h"
#include "init.h"
#include "conversions.h"
#include "wave.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);

 int nnx,nnz,nx,nz,nt,npad;
 float ox,oz,ot,dx,dz,dt;
 init2d(nnx,nnz,nx,nz,nt,dx,dz,dt,ox,oz,ot,npad);

 float *wavelet=new float[nt];
 read("wavelet",wavelet,nt);
 
 float *c11=new float[nnx*nnz]();
 float *c13=new float[nnx*nnz]();
 float *c33=new float[nnx*nnz]();

 string parameter=get_s("parameter");

 if(parameter.compare("vepsdel")==0){
  fprintf(stderr,"parameter v eps del\n");

  float *v=new float[nnx*nnz]();
  init_model("v",v,nx,nz,npad); 
  float *eps=new float[nnx*nnz]();
  init_model("eps",eps,nx,nz,npad); 
  float *del=new float[nnx*nnz]();
  init_model("del",del,nx,nz,npad); 
 
  VEpsDel2Cij(c11,c13,c33,v,eps,del,1.,1.,1.,nnx*nnz);
  delete []v;delete []eps;delete []del;
 }
 else if(parameter.compare("vhepsdel")==0){
  fprintf(stderr,"parameter vh eps del\n");

  float *vh=new float[nnx*nnz]();
  init_model("vh",vh,nx,nz,npad); 
  float *eps=new float[nnx*nnz]();
  init_model("eps",eps,nx,nz,npad); 
  float *del=new float[nnx*nnz]();
  init_model("del",del,nx,nz,npad); 
 
  VhEpsDel2Cij(c11,c13,c33,vh,eps,del,1.,1.,1.,nnx*nnz);
  delete []vh;delete []eps;delete []del;
 }
 else if(parameter.compare("vnetadel")==0){
  fprintf(stderr,"parameter vn eta del\n");

  float *vn=new float[nnx*nnz]();
  init_model("vn",vn,nx,nz,npad); 
  float *eta=new float[nnx*nnz]();
  init_model("eta",eta,nx,nz,npad); 
  float *del=new float[nnx*nnz]();
  init_model("del",del,nx,nz,npad); 
 
  VnEtaDel2Cij(c11,c13,c33,vn,eta,del,1.,1.,1.,nnx*nnz);
  delete []vn;delete []eta;delete []del;
 }
 else if(parameter.compare("vhepseta")==0){
  fprintf(stderr,"parameter vh eps eta\n");

  float *vh=new float[nnx*nnz]();
  init_model("vh",vh,nx,nz,npad); 
  float *eps=new float[nnx*nnz]();
  init_model("eps",eps,nx,nz,npad); 
  float *eta=new float[nnx*nnz]();
  init_model("eta",eta,nx,nz,npad); 
 
  VhEpsEta2Cij(c11,c13,c33,vh,eps,eta,1.,1.,1.,nnx*nnz);
  delete []vh;delete []eta;delete []eps;
 }
 else if(parameter.compare("vvhdel")==0){
  fprintf(stderr,"parameter v vh del\n");

  float *v=new float[nnx*nnz]();
  init_model("v",v,nx,nz,npad); 
  float *vh=new float[nnx*nnz]();
  init_model("vh",vh,nx,nz,npad); 
  float *del=new float[nnx*nnz]();
  init_model("del",del,nx,nz,npad); 
 
  VVhDel2Cij(c11,c13,c33,v,vh,del,1.,1.,1.,nnx*nnz);
  delete []v;delete []vh;delete []del;
 }
 else if(parameter.compare("vnvhdel")==0){
  fprintf(stderr,"parameter vn vh del\n");

  float *vn=new float[nnx*nnz]();
  init_model("vn",vn,nx,nz,npad); 
  float *vh=new float[nnx*nnz]();
  init_model("vh",vh,nx,nz,npad); 
  float *del=new float[nnx*nnz]();
  init_model("del",del,nx,nz,npad); 
 
  VnVhDel2Cij(c11,c13,c33,vn,vh,del,1.,1.,1.,nnx*nnz);
  delete []vn;delete []vh;delete []del;
 }
 else if(parameter.compare("vvnvh")==0){
  fprintf(stderr,"parameter v vn vh\n");

  float *v=new float[nnx*nnz]();
  init_model("v",v,nx,nz,npad); 
  float *vn=new float[nnx*nnz]();
  init_model("vn",vn,nx,nz,npad); 
  float *vh=new float[nnx*nnz]();
  init_model("vh",vh,nx,nz,npad); 
 
  VVnVh2Cij(c11,c13,c33,v,vn,vh,1.,1.,1.,nnx*nnz);
  delete []v;delete []vh;delete []vn;
 }
 else if(parameter.compare("cij")==0){
  init_model("c11",c11,nx,nz,npad); 
  init_model("c13",c13,nx,nz,npad);
  init_model("c33",c33,nx,nz,npad);
 }
 else{
  fprintf(stderr,"please specify parameterization by parameter=something in commandline where something is one of vepsdel, vhepsdel, vnetadel,vhepseta, vvhdel, vnvhdel, cij, or vvnvh\n");
 }
 
 
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
 float *d0=new float[nr*nnt_data]();
 read("data",d0,nr*nnt_data);
 
 float *taper=new float[nnx*nnz]();
 init_abc(taper,nx,nz,npad);

 float max_subhx;
 get_param("max_subhx",max_subhx);
 int nhx=max_subhx/dx;
 float *image=new float[nnx*nnz*(2*nhx+1)]();
 
 odcig_f(image,d0,c11,c13,c33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nhx,nt,npad,dx,dz,dt,rate,ot);
 
 write("image",image,nnx*nnz*(2*nhx+1));
 to_header("image","n1",nnx,"o1",-dx*npad,"d1",dx);
 to_header("image","n2",nnz,"o2",-dz*npad,"d2",dz);
 to_header("image","n3",2*nhx+1,"o3",-max_subhx,"d3",dx);
 
 delete []wavelet;delete []d0;delete []sloc;delete []rloc;delete []taper;
 delete []c11;delete []c13;delete []c33;delete []image;
 delete []souloc;delete []recloc;

 myio_close();
 return 0;
}
