#include <iostream>
#include <cmath>
#include <omp.h>
#include <cstdio>

#include "myio.h"
#include "mylib.h"
#include "init.h"
#include "wave.h"
#include "HessianOp.h"
#include "LinearSolver.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);

int nnx,nnz,nx,nz,nt,npad;
float ox,oz,ot,dx,dz,dt;
init2d(nnx,nnz,nx,nz,nt,dx,dz,dt,ox,oz,ot,npad);
int nnxz=nnx*nnz;

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
 read("data",data,nr*nnt_data);

 float *taper=new float[nnxz]();
 init_abc(taper,nx,nz,npad);
 
 float wbottom; get_param("wbottom",wbottom);

 int padded; get_param("padded",padded);
 
 float *m=new float[nnxz]();

 string parameter=get_s("parameter");
 string hesstype=get_s("hesstype");

 if(parameter.compare("vepsdel")==0){
  fprintf(stderr,"parameter v eps del\n");
  float *vepsdel=new float[3*nnxz]();
  float *v=vepsdel,*eps=vepsdel+nnxz,*del=vepsdel+2*nnxz;
  
  float *pvepsdel=new float[3*nnxz]();
  float *pv=pvepsdel,*peps=pvepsdel+nnxz,*pdel=pvepsdel+2*nnxz;
  
  float *gvepsdel=new float[3*nnxz]();
  float *gv=gvepsdel,*geps=gvepsdel+nnxz,*gdel=gvepsdel+2*nnxz;
  
  init_model("v",v,nx,nz,npad); 
  init_model("eps",eps,nx,nz,npad); 
  init_model("del",del,nx,nz,npad); 

  float v0,eps0,del0;
  get_param("v0",v0,"eps0",eps0,"del0",del0);
 
  scale(v,v,1./v0,nnxz);
  scale(eps,eps,1./eps0,nnxz);
  scale(del,del,1./del0,nnxz);
  
  read("gv",gv,nnxz);
  read("geps",geps,nnxz);
  read("gdel",gdel,nnxz);
  
  mynegate(gvepsdel,gvepsdel,3*nnxz);

  int niter_max;
  get_param("niter_max",niter_max);

  string solver=get_s("solver");
  fprintf(stderr,"use solver %s",solver.c_str());
 
  if(hesstype.compare("full")==0){
   HessianOpVEpsDel H(data,v,eps,del,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
   if(solver.compare("simple")==0) simpleSolver(&H,pvepsdel,gvepsdel,niter_max);
   else if(solver.compare("CG")==0) CG(&H,pvepsdel,gvepsdel,niter_max);
   else if(solver.compare("indefCG")==0){
       mynegate(gvepsdel,gvepsdel,3*nnxz);
        indefCG(&H,pvepsdel,gvepsdel,niter_max);
   }
   else if(solver.compare("lsqr")==0) lsqr(&H,pvepsdel,gvepsdel,niter_max);
   else fprintf(stderr,"please specify solver by solver=simple or solver=CG or solver=lsqr\n");
  }
  else if(hesstype.compare("GN")==0){
   GNHessianOpVEpsDel H(data,v,eps,del,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
   if(solver.compare("simple")==0) simpleSolver(&H,pvepsdel,gvepsdel,niter_max);
   else if(solver.compare("CG")==0) CG(&H,pvepsdel,gvepsdel,niter_max);
   else if(solver.compare("indefCG")==0){
       mynegate(gvepsdel,gvepsdel,3*nnxz);
        indefCG(&H,pvepsdel,gvepsdel,niter_max);
   }
   else if(solver.compare("lsqr")==0) lsqr(&H,pvepsdel,gvepsdel,niter_max);
   else fprintf(stderr,"please specify solver by solver=simple or solver=CG or solver=lsqr\n");
  }
  else fprintf(stderr,"please specify hessian type by hesstype=full or hesstype=GN\n");
 
  write("pv",pv,nnxz);
  to_header("pv","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("pv","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("peps",peps,nnxz);
  to_header("peps","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("peps","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("pdel",pdel,nnxz);
  to_header("pdel","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("pdel","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []vepsdel;
  delete []pvepsdel;
  delete []gvepsdel;
 }
 else if(parameter.compare("vhepsdel")==0){
  fprintf(stderr,"parameter vh eps del\n");
  float *vh=new float[nnxz]();
  float *eps=new float[nnxz]();
  float *del=new float[nnxz]();
  
  float *dvh=new float[nnxz]();
  float *deps=new float[nnxz]();
  float *ddel=new float[nnxz]();
  
  if(padded==0){
   init_model("vh",vh,nx,nz,npad); 
   init_model("eps",eps,nx,nz,npad); 
   init_model("del",del,nx,nz,npad); 
   init_model("dvh",dvh,nx,nz,npad); 
   init_model("deps",deps,nx,nz,npad); 
   init_model("ddel",ddel,nx,nz,npad); 
  } 
  else{ 
   read("vh",vh,nnxz);
   read("eps",eps,nnxz);
   read("del",del,nnxz);
   read("dvh",dvh,nnxz);
   read("deps",deps,nnxz);
   read("ddel",ddel,nnxz);
  }
 
  float vh0,eps0,del0;
  get_param("vh0",vh0,"eps0",eps0,"del0",del0);
 
  scale(vh,vh,1./vh0,nnxz);
  scale(eps,eps,1./eps0,nnxz);
  scale(del,del,1./del0,nnxz);
  
  scale(dvh,dvh,1./vh0,nnxz);
  scale(deps,deps,1./eps0,nnxz);
  scale(ddel,ddel,1./del0,nnxz);
  
  float *gvh=new float[nnxz]();
  float *geps=new float[nnxz]();
  float *gdel=new float[nnxz]();
  
  if(hesstype.compare("full")==0) hessianVhEpsDel(gvh,geps,gdel,data,vh,eps,del,dvh,deps,ddel,vh0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  else if(hesstype.compare("GN")==0) GNhessianVhEpsDel(gvh,geps,gdel,data,vh,eps,del,dvh,deps,ddel,vh0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  else fprintf(stderr,"please specify hessian type by hesstype=full or hesstype=GN\n");
 
  write("gvh",gvh,nnxz);
  to_header("gvh","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gvh","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("geps",geps,nnxz);
  to_header("geps","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("geps","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gdel",gdel,nnxz);
  to_header("gdel","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gdel","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []vh; delete []eps; delete []del;
  delete []dvh; delete []deps; delete []ddel;
  delete []gvh; delete []geps; delete []gdel;
 }
 else if(parameter.compare("cij")==0){
  fprintf(stderr,"parameter c11 c13 c33\n");
  float *c11c13c33=new float[3*nnxz]();
  float *c11=c11c13c33,*c13=c11c13c33+nnxz,*c33=c11c13c33+2*nnxz;
  float *gc11c13c33=new float[3*nnxz]();
  float *gc11=gc11c13c33,*gc13=gc11c13c33+nnxz,*gc33=gc11c13c33+2*nnxz;
 
  if(padded==0){
   init_model("c11",c11,nx,nz,npad); 
   init_model("c13",c13,nx,nz,npad); 
   init_model("c33",c33,nx,nz,npad); 
  } 
  else{ 
   read("c11",c11,nnxz);
   read("c13",c13,nnxz);
   read("c33",c33,nnxz);
  }
  
  float c110,c130,c330;
  get_param("c110",c110,"c130",c130,"c330",c330);
 
  scale(c11,c11,1./c110,nnxz);
  scale(c13,c13,1./c130,nnxz);
  scale(c33,c33,1./c330,nnxz);

  read("gc11",gc11,nnxz);
  read("gc13",gc13,nnxz);
  read("gc33",gc33,nnxz);
  
  float *pc11c13c33=new float[3*nnxz]();
  float *pc11=pc11c13c33,*pc13=pc11c13c33+nnxz,*pc33=pc11c13c33+2*nnxz;
 
  mynegate(gc11c13c33,gc11c13c33,3*nnxz);

  int niter_max;
  get_param("niter_max",niter_max);

  string solver=get_s("solver");
  fprintf(stderr,"use solver %s",solver.c_str());
 
  if(hesstype.compare("full")==0){
   HessianOpCij H(data,c11,c13,c33,c110,c130,c330,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
   if(solver.compare("simple")==0) simpleSolver(&H,pc11c13c33,gc11c13c33,niter_max);
   else if(solver.compare("CG")==0) CG(&H,pc11c13c33,gc11c13c33,niter_max);
   else if(solver.compare("indefCG")==0){
       mynegate(gc11c13c33,gc11c13c33,3*nnxz);
        indefCG(&H,pc11c13c33,gc11c13c33,niter_max);
   }
   else if(solver.compare("lsqr")==0) lsqr(&H,pc11c13c33,gc11c13c33,niter_max);
   else fprintf(stderr,"please specify solver by solver=simple or solver=CG or solver=lsqr\n");
  }
  else if(hesstype.compare("GN")==0){
   GNHessianOpCij H(data,c11,c13,c33,c110,c130,c330,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
   if(solver.compare("simple")==0) simpleSolver(&H,pc11c13c33,gc11c13c33,niter_max);
   else if(solver.compare("CG")==0) CG(&H,pc11c13c33,gc11c13c33,niter_max);
   else if(solver.compare("indefCG")==0){
       mynegate(gc11c13c33,gc11c13c33,3*nnxz);
        indefCG(&H,pc11c13c33,gc11c13c33,niter_max);
   }
   else if(solver.compare("lsqr")==0) lsqr(&H,pc11c13c33,gc11c13c33,niter_max);
   else fprintf(stderr,"please specify solver by solver=simple or solver=CG or solver=lsqr\n");
  }
  else fprintf(stderr,"please specify hessian type by hesstype=full or hesstype=GN\n");

  write("pc11",pc11,nnxz);
  to_header("pc11","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("pc11","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("pc13",pc13,nnxz);
  to_header("pc13","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("pc13","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("pc33",pc33,nnxz);
  to_header("pc33","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("pc33","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []c11c13c33;
  delete []pc11c13c33;
  delete []gc11c13c33;
 }
 else{
     fprintf(stderr,"please specify parameterization by parameter=something in commandline where something is one of vepsdel, vhepsdel, vnetadel,vhepseta, vvhdel, vnvhdel, cij, or vvnvh\n");
 }
 delete []wavelet;delete []data;delete []sloc;delete []rloc;delete []taper;
 delete []m;
 delete []souloc;delete []recloc;

 myio_close();
 return 0;
}
