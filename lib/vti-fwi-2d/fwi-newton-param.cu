#include <cstdio>
#include "myio.h"
#include "mylib.h"
#include "init.h"
#include "wave.h"
#include "HessianOp.h"
#include "lbfgs.h"

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
 
 int nfg,niter_max; get_param("nfg",nfg,"niter_max",niter_max);

 int nn=3*nnxz,icall=0,iflag=0;
 float f;
 float *diag=new float[nn]();
 float *w=new float[2*nn]();
 int *isave=new int[nisave]();
 float *dsave=new float[ndsave]();
  
 string parameter=get_s("parameter");

 if(parameter.compare("vepsdel")==0){
  fprintf(stderr,"parameter v eps del\n");
  float *vepsdel=new float[3*nnxz]();
  float *v=vepsdel,*eps=vepsdel+nnxz,*del=vepsdel+2*nnxz;
//  float *eps=vepsdel,*del=vepsdel+nnxz,*v=vepsdel+2*nnxz;
 
  if(padded==0){
   init_model("v",v,nx,nz,npad); 
   init_model("eps",eps,nx,nz,npad); 
   init_model("del",del,nx,nz,npad); 
  } 
  else{ 
   read("v",v,nnxz);
   read("eps",eps,nnxz);
   read("del",del,nnxz);
  }
  
  float v0,eps0,del0;
  get_param("v0",v0,"eps0",eps0,"del0",del0);
 
  scale(v,v,1./v0,nnxz);
  scale(eps,eps,1./eps0,nnxz);
  scale(del,del,1./del0,nnxz);
 
  float *gvepsdel=new float[3*nnxz]();
  float *gv=gvepsdel,*geps=gvepsdel+nnxz,*gdel=gvepsdel+2*nnxz;
//  float *geps=gvepsdel,*gdel=gvepsdel+nnxz,*gv=gvepsdel+2*nnxz;
 
   HessianOpVEpsDel H(data,v,eps,del,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);

  to_header("iv","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("iv","n2",nnz,"o2",-dz*npad,"d2",dz);
  to_header("ieps","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("ieps","n2",nnz,"o2",-dz*npad,"d2",dz);
  to_header("idel","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("idel","n2",nnz,"o2",-dz*npad,"d2",dz);
  
  to_header("gv","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gv","n2",nnz,"o2",-dz*npad,"d2",dz);
  to_header("geps","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("geps","n2",nnz,"o2",-dz*npad,"d2",dz);
  to_header("gdel","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gdel","n2",nnz,"o2",-dz*npad,"d2",dz);
  
  to_header("check","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("check","n2",nnz,"o2",-dz*npad,"d2",dz);
  
  while(true){
   icall++;
   
   f=objFuncGradientVEpsDel(gv,geps,gdel,data,v,eps,del,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
   fprintf(stderr,"objfunc f=%.10f\n",f);
 
   write("objfunc",&f,1,"native_float",std::ios_base::app);
   to_header("objfunc","n1",icall,"o1",0.,"d1",1.);
   
   write("iv",v,nnxz,"native_float",std::ios_base::app);
   to_header("iv","n3",icall,"o3",0.,"d3",1.);
   
   write("gv",gv,nnxz,"native_float",std::ios_base::app);
   to_header("gv","n3",icall,"o3",0.,"d3",1.);
   
   write("ieps",eps,nnxz,"native_float",std::ios_base::app);
   to_header("ieps","n3",icall,"o3",0.,"d3",1.);
   
   write("geps",geps,nnxz,"native_float",std::ios_base::app);
   to_header("geps","n3",icall,"o3",0.,"d3",1.);
   
   write("idel",del,nnxz,"native_float",std::ios_base::app);
   to_header("idel","n3",icall,"o3",0.,"d3",1.);
   
   write("gdel",gdel,nnxz,"native_float",std::ios_base::app);
   to_header("gdel","n3",icall,"o3",0.,"d3",1.);
   
   write("check",m,nnxz,"native_float",std::ios_base::app);
   to_header("check","n3",icall,"o3",0.,"d3",1.);
   
   newton(nn,vepsdel,f,gvepsdel,diag,w,iflag,isave,dsave,&H,niter_max);
   
   if(iflag<=0 || isave[5]>=nfg){
    fprintf(stderr,"iflag %d\n",iflag);
    break;
   }
  }
  
  delete []vepsdel;delete []gvepsdel;
 }
 else if(parameter.compare("cij")==0){
  fprintf(stderr,"parameter c11 c13 c33\n");
  float *c11c13c33=new float[3*nnxz]();
  float *c11=c11c13c33,*c13=c11c13c33+nnxz,*c33=c11c13c33+2*nnxz;
 
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
 
  float *gc11c13c33=new float[3*nnxz]();
  float *gc11=gc11c13c33,*gc13=gc11c13c33+nnxz,*gc33=gc11c13c33+2*nnxz;
 
  HessianOpCij H(data,c11,c13,c33,c110,c130,c330,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);

  to_header("ic11","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("ic11","n2",nnz,"o2",-dz*npad,"d2",dz);
  to_header("ic13","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("ic13","n2",nnz,"o2",-dz*npad,"d2",dz);
  to_header("ic33","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("ic33","n2",nnz,"o2",-dz*npad,"d2",dz);
  
  to_header("gc11","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gc11","n2",nnz,"o2",-dz*npad,"d2",dz);
  to_header("gc13","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gc13","n2",nnz,"o2",-dz*npad,"d2",dz);
  to_header("gc33","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gc33","n2",nnz,"o2",-dz*npad,"d2",dz);
  
  to_header("check","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("check","n2",nnz,"o2",-dz*npad,"d2",dz);
  
  while(true){
   icall++;
   
   f=objFuncGradientCij(gc11,gc13,gc33,data,c11,c13,c33,c110,c130,c330,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
   fprintf(stderr,"objfunc f=%.10f\n",f);
 
   write("objfunc",&f,1,"native_float",std::ios_base::app);
   to_header("objfunc","n1",icall,"o1",0.,"d1",1.);
   
   write("ic11",c11,nnxz,"native_float",std::ios_base::app);
   to_header("ic11","n3",icall,"o3",0.,"d3",1.);
   
   write("gc11",gc11,nnxz,"native_float",std::ios_base::app);
   to_header("gc11","n3",icall,"o3",0.,"d3",1.);
   
   write("ic13",c13,nnxz,"native_float",std::ios_base::app);
   to_header("ic13","n3",icall,"o3",0.,"d3",1.);
   
   write("gc13",gc13,nnxz,"native_float",std::ios_base::app);
   to_header("gc13","n3",icall,"o3",0.,"d3",1.);
   
   write("ic33",c33,nnxz,"native_float",std::ios_base::app);
   to_header("ic33","n3",icall,"o3",0.,"d3",1.);
   
   write("gc33",gc33,nnxz,"native_float",std::ios_base::app);
   to_header("gc33","n3",icall,"o3",0.,"d3",1.);
   
   write("check",m,nnxz,"native_float",std::ios_base::app);
   to_header("check","n3",icall,"o3",0.,"d3",1.);
   
   newton(nn,c11c13c33,f,gc11c13c33,diag,w,iflag,isave,dsave,&H,niter_max);
   
   if(iflag<=0 || isave[5]>=nfg){
    fprintf(stderr,"iflag %d\n",iflag);
    break;
   }
  }
  
  delete []c11c13c33;delete []gc11c13c33;
 }
 else{
     fprintf(stderr,"please specify parameterization by parameter=something in commandline where something is one of vepsdel, vhepsdel, vnetadel,vhepseta, vvhdel, vnvhdel, cij, or vvnvh\n");
 }

 delete []diag;delete []w;
 delete []isave;delete []dsave;
 
 delete []wavelet;delete []data;delete []sloc;delete []rloc;delete []taper;
 delete []souloc; delete []recloc;

 myio_close();
 return 0;
}
