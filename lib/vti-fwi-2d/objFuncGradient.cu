#include <omp.h>
#include <cstdio>

#include "myio.h"
#include "mylib.h"
#include "wave.h"
#include "check.h"
#include "conversions.h"
#include "kernels.h"

const double pi=4.f*atan(1.f);

float objFuncGradientCij(float *gc11,float *gc13,float *gc33,const float *d0,float *c11,float *c13,float *c33,float c110,float c130,float c330,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkCij(c11,c13,c33,c110,c130,c330,nnxz,m);
 float *tc11=new float[nnxz](); scale(tc11,c11,c110,nnxz);
 float *tc13=new float[nnxz](); scale(tc13,c13,c130,nnxz);
 float *tc33=new float[nnxz](); scale(tc33,c33,c330,nnxz);
 double val=objFuncGradient_f(gc11,gc13,gc33,d0,tc11,tc13,tc33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 int n=npad+wbottom/dz+1;
 memset(gc11,0,n*nnx*sizeof(float)); scale(gc11,gc11,c110,nnxz);
 memset(gc13,0,n*nnx*sizeof(float)); scale(gc13,gc13,c130,nnxz);
 memset(gc33,0,n*nnx*sizeof(float)); scale(gc33,gc33,c330,nnxz);
 delete []tc11;delete []tc13;delete []tc33;
 return val; 
}

float objFuncGradientVEpsDel(float *gv,float *geps,float *gdel,const float *d0,float *v,float *eps,float *del,float v0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkEpsDel(eps,del,eps0,del0,nnxz,m);

 float *c11=new float[nnxz]();
 float *c13=new float[nnxz]();
 float *c33=new float[nnxz]();
 float *gc11=new float[nnxz]();
 float *gc13=new float[nnxz]();
 float *gc33=new float[nnxz]();
 
 VEpsDel2Cij(c11,c13,c33,v,eps,del,v0,eps0,del0,nnxz);
 
 double val=objFuncGradient_f(gc11,gc13,gc33,d0,c11,c13,c33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 
 GradCij2GradVEpsDel(gv,geps,gdel,gc11,gc13,gc33,v,eps,del,v0,eps0,del0,nnxz);
 
 int nwbottom=wbottom/dz+1;
 int n=npad+nwbottom;
 memset(gv,0,n*nnx*sizeof(float));
 memset(geps,0,n*nnx*sizeof(float));
 memset(gdel,0,n*nnx*sizeof(float));
 
 delete []c11;delete []c13;delete []c33;
 delete []gc11;delete []gc13;delete []gc33;
 return val; 
}

float objFuncGradientVnEtaDel(float *gvn,float *geta,float *gdel,const float *d0,float *vn,float *eta,float *del,float vn0,float eta0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkEta(eta,nnxz,m);

 float *c11=new float[nnxz]();
 float *c13=new float[nnxz]();
 float *c33=new float[nnxz]();
 float *gc11=new float[nnxz]();
 float *gc13=new float[nnxz]();
 float *gc33=new float[nnxz]();
 
 VnEtaDel2Cij(c11,c13,c33,vn,eta,del,vn0,eta0,del0,nnxz);
 
 double val=objFuncGradient_f(gc11,gc13,gc33,d0,c11,c13,c33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 
 GradCij2GradVnEtaDel(gvn,geta,gdel,gc11,gc13,gc33,vn,eta,del,vn0,eta0,del0,nnxz);
 
 int nwbottom=wbottom/dz+1;
 int n=npad+nwbottom;
 memset(gvn,0,n*nnx*sizeof(float));
 memset(geta,0,n*nnx*sizeof(float));
 memset(gdel,0,n*nnx*sizeof(float));
 
 delete []c11;delete []c13;delete []c33;
 delete []gc11;delete []gc13;delete []gc33;
 return val; 
}

float objFuncGradientVhEpsEta(float *gvh,float *geps,float *geta,const float *d0,float *vh,float *eps,float *eta,float vh0,float eps0,float eta0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkEta(eta,nnxz,m);

 float *c11=new float[nnxz]();
 float *c13=new float[nnxz]();
 float *c33=new float[nnxz]();
 float *gc11=new float[nnxz]();
 float *gc13=new float[nnxz]();
 float *gc33=new float[nnxz]();
 
 VhEpsEta2Cij(c11,c13,c33,vh,eps,eta,vh0,eps0,eta0,nnxz);
 
 double val=objFuncGradient_f(gc11,gc13,gc33,d0,c11,c13,c33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 
 GradCij2GradVhEpsEta(gvh,geps,geta,gc11,gc13,gc33,vh,eps,eta,vh0,eps0,eta0,nnxz);
 
 int nwbottom=wbottom/dz+1;
 int n=npad+nwbottom;
 memset(gvh,0,n*nnx*sizeof(float));
 memset(geps,0,n*nnx*sizeof(float));
 memset(geta,0,n*nnx*sizeof(float));
 
 delete []c11;delete []c13;delete []c33;
 delete []gc11;delete []gc13;delete []gc33;
 return val; 
}

float objFuncGradientVVhDel(float *gv,float *gvh,float *gdel,const float *d0,float *v,float *vh,float *del,float v0,float vh0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkVVhDel(v,vh,del,v0,vh0,del0,nnxz,m);

 float *c11=new float[nnxz]();
 float *c13=new float[nnxz]();
 float *c33=new float[nnxz]();
 float *gc11=new float[nnxz]();
 float *gc13=new float[nnxz]();
 float *gc33=new float[nnxz]();
 
 VVhDel2Cij(c11,c13,c33,v,vh,del,v0,vh0,del0,nnxz);
 
 double val=objFuncGradient_f(gc11,gc13,gc33,d0,c11,c13,c33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 
 GradCij2GradVVhDel(gv,gvh,gdel,gc11,gc13,gc33,v,vh,del,v0,vh0,del0,nnxz);
 
 int nwbottom=wbottom/dz+1;
 int n=npad+nwbottom;
 memset(gv,0,n*nnx*sizeof(float));
 memset(gvh,0,n*nnx*sizeof(float));
 memset(gdel,0,n*nnx*sizeof(float));
 
 delete []c11;delete []c13;delete []c33;
 delete []gc11;delete []gc13;delete []gc33;
 return val; 
}

float objFuncGradientVnVhDel(float *gvn,float *gvh,float *gdel,const float *d0,float *vn,float *vh,float *del,float vn0,float vh0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkVnVh(vn,vh,vn0,vh0,nnxz,m);

 float *c11=new float[nnxz]();
 float *c13=new float[nnxz]();
 float *c33=new float[nnxz]();
 float *gc11=new float[nnxz]();
 float *gc13=new float[nnxz]();
 float *gc33=new float[nnxz]();
 
 VnVhDel2Cij(c11,c13,c33,vn,vh,del,vn0,vh0,del0,nnxz);
 
 double val=objFuncGradient_f(gc11,gc13,gc33,d0,c11,c13,c33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 
 GradCij2GradVnVhDel(gvn,gvh,gdel,gc11,gc13,gc33,vn,vh,del,vn0,vh0,del0,nnxz);
 
 int nwbottom=wbottom/dz+1;
 int n=npad+nwbottom;
 memset(gvn,0,n*nnx*sizeof(float));
 memset(gvh,0,n*nnx*sizeof(float));
 memset(gdel,0,n*nnx*sizeof(float));
 
 delete []c11;delete []c13;delete []c33;
 delete []gc11;delete []gc13;delete []gc33;
 return val; 
}

float objFuncGradientVhEpsDel(float *gvh,float *geps,float *gdel,const float *d0,float *vh,float *eps,float *del,float vh0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkEpsDel(eps,del,eps0,del0,nnxz,m);

 float *c11=new float[nnxz]();
 float *c13=new float[nnxz]();
 float *c33=new float[nnxz]();
 float *gc11=new float[nnxz]();
 float *gc13=new float[nnxz]();
 float *gc33=new float[nnxz]();
 
 VhEpsDel2Cij(c11,c13,c33,vh,eps,del,vh0,eps0,del0,nnxz);
 
 double val=objFuncGradient_f(gc11,gc13,gc33,d0,c11,c13,c33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 
 GradCij2GradVhEpsDel(gvh,geps,gdel,gc11,gc13,gc33,vh,eps,del,vh0,eps0,del0,nnxz);
 
 int nwbottom=wbottom/dz+1;
 int n=npad+nwbottom;
 memset(gvh,0,n*nnx*sizeof(float));
 memset(geps,0,n*nnx*sizeof(float));
 memset(gdel,0,n*nnx*sizeof(float));
 
 delete []c11;delete []c13;delete []c33;
 delete []gc11;delete []gc13;delete []gc33;
 return val; 
}

float objFuncGradientVVnVh(float *gv,float *gvn,float *gvh,const float *d0,float *v,float *vn,float *vh,float v0,float vn0,float vh0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkVnVh(vn,vh,vn0,vh0,nnxz,m);

 float *c11=new float[nnxz]();
 float *c13=new float[nnxz]();
 float *c33=new float[nnxz]();
 float *gc11=new float[nnxz]();
 float *gc13=new float[nnxz]();
 float *gc33=new float[nnxz]();
 
 VVnVh2Cij(c11,c13,c33,v,vn,vh,v0,vn0,vh0,nnxz);
 
 double val=objFuncGradient_f(gc11,gc13,gc33,d0,c11,c13,c33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 
 GradCij2GradVVnVh(gv,gvn,gvh,gc11,gc13,gc33,v,vn,vh,v0,vn0,vh0,nnxz);
 
 int nwbottom=wbottom/dz+1;
 int n=npad+nwbottom;
 memset(gv,0,n*nnx*sizeof(float));
 memset(gvn,0,n*nnx*sizeof(float));
 memset(gvh,0,n*nnx*sizeof(float));
 
 delete []c11;delete []c13;delete []c33;
 delete []gc11;delete []gc13;delete []gc33;
 return val; 
}
