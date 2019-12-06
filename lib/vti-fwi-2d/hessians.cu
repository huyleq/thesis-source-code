#include <omp.h>
#include <cstdio>
#include "mylib.h"
#include "wave.h"
#include "check.h"
#include "conversions.h"
#include "kernels.h"

void hessianCij(float *gc11,float *gc13,float *gc33,const float *d0,float *c11,float *c13,float *c33,const float *dc11,const float *dc13,const float *dc33,float c110,float c130,float c330,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkCij(c11,c13,c33,c110,c130,c330,nnxz,m);
 float *tc11=new float[nnxz](); scale(tc11,c11,c110,nnxz);
 float *tc13=new float[nnxz](); scale(tc13,c13,c130,nnxz);
 float *tc33=new float[nnxz](); scale(tc33,c33,c330,nnxz);
 float *tdc11=new float[nnxz](); scale(tdc11,dc11,c110,nnxz);
 float *tdc13=new float[nnxz](); scale(tdc13,dc13,c130,nnxz);
 float *tdc33=new float[nnxz](); scale(tdc33,dc33,c330,nnxz);
 hessian_f(gc11,gc13,gc33,d0,tc11,tc13,tc33,tdc11,tdc13,tdc33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 int n=npad+wbottom/dz+1;
 memset(gc11,0,n*nnx*sizeof(float)); scale(gc11,gc11,c110,nnxz);
 memset(gc13,0,n*nnx*sizeof(float)); scale(gc13,gc13,c130,nnxz);
 memset(gc33,0,n*nnx*sizeof(float)); scale(gc33,gc33,c330,nnxz);
 delete []tc11;delete []tc13;delete []tc33;
 delete []tdc11;delete []tdc13;delete []tdc33;
 return; 
}

void GNhessianCij(float *gc11,float *gc13,float *gc33,const float *d0,float *c11,float *c13,float *c33,const float *dc11,const float *dc13,const float *dc33,float c110,float c130,float c330,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkCij(c11,c13,c33,c110,c130,c330,nnxz,m);
 float *tc11=new float[nnxz](); scale(tc11,c11,c110,nnxz);
 float *tc13=new float[nnxz](); scale(tc13,c13,c130,nnxz);
 float *tc33=new float[nnxz](); scale(tc33,c33,c330,nnxz);
 float *tdc11=new float[nnxz](); scale(tdc11,dc11,c110,nnxz);
 float *tdc13=new float[nnxz](); scale(tdc13,dc13,c130,nnxz);
 float *tdc33=new float[nnxz](); scale(tdc33,dc33,c330,nnxz);
 GNhessian_f(gc11,gc13,gc33,d0,tc11,tc13,tc33,tdc11,tdc13,tdc33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 int n=npad+wbottom/dz+1;
 memset(gc11,0,n*nnx*sizeof(float)); scale(gc11,gc11,c110,nnxz);
 memset(gc13,0,n*nnx*sizeof(float)); scale(gc13,gc13,c130,nnxz);
 memset(gc33,0,n*nnx*sizeof(float)); scale(gc33,gc33,c330,nnxz);
 delete []tc11;delete []tc13;delete []tc33;
 delete []tdc11;delete []tdc13;delete []tdc33;
 return; 
}

void hessianVEpsDel(float *gv,float *geps,float *gdel,const float *d0,float *v,float *eps,float *del,const float *dv,const float *deps,const float *ddel,float v0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkEpsDel(eps,del,eps0,del0,nnxz,m);
 float *c11=new float[nnxz]();
 float *c13=new float[nnxz]();
 float *c33=new float[nnxz]();
 float *dc11=new float[nnxz]();
 float *dc13=new float[nnxz]();
 float *dc33=new float[nnxz]();
 float *gc11=new float[nnxz]();
 float *gc13=new float[nnxz]();
 float *gc33=new float[nnxz]();
 dVEpsDel2dCij(c11,c13,c33,dc11,dc13,dc33,v,eps,del,dv,deps,ddel,v0,eps0,del0,nnxz);
 hessian_f(gc11,gc13,gc33,d0,c11,c13,c33,dc11,dc13,dc33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 GradCij2GradVEpsDel(gv,geps,gdel,gc11,gc13,gc33,v,eps,del,v0,eps0,del0,nnxz);
 int n=npad+wbottom/dz+1;
 memset(gv,0,n*nnx*sizeof(float));
 memset(geps,0,n*nnx*sizeof(float));
 memset(gdel,0,n*nnx*sizeof(float));
 delete []c11;delete []c13;delete []c33;
 delete []dc11;delete []dc13;delete []dc33;
 delete []gc11;delete []gc13;delete []gc33;
 return; 
}

void GNhessianVEpsDel(float *gv,float *geps,float *gdel,const float *d0,float *v,float *eps,float *del,const float *dv,const float *deps,const float *ddel,float v0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkEpsDel(eps,del,eps0,del0,nnxz,m);
 float *c11=new float[nnxz]();
 float *c13=new float[nnxz]();
 float *c33=new float[nnxz]();
 float *dc11=new float[nnxz]();
 float *dc13=new float[nnxz]();
 float *dc33=new float[nnxz]();
 float *gc11=new float[nnxz]();
 float *gc13=new float[nnxz]();
 float *gc33=new float[nnxz]();
 dVEpsDel2dCij(c11,c13,c33,dc11,dc13,dc33,v,eps,del,dv,deps,ddel,v0,eps0,del0,nnxz);
 GNhessian_f(gc11,gc13,gc33,d0,c11,c13,c33,dc11,dc13,dc33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 GradCij2GradVEpsDel(gv,geps,gdel,gc11,gc13,gc33,v,eps,del,v0,eps0,del0,nnxz);
 int n=npad+wbottom/dz+1;
 memset(gv,0,n*nnx*sizeof(float));
 memset(geps,0,n*nnx*sizeof(float));
 memset(gdel,0,n*nnx*sizeof(float));
 delete []c11;delete []c13;delete []c33;
 delete []dc11;delete []dc13;delete []dc33;
 delete []gc11;delete []gc13;delete []gc33;
 return; 
}

void hessianVhEpsDel(float *gvh,float *geps,float *gdel,const float *d0,float *vh,float *eps,float *del,const float *dvh,const float *deps,const float *ddel,float vh0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkEpsDel(eps,del,eps0,del0,nnxz,m);
 float *c11=new float[nnxz]();
 float *c13=new float[nnxz]();
 float *c33=new float[nnxz]();
 float *dc11=new float[nnxz]();
 float *dc13=new float[nnxz]();
 float *dc33=new float[nnxz]();
 float *gc11=new float[nnxz]();
 float *gc13=new float[nnxz]();
 float *gc33=new float[nnxz]();
 dVhEpsDel2dCij(c11,c13,c33,dc11,dc13,dc33,vh,eps,del,dvh,deps,ddel,vh0,eps0,del0,nnxz);
 hessian_f(gc11,gc13,gc33,d0,c11,c13,c33,dc11,dc13,dc33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 GradCij2GradVhEpsDel(gvh,geps,gdel,gc11,gc13,gc33,vh,eps,del,vh0,eps0,del0,nnxz);
 int n=npad+wbottom/dz+1;
 memset(gvh,0,n*nnx*sizeof(float));
 memset(geps,0,n*nnx*sizeof(float));
 memset(gdel,0,n*nnx*sizeof(float));
 delete []c11;delete []c13;delete []c33;
 delete []dc11;delete []dc13;delete []dc33;
 delete []gc11;delete []gc13;delete []gc33;
 return; 
}

void GNhessianVhEpsDel(float *gvh,float *geps,float *gdel,const float *d0,float *vh,float *eps,float *del,const float *dvh,const float *deps,const float *ddel,float vh0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m){
 int nnx=nx+2*npad,nnz=nz+2*npad,nnxz=nnx*nnz;
 checkEpsDel(eps,del,eps0,del0,nnxz,m);
 float *c11=new float[nnxz]();
 float *c13=new float[nnxz]();
 float *c33=new float[nnxz]();
 float *dc11=new float[nnxz]();
 float *dc13=new float[nnxz]();
 float *dc33=new float[nnxz]();
 float *gc11=new float[nnxz]();
 float *gc13=new float[nnxz]();
 float *gc33=new float[nnxz]();
 dVhEpsDel2dCij(c11,c13,c33,dc11,dc13,dc33,vh,eps,del,dvh,deps,ddel,vh0,eps0,del0,nnxz);
 GNhessian_f(gc11,gc13,gc33,d0,c11,c13,c33,dc11,dc13,dc33,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
 GradCij2GradVhEpsDel(gvh,geps,gdel,gc11,gc13,gc33,vh,eps,del,vh0,eps0,del0,nnxz);
 int n=npad+wbottom/dz+1;
 memset(gvh,0,n*nnx*sizeof(float));
 memset(geps,0,n*nnx*sizeof(float));
 memset(gdel,0,n*nnx*sizeof(float));
 delete []c11;delete []c13;delete []c33;
 delete []dc11;delete []dc13;delete []dc33;
 delete []gc11;delete []gc13;delete []gc33;
 return; 
}

