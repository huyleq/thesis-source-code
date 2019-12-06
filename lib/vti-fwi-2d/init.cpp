#include <cmath>
#include <omp.h>
#include <string>
#include <iostream>
#include "init.h"
#include "myio.h"
#include "mylib.h"

using namespace std;

void init2d(int &nnx,int &nnz,int &nx,int &nz,int &nt,float &dx,float &dz,float &dt,float &ox,float &oz,float &ot,int &npad){
 get_param("nx",nx,"dx",dx,"ox",ox);
 get_param("nz",nz,"dz",dz,"oz",oz);
 get_param("nt",nt,"dt",dt,"ot",ot);
 get_param("npad",npad);
 nnx=nx+2*npad;
 nnz=nz+2*npad;
 return;
}

void init_rec(int &n_rec,float &d_rec,float &o_rec,float &z_rec){ 
 get_param("n_rec",n_rec,"d_rec",d_rec,"o_rec",o_rec);
 get_param("z_rec",z_rec);
 return;
}

void init_rec_loc(int *rec_loc,int n_rec,float d_rec,float o_rec,float z_rec,float dx,float dz,float ox,float oz,int npad){ 
 #pragma omp parallel for num_threads(8) 
 for(int i_rec=0;i_rec<n_rec;++i_rec){
  rec_loc[0+i_rec*2]=(o_rec+i_rec*d_rec-ox)/dx+0.5+npad;
  rec_loc[1+i_rec*2]=(z_rec-oz)/dz+0.5+npad;
 }
 return;
}

void init_shot(int &n_shot,float &d_shot,float &o_shot,float &z_shot){ 
 get_param("n_shot",n_shot,"d_shot",d_shot,"o_shot",o_shot);
 get_param("z_shot",z_shot);
 return;
}

void init_shot_loc(int *shot_loc,int n_shot,float d_shot,float o_shot,float z_shot,float dx,float dz,float ox,float oz,int npad){ 
 #pragma omp parallel for num_threads(8)
 for(int i_shot=0;i_shot<n_shot;++i_shot){
  shot_loc[0+i_shot*2]=(o_shot+i_shot*d_shot-ox)/dx+0.5+npad;
  shot_loc[1+i_shot*2]=(z_shot-oz)/dz+0.5+npad;
 }
 return;
}

void pad(float *m,int nx,int nz,int npad){
 for(int iz=0;iz<npad;++iz){
  equate(m+npad+iz*(nx+2*npad),m+npad+npad*(nx+2*npad),nx);
  equate(m+npad+(iz+nz+npad)*(nx+2*npad),m+npad+(npad+nz-1)*(nx+2*npad),nx);
 }
 for(int iz=0;iz<nz+2*npad;++iz){
  set(m+iz*(nx+2*npad),m[npad+iz*(nx+2*npad)],npad);
  set(m+nx+npad+iz*(nx+2*npad),m[nx+npad-1+iz*(nx+2*npad)],npad);
 }
 return;
}

void init_model(const string &s,float *m,int nx,int nz,int npad){
 #pragma omp parallel for num_threads(8) 
 for(int iz=0;iz<nz;++iz) read(s,m+npad+(iz+npad)*(nx+2*npad),nx,iz*nx); 
 pad(m,nx,nz,npad);
 return;
}

void init_abc(float *taper,int nx,int nz,int npad){
 float *taper0=new float[npad]();
 #pragma omp parallel for num_threads(8) 
 for(int i=0;i<npad;++i){
  taper0[i]=DAMPER+(1.-DAMPER)*cos(pi*(float)(npad-1-i)/npad);
 }
 int nnx=nx+2*npad;
 int nnz=nz+2*npad;
 set(taper,1.,nnx*nnz);
 for(int i=0;i<nnz;++i){
  multiply(taper+i*nnx,taper+i*nnx,taper0,npad);
  reverse_multiply(taper+i*nnx+npad+nx,taper+i*nnx+npad+nx,taper0,npad);
 }
 for(int i=0;i<npad;++i){
  scale(taper+i*nnx,taper+i*nnx,taper0[i],nnx);
  scale(taper+(i+npad+nz)*nnx,taper+(i+npad+nz)*nnx,taper0[npad-1-i],nnx);
 }
 delete []taper0;
 return;
}
