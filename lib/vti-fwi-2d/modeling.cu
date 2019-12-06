#include <cmath>
#include <omp.h>
#include <cstdio>
#include <iostream>

#include "myio.h"
#include "mylib.h"
#include "init.h"
#include "wave.h"
#include "conversions.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);

 int nnx,nnz,nx,nz,nt,npad;
 float ox,oz,ot,dx,dz,dt;
 init2d(nnx,nnz,nx,nz,nt,dx,dz,dt,ox,oz,ot,npad);
 int nnxz=nnx*nnz;

 float *wavelet=new float[nt];
 read("wavelet",wavelet,nt);
 
 float rate;
 get_param("rate",rate);
 int ratio=rate/dt+0.5;
 cout<<rate<<" "<<ratio<<" "<<dt<<endl;
 int ntNeg=std::round(abs(ot)/dt);
 int nnt_data=(nt-ntNeg-1)/ratio+1;

 float *wavefield=new float[nnxz*nnt_data]();
 
 float *taper=new float[nnxz]();
 init_abc(taper,nx,nz,npad);
 
 float sx; get_param("sx",sx);
 float sz; get_param("sz",sz);
 int slocxz=(sx/dx+npad+1)+(sz/dz+npad+1)*nnx;

 float *v=new float[nnxz]();
 init_model("v",v,nx,nz,npad); 
 float *eps=new float[nnxz]();
 init_model("eps",eps,nx,nz,npad); 
 float *del=new float[nnxz]();
 init_model("del",del,nx,nz,npad); 
 
 string system=get_s("system");

 if(system.compare("RDR")==0){
  float *r11=new float[nnxz]();
  float *r13=new float[nnxz]();
  float *r33=new float[nnxz]();
  VEpsDel2R(r11,r13,r33,v,eps,del,nnxz);
  modelingR_f(wavefield,r11,r13,r33,wavelet,slocxz,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
  delete []r11;delete []r13;delete []r33;
 }
 else if(system.compare("ABCD")==0){
  float *a1=new float[nnxz]();
  float *b1c1=new float[nnxz]();
  float *d1=new float[nnxz]();
  float *a2=new float[nnxz]();
  float *b2c2=new float[nnxz]();
  float *d2=new float[nnxz]();
  VEpsDel2ABCD(a1,b1c1,d1,a2,b2c2,d2,v,eps,del,nnxz);
  modelingABCD_f(wavefield,a1,b1c1,d1,a2,b2c2,d2,wavelet,slocxz,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
  delete []a1;delete []b1c1;delete []d1;
  delete []a2;delete []b2c2;delete []d2;
 }
 else{
  float *c11=new float[nnxz]();
  float *c13=new float[nnxz]();
  float *c33=new float[nnxz]();
  VEpsDel2Cij(c11,c13,c33,v,eps,del,1.,1.,1.,nnxz);
  modeling_f(wavefield,c11,c13,c33,wavelet,slocxz,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot);
  delete []c11;delete []c13;delete []c33;
 }

 delete []v;delete []eps;delete []del;

 write("wavefield",wavefield,nnxz*nnt_data);
 to_header("wavefield","n1",nnx,"o1",-dx*npad,"d1",dx);
 to_header("wavefield","n2",nnz,"o2",-dz*npad,"d2",dz);
 to_header("wavefield","n3",nnt_data,"o3",0.,"d3",rate);

 delete []wavelet;
 delete []wavefield;
 delete []taper;

 myio_close();
 return 0;
}
