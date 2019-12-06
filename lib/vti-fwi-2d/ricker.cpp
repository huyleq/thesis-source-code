#include <cmath>
#include <cstdlib>

#include "myio.h"
#include "mylib.h"

using namespace std;

const double pi=4.*atan(1.);

void ricker(float *wavelet,float freq,int nt,float dt,float ot,float tdelay,float scalefactor){
 float pi2f2=pi*pi*freq*freq;
 #pragma omp parallel for num_threads(8) 
 for(int it=0;it<nt;++it){
  float t=it*dt+ot-tdelay;
  float pi2f2t2=pi2f2*t*t;
  wavelet[it]=(1.-2.*pi2f2t2)*exp(-pi2f2t2);
  wavelet[it]*=scalefactor;
 }
 return;
}

void ricker1(float *wavelet,float freq,int nt,float dt,float ot,float tdelay,float scalefactor){
 float pi2f2=pi*pi*freq*freq;
 #pragma omp parallel for num_threads(8) 
 for(int it=0;it<nt;++it){
  float t=it*dt+ot-tdelay;
  float pi2f2t2=pi2f2*t*t;
  wavelet[it]=-2.f*pi2f2*t*(3.f-2.*pi2f2t2)*exp(-pi2f2t2);
 }
 float m=max(wavelet,nt);
 #pragma omp parallel for num_threads(8) 
 for(int it=0;it<nt;it++) wavelet[it]*=scalefactor/m;
 return;
}

int main(int argc,char ** argv){
 myio_init(argc,argv);
 float ot,dt;
 int nt;
 get_param("nt",nt,"ot",ot,"dt",dt);
 float freq,tdelay,scalefactor; 
 get_param("freq",freq,"tdelay",tdelay,"scalefactor",scalefactor);
 float *wavelet=new float[nt]();
 int opt=0;
 get_param("opt",opt);
 if(opt==0) ricker(wavelet,freq,nt,dt,ot,tdelay,scalefactor);
 else if(opt==1) ricker1(wavelet,freq,nt,dt,ot,tdelay,scalefactor);
 write("wavelet",wavelet,nt);
 to_header("wavelet","n1",nt,"o1",ot,"d1",dt);
 delete []wavelet;
 myio_close();
 return 0;
}
