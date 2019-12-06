#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <cmath>
#include "conversions.h"

void VEpsDel2ABCD(float *a1,float *b1c1,float *d1,float *a2,float *b2c2,float *d2,const float *v,const float *eps,const float *del,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float a=sqrt(1+2.f*eps[i]),b=sqrt(1+2.f*del[i]);
  float r=b/a;
  float z=sqrt(1-r*r);
//  fprintf(stderr,"i %d %.10f %.10f\n",i,r,z);
  float vpx=v[i]*a;
  float vpx2=vpx*vpx;
  float v2=v[i]*v[i];
  b1c1[i]=0.5*r*vpx2;
  b2c2[i]=0.5*r*v2;
  a1[i]=0.5*vpx2*(1+z);
  d1[i]=0.5*vpx2*(1-z);
  a2[i]=0.5*v2*(1-z);
  d2[i]=0.5*v2*(1+z);
 }
 return;
}

void VEpsDel2R(float *r11,float *r13,float *r33,const float *v,const float *eps,const float *del,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float tau=2.*(eps[i]+1.);
  float delta=2.*(eps[i]-del[i]);
  float s=sqrt(delta);
  float t=sqrt(tau+2.*s);
  float vt=v[i]/t;
  r11[i]=vt*(1.+2.*eps[i]+s);
  r13[i]=vt*sqrt(1.+2.*del[i]);
  r33[i]=vt*(1.+s);
 }
 return;
}

void VEpsDel2Cij(float *c11,float *c13,float *c33,const float *v,const float *eps,const float *del,float v0,float eps0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  c33[i]=v0*v0*v[i]*v[i]; 
  c11[i]=c33[i]*(1.+2*eps0*eps[i]); 
  c13[i]=c33[i]*sqrt(1.+2*del0*del[i]); 
 }
 return;
}

void dVEpsDel2dCij(float *c11,float *c13,float *c33,float *dc11,float *dc13,float *dc33,const float *v,const float *eps,const float *del,const float *dv,const float *deps,const float *ddel,float v0,float eps0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float vv=v0*v[i],ee=eps0*eps[i],dd=del0*del[i];
  float vv2=vv*vv;
  ee=1.+2*ee;
  dd=sqrt(1.+2*dd);
  c11[i]=vv2*ee; 
  c13[i]=vv2*dd;
  c33[i]=vv2; 
  dc33[i]=2*vv*dv[i]*v0;
  dc11[i]=dc33[i]*ee+2*vv2*deps[i]*eps0;
  dc13[i]=dc33[i]*dd+vv2/dd*ddel[i]*del0;
 }
 return;
}

void GradCij2GradVEpsDel(float *gv,float *geps,float *gdel,const float *gc11,const float *gc13,const float *gc33,const float *v,const float *eps,const float *del,float v0,float eps0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float tv=v0*v[i];
  float teps=1.+2*eps0*eps[i];
  float tdel=sqrt(1.+2*del0*del[i]);
  gv[i]=2*tv*(gc11[i]*teps+gc13[i]*tdel+gc33[i])*v0;
  tv=tv*tv;
  geps[i]=2*tv*gc11[i]*eps0;
  gdel[i]=tv/tdel*gc13[i]*del0;
 }
 return;
}

void VnEtaDel2Cij(float *c11,float *c13,float *c33,const float *vn,const float *eta,const float *del,float vn0,float eta0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float vn2=vn0*vn0*vn[i]*vn[i];
  c11[i]=vn2*(1.+2*eta0*eta[i]); 
  float tdel=1.+2*del0*del[i];
  c13[i]=vn2/sqrt(tdel); 
  c33[i]=vn2/tdel; 
 }
 return;
}

void dVnEtaDel2dCij(float *c11,float *c13,float *c33,float *dc11,float *dc13,float *dc33,const float *vn,const float *eta,const float *del,const float *dvn,const float *deta,const float *ddel,float vn0,float eta0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float vv=vn0*vn[i],ee=eta0*eta[i],dd=del0*del[i];
  float vv2=vv*vv;
  ee=1.+2*ee;
  dd=1.+2*dd;
  float dd2=sqrt(dd);
  c11[i]=vv2*ee; 
  c13[i]=vv2/dd2;
  c33[i]=vv2/dd; 
  dc11[i]=2*vv*(ee*dvn[i]*vn0+vv*deta[i]*eta0);
  dc13[i]=vv/dd2*(2*dvn[i]*vn0-vv*ddel[i]*del0/dd);
  dc33[i]=vv/dd*(dvn[i]*vn0-2*vv*ddel[i]*del0/dd);
 }
 return;
}

void GradCij2GradVnEtaDel(float *gvn,float *geta,float *gdel,const float *gc11,const float *gc13,const float *gc33,const float *vn,const float *eta,const float *del,float vn0,float eta0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float tvn=vn0*vn[i];
  float teta=1.+2*eta0*eta[i];
  float tdel=1.+2*del0*del[i];
  float tt=sqrt(tdel);
  gvn[i]=2*tvn*(gc11[i]*teta+gc13[i]/tt+gc33[i]/tdel)*vn0;
  tvn=tvn*tvn;
  geta[i]=2*tvn*gc11[i]*eta0;
  gdel[i]=-tvn/tdel*(gc13[i]/tt+2*gc33[i]/tdel)*del0;
 }
 return;
}

void VhEpsEta2Cij(float *c11,float *c13,float *c33,const float *vh,const float *eps,const float *eta,float vh0,float eps0,float eta0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  c11[i]=vh0*vh0*vh[i]*vh[i]; 
  float teps=1.+2*eps0*eps[i];
  c13[i]=c11[i]/sqrt(teps*(1.+2*eta0*eta[i])); 
  c33[i]=c11[i]/teps; 
 }
 return;
}

void dVhEpsEta2dCij(float *c11,float *c13,float *c33,float *dc11,float *dc13,float *dc33,const float *vh,const float *eps,const float *eta,const float *dvh,const float *deps,const float *deta,float vh0,float eps0,float eta0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float vv=vh0*vh[i],ee=eps0*eps[i],dd=eta0*eta[i];
  ee=1.+2*ee;
  dd=1.+2*dd;
  float vv2=vv*vv,ee2=sqrt(ee),dd2=sqrt(dd);
  c11[i]=vv2; 
  c13[i]=vv2/ee2/dd2;
  c33[i]=vv2/ee; 
  dc11[i]=2*vv*dvh[i]*vh0;
  dc13[i]=vv/dd2/ee2*(2*dvh[i]*vh0-vv/ee*deps[i]*eps0-vv/dd*deta[i]*eta0);
  dc33[i]=2*vv/ee*(dvh[i]*vh0-vv/ee*deta[i]*eta0);
 }
 return;
}

void GradCij2GradVhEpsEta(float *gvh,float *geps,float *geta,const float *gc11,const float *gc13,const float *gc33,const float *vh,const float *eps,const float *eta,float vh0,float eps0,float eta0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float tvh=vh0*vh[i];
  float teps=1.+2*eps0*eps[i];
  float teta=1.+2*eta0*eta[i];
  float tt=sqrt(teps*teta);
  gvh[i]=2*tvh*(gc11[i]+gc13[i]/tt+gc33[i]/teps)*vh0;
  tvh=tvh*tvh;
  geps[i]=-tvh/teps*(gc13[i]/tt+2*gc33[i]/teps)*eps0;
  geta[i]=-tvh/tt/teta*gc13[i]*eta0;
 }
 return;
}

void VVhDel2Cij(float *c11,float *c13,float *c33,const float *v,const float *vh,const float *del,float v0,float vh0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  c11[i]=vh0*vh0*vh[i]*vh[i]; 
  c33[i]=v0*v0*v[i]*v[i]; 
  c13[i]=c33[i]*sqrt(1.+2*del0*del[i]); 
 }
 return;
}

void GradCij2GradVVhDel(float *gv,float *gvh,float *gdel,const float *gc11,const float *gc13,const float *gc33,const float *v,const float *vh,const float *del,float v0,float vh0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float tv=v0*v[i];
  float tvh=vh0*vh[i];
  float tdel=sqrt(1.+2*del0*del[i]);
  gv[i]=2*tv*(gc13[i]*tdel+gc33[i])*v0;
  gvh[i]=2*tvh*gc11[i]*vh0;
  gdel[i]=tv*tv/tdel*gc13[i]*del0;
 }
 return;
}

void VnVhDel2Cij(float *c11,float *c13,float *c33,const float *vn,const float *vh,const float *del,float vn0,float vh0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  c11[i]=vh0*vh0*vh[i]*vh[i]; 
  float tvn2=vn0*vn[i];
  tvn2=tvn2*tvn2;
  float tdel=1.+2*del0*del[i];
  c13[i]=tvn2/sqrt(tdel); 
  c33[i]=tvn2/tdel; 
 }
 return;
}

void GradCij2GradVnVhDel(float *gvn,float *gvh,float *gdel,const float *gc11,const float *gc13,const float *gc33,const float *vn,const float *vh,const float *del,float vn0,float vh0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float tvn=vn0*vn[i];
  float tvh=vh0*vh[i];
  float tdel=1.+2*del0*del[i];
  gvn[i]=2*tvn*(gc13[i]/sqrt(tdel)+gc33[i]/tdel)*vn0;
  gvh[i]=2*tvh*gc11[i]*vh0;
  gdel[i]=-tvn*tvn/tdel*(gc13[i]/sqrt(tdel)+2*gc33[i]/tdel)*del0;
 }
 return;
}

void VVnVh2Cij(float *c11,float *c13,float *c33,const float *v,const float *vn,const float *vh,float v0,float vn0,float vh0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  c11[i]=vh0*vh[i]*vh0*vh[i]; 
  float tv=v0*v[i];
  c13[i]=tv*vn0*vn[i]; 
  c33[i]=tv*tv; 
 }
 return;
}

void GradCij2GradVVnVh(float *gv,float *gvn,float *gvh,const float *gc11,const float *gc13,const float *gc33,const float *v,const float *vn,const float *vh,float v0,float vn0,float vh0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float tv=v0*v[i];
  float tvn=vn0*vn[i];
  float tvh=vh0*vh[i];
  gv[i]=(gc13[i]*tvn+gc33[i]*2*tv)*v0;
  gvn[i]=(gc13[i]*tv)*vn0;
  gvh[i]=(gc11[i]*2*tvh)*vh0;
 }
 return;
}

void VhEpsDel2Cij(float *c11,float *c13,float *c33,const float *vh,const float *eps,const float *del,float vh0,float eps0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  c11[i]=vh0*vh0*vh[i]*vh[i]; 
  float teps=1.+2*eps0*eps[i];
  c33[i]=c11[i]/teps; 
  c13[i]=c33[i]*sqrt(1.+2*del0*del[i]); 
 }
 return;
}

void dVhEpsDel2dCij(float *c11,float *c13,float *c33,float *dc11,float *dc13,float *dc33,const float *vh,const float *eps,const float *del,const float *dvh,const float *deps,const float *ddel,float vh0,float eps0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float vv=vh0*vh[i],ee=eps0*eps[i],dd=del0*del[i];
  float vv2=vv*vv;
  ee=1.+2*ee;
  dd=1.+2*dd;
  float dd2=sqrt(dd);
  c11[i]=vv2; 
  c13[i]=vv2*dd2/ee;
  c33[i]=vv2/ee; 
  dc11[i]=2*vv*dvh[i]*vh0;
  dc13[i]=vv/ee*dd2*(2*dvh[i]*vh0-2*vv/ee*deps[i]*eps0+vv/dd*ddel[i]*del0);
  dc33[i]=2*vv/ee*(dvh[i]*vh0-vv*dd2/ee*deps[i]*eps0);
 }
 return;
}

void GradCij2GradVhEpsDel(float *gvh,float *geps,float *gdel,const float *gc11,const float *gc13,const float *gc33,const float *vh,const float *eps,const float *del,float vh0,float eps0,float del0,size_t n){
 #pragma omp parallel for num_threads(16)
 for(size_t i=0;i<n;++i){
  float tvh=vh0*vh[i];
  float teps=1.+2*eps0*eps[i];
  float tdel=sqrt(1.+2*del0*del[i]);
  gvh[i]=2*tvh*(gc11[i]+gc13[i]*tdel/teps+gc33[i]/teps)*vh0;
  tvh=tvh*tvh;
  geps[i]=-2*tvh/teps/teps*(gc13[i]*tdel+gc33[i])*eps0;
  gdel[i]=tvh/tdel/teps*gc13[i]*del0;
 }
 return;
}

