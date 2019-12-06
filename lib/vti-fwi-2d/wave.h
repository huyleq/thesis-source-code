#ifndef WAVE_H
#define WAVE_H

void synthetic_f(float *d,const float *c11,const float *c13,const float *c33,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot=0.);

void rtm_f(float *image,const float *data,const float *c11,const float *c13,const float *c33,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot=0.);

float objFuncGradientCij(float *gc11,float *gc13,float *gc33,const float *d0,float *c11,float *c13,float *c33,float c110,float c130,float c330,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

float objFuncGradientVEpsDel(float *gv,float *geps,float *gdel,const float *data,float *v,float *eps,float *del,float v0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

float objFuncGradient_f(float *gc11,float *gc13,float *gc33,const float *data,const float *c11,const float *c13,const float *c33,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot=0.);

void modeling_f(float *d,const float *c11,const float *c13,const float *c33,const float *wavelet,int slocxz,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot=0.);

void odcig_f(float *image,const float *data,const float *c11,const float *c13,const float *c33,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nhx,int nt,int npad,float dx,float dz,float dt,float rate,float ot=0.);

void waveletGradient_f(float *gwavelet,const float *d0,const float *c11,const float *c13,const float *c33,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate);

float objFuncGradientVnEtaDel(float *gvn,float *geta,float *gdel,const float *d0,float *vn,float *eta,float *del,float vn0,float eta0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

float objFuncGradientVhEpsEta(float *gvh,float *geps,float *geta,const float *d0,float *vh,float *eps,float *eta,float vh0,float eps0,float eta0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

float objFuncGradientVVhDel(float *gv,float *gvh,float *gdel,const float *d0,float *v,float *vh,float *del,float v0,float vh0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

float objFuncGradientVnVhDel(float *gvn,float *gvh,float *gdel,const float *d0,float *vn,float *vh,float *del,float vn0,float vh0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

float objFuncGradientVhEpsDel(float *gvh,float *geps,float *gdel,const float *d0,float *vh,float *eps,float *del,float vh0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

float objFuncGradientVVnVh(float *gv,float *gvn,float *gvh,const float *d0,float *v,float *vn,float *vh,float v0,float vn0,float vh0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);
 
void hessian_f(float *gc11,float *gc13,float *gc33,const float *data,const float *c11,const float *c13,const float *c33,const float *dc11,const float *dc13,const float *dc33,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot);

void GNhessian_f(float *gc11,float *gc13,float *gc33,const float *data,const float *c11,const float *c13,const float *c33,const float *dc11,const float *dc13,const float *dc33,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot);

void hessianCij(float *gc11,float *gc13,float *gc33,const float *d0,float *c11,float *c13,float *c33,const float *dc11,const float *dc13,const float *dc33,float c110,float c130,float c330,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

void GNhessianCij(float *gc11,float *gc13,float *gc33,const float *d0,float *c11,float *c13,float *c33,const float *dc11,const float *dc13,const float *dc33,float c110,float c130,float c330,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

void hessianVEpsDel(float *gv,float *geps,float *gdel,const float *d0,float *v,float *eps,float *del,const float *dv,const float *deps,const float *ddel,float v0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

void GNhessianVEpsDel(float *gv,float *geps,float *gdel,const float *d0,float *v,float *eps,float *del,const float *dv,const float *deps,const float *ddel,float v0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

void hessianVhEpsDel(float *gvh,float *geps,float *gdel,const float *d0,float *vh,float *eps,float *del,const float *dvh,const float *deps,const float *ddel,float vh0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

void GNhessianVhEpsDel(float *gvh,float *geps,float *gdel,const float *d0,float *vh,float *eps,float *del,const float *dvh,const float *deps,const float *ddel,float vh0,float eps0,float del0,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m);

void modelingR_f(float *wavefield,const float *r11,const float *r13,const float *r33,const float *wavelet,int slocxz,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot);

void modelingABCD_f(float *wavefield,const float *a1,const float *b1c1,const float *d1,const float *a2,const float *b2c2,const float *d2,const float *wavelet,int slocxz,const float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot);

#endif
