#ifndef CONVERSIONS_H
#define CONVERSIONS_H

void VEpsDel2ABCD(float *a1,float *b1c1,float *d1,float *a2,float *b2c2,float *d2,const float *v,const float *eps,const float *del,size_t n);

void VEpsDel2R(float *r11,float *r13,float *r33,const float *v,const float *eps,const float *del,size_t n);

void VEpsDel2Cij(float *c11,float *c13,float *c33,const float *v,const float *eps,const float *del,float v0,float eps0,float del0,size_t n);

void dVEpsDel2dCij(float *c11,float *c13,float *c33,float *dc11,float *dc13,float *dc33,const float *v,const float *eps,const float *del,const float *dv,const float *deps,const float *ddel,float v0,float eps0,float del0,size_t n);

void GradCij2GradVEpsDel(float *gv,float *geps,float *gdel,const float *gc11,const float *gc13,const float *gc33,const float *v,const float *eps,const float *del,float v0,float eps0,float del0,size_t n);

void VnEtaDel2Cij(float *c11,float *c13,float *c33,const float *vn,const float *eta,const float *del,float vn0,float eta0,float del0,size_t n);

void dVnEtaDel2dCij(float *c11,float *c13,float *c33,float *dc11,float *dc13,float *dc33,const float *vn,const float *eta,const float *del,const float *dvn,const float *deta,const float *ddel,float vn0,float eta0,float del0,size_t n);

void GradCij2GradVnEtaDel(float *gvn,float *geta,float *gdel,const float *gc11,const float *gc13,const float *gc33,const float *vn,const float *eta,const float *del,float vn0,float eta0,float del0,size_t n);

void VhEpsEta2Cij(float *c11,float *c13,float *c33,const float *vh,const float *eps,const float *eta,float vh0,float eps0,float eta0,size_t n);

void dVhEpsEta2dCij(float *c11,float *c13,float *c33,float *dc11,float *dc13,float *dc33,const float *vh,const float *eps,const float *eta,const float *dvh,const float *deps,const float *deta,float vh0,float eps0,float eta0,size_t n);

void GradCij2GradVhEpsEta(float *gvh,float *geps,float *geta,const float *gc11,const float *gc13,const float *gc33,const float *vh,const float *eps,const float *eta,float vh0,float eps0,float eta0,size_t n);

void VVhDel2Cij(float *c11,float *c13,float *c33,const float *v,const float *vh,const float *del,float v0,float vh0,float del0,size_t n);

void GradCij2GradVVhDel(float *gv,float *gvh,float *gdel,const float *gc11,const float *gc13,const float *gc33,const float *v,const float *vh,const float *del,float v0,float vh0,float del0,size_t n);

void VnVhDel2Cij(float *c11,float *c13,float *c33,const float *vn,const float *vh,const float *del,float vn0,float vh0,float del0,size_t n);

void GradCij2GradVnVhDel(float *gvn,float *gvh,float *gdel,const float *gc11,const float *gc13,const float *gc33,const float *vn,const float *vh,const float *del,float vn0,float vh0,float del0,size_t n);

void VVnVh2Cij(float *c11,float *c13,float *c33,const float *v,const float *vn,const float *vh,float v0,float vn0,float vh0,size_t n);

void GradCij2GradVVnVh(float *gv,float *gvn,float *gvh,const float *gc11,const float *gc13,const float *gc33,const float *v,const float *vn,const float *vh,float v0,float vn0,float vh0,size_t n);

void VhEpsDel2Cij(float *c11,float *c13,float *c33,const float *vh,const float *eps,const float *del,float vh0,float eps0,float del0,size_t n);

void dVhEpsDel2dCij(float *c11,float *c13,float *c33,float *dc11,float *dc13,float *dc33,const float *vh,const float *eps,const float *del,const float *dvh,const float *deps,const float *ddel,float vw0,float eps0,float del0,size_t n);

void GradCij2GradVhEpsDel(float *gvh,float *geps,float *gdel,const float *gc11,const float *gc13,const float *gc33,const float *vh,const float *eps,const float *del,float vh0,float eps0,float del0,size_t n);

#endif
