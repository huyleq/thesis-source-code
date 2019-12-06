#ifndef ELASTIC_H
#define ELASTIC_H

#define HALF_STENCIL 4
#define STENCIL 8
#define DAMPER 0.95f
#define pi 3.14159265359

#define L 3

#define c0 1.1962890625
#define c1 -0.07975260416
#define c2 0.0095703125
#define c3 -0.00069754464

#define a0 1.27323954474
#define a1 -0.141471060526
#define a2 0.0509295817894
#define a4 -0.0259844805048

void vepsdel2cij(float *c11,float *c13,float *c33,float *c44,const float *vp,const float *vs,const float *eps,const float *del,const float *rho,int nxz);

void inject_source(float *sx,float *sz,float dtsou,int slocxz);

void inject_dipole_source(float *sx,float *sz,float dtsou,int slocxz,int nx);

void update_vel(float *vx,float *vz,const float *sx,const float *sz,const float *sxz,const float *buoy,int nx,int nz,const float *coeff);

void update_stress(float *sx,float *sz,float *sxz,const float *vx,const float *vz,const float *c11,const float *c13,const float *c33,const float *c44,int nx,int nz,const float *coeff);

void update_stress_memory(float *sx,float *sz,float *sxz,float **xix,float **xiz,float **xixz,const float *vx,const float *vz,const float *c11,const float *c13,const float *c33,const float *c44,const float *dtgbarQinv,const float *g,const float *w,int nx,int nz,const float *coeff,float dt);

void elastic_modeling_f(float *sx_wfld,float *sz_wfld,float *sxz_wfld,float *vx_wfld,float *vz_wfld,const float *wavelet,const float *c11,const float *c13,const float *c33,const float *c44,const float *buoy,int nx,int nz,int nt,int npad, float dxz,float dt,int isouxz,float sampling_rate);

void viscoelastic_modeling_f(float *sx_wfld,float *sz_wfld,float *sxz_wfld,float *vx_wfld,float *vz_wfld,const float *wavelet,const float *c11,const float *c13,const float *c33,const float *c44,const float *buoy,const float *dtgbarQinv,const float *g,const float *w,int nx,int nz,int nt,int npad, float dxz,float dt,int isouxz,float sampling_rate);

void init_abc(float *taper,int npad);

void abc(float *u,const float *taper,int nx,int nz,int npad);

void record_pressure(float *pdata,const int *rloc,int nr,const float *sx,const float *sz);

void record_vel(float *vxdata,float *vzdta,const int *rloc,int nr,const float *vx,const float *vz);

void elastic_synthetic_f(float *pdata,float *vxdata,float *vzdata,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *c11,const float *c13,const float *c33,const float *c44,const float *buoy,int nx,int nz,int nt,int npad, float dxz,float dt,float sampling_rate);

void viscoelastic_synthetic_f(float *pdata,float *vxdata,float *vzdata,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *c11,const float *c13,const float *c33,const float *c44,const float *buoy,const float *dtgbarQinv,const float *g,const float *w,int nx,int nz,int nt,int npad, float dxz,float dt,float sampling_rate);

#endif
