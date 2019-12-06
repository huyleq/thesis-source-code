#ifndef KERNELS_H
#define KERNELS_H

#define C0 -2.84722222222f
#define C1 1.6f
#define C2 -0.2f
#define C3 0.02539682539f
#define C4 -0.00178571428f

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define RADIUS 4

__global__ void forwardABCD(float *p0,float *q0,const float *p1,const float *q1,const float *a1,const float *b1c1,const float *d1,const float *a2,const float *b2c2,const float *d2,float dx2,float dz2,float dt2,int nnx,int nnz);

__global__ void forwardRDR(float *p0,float *q0,const float *p1,const float *q1,const float *r11,const float *r13,const float *r33,float dx2,float dz2,float dt2,int nnx,int nnz);

__global__ void backwardDC(float *p1,float *q1,const float *p0,const float *q0,const float *c11,const float *c13,const float *c33,float dx2,float dz2,float dt2,int nnx,int nnz);

__global__ void scatteringa(float *dp,float *dq,const float *p0,const float *q0,const float *dc11,const float *dc13,const float *dc33,float dx2,float dz2,float dt2,int nnx,int nnz);

__global__ void injectData(float *p,float *q,const float *data0,const float *data1,float f,const int *rloc,int nr,int nnx,float dt2);

__global__ void injectResidual(float *p,float *q,const float *d00,const float *d01,const float *d0,const float *d1,float f,const int *rloc,int nr,int nnx,float dt2);

__global__ void injectDipoleData(float *p,float *q,const float *data0,const float *data1,float f,const int *rloc,int nr,int nnx,float dt2);

__global__ void injectDipoleResidual(float *p,float *q,const float *d00,const float *d01,const float *d0,const float *d1,float f,const int *rloc,int nr,int nnx,float dt2);

__global__ void injectSource(float *p,float *q,float dt2source,int slocxz);

__global__ void injectDipoleSource(float *p,float *q,float dt2source,int slocxz,int nnx);

__global__ void extractDipoleWavelet(float *gwavelet,float *p,float *q,int slocxz,int nnx,float dt2);

__global__ void extractWavelet(float *gwavelet,float *p,float *q,int slocxz,int nnx,float dt2);

__global__ void scattering(float *p,float *q,const float *dc11,const float *dc13,const float *dc33,const float *Dp0,const float *Dq0,const float *Dp1,const float *Dq1,float f,float dt2,int nnx,int nnz);

__global__ void scattering(float *dp,float *dq,const float *dc11,const float *dc13,const float *dc33,const float *Dp,const float *Dq,float dt2,int nnx,int nnz);

__global__ void recordData(float *d,const float *p,const float *q,const int *rloc,int nr,int nnx);

__global__ void recordDipoleData(float *d,const float *p,const float *q,const int *rloc,int nr,int nnx);

__global__ void recordWavefieldSlice(float *d,const float *p,const float *q,int nnx,int nnz);

__global__ void abc(float *p,float *q,const float *taper,int nnx,int nnz);

__global__ void abc(float *p1,float *q1,float *p0,float *q0,const float *taper,int nnx,int nnz);

__global__ void D(float *tp,float *tq,const float *p,const float *q,float dx2,float dz2,int nnx,int nnz);

__global__ void C(float *tpp,float *tqq,const float *tp,const float *tq,
                  const float *c11,const float *c13,const float *c33,int nnx,int nnz);

__global__ void dC(float *tpp,float *tqq,const float *tp0,const float *tq0,const float *tp1,const float *tq1,float f,const float *dc11,const float *dc13,const float *dc33,int nnx,int nnz);

__global__ void imagingCrossCor(float *image,const float *ap,const float *aq,const float *sourceWavefieldSlice0,const float *sourceWavefieldSlice1,float f,int nnx,int nnz);

__global__ void extendedImagingCrossCor(float *image,const float *ap,const float *aq,const float *sourceWavefiedlSlice0,const float *sourceWavefieldSlice1,float f,int nnx,int nnz,int nhx);

__global__ void gradientCrossCor(float *dc11,float *dc13,float *dc33,const float *ap,const float *aq,const float *Dp0,const float *Dq0,const float *Dp1,const float *Dq1,float f,int nnx,int nnz);

__global__ void forwardCD(float *p0,float *q0,const float *p1,const float *q1,const float *c11,const float *c13,const float *c33,float dx2,float dz2,float dt2,int nnx,int nnz);

__global__ void backwardD(float *p1,float *q1,const float *p0,const float *q0,const float *tp,const float *tq,float dx2,float dz2,float dt2,int nnx,int nnz);

__global__ void forwardC(float *p0,float *q0,const float *p1,const float *q1,const float *tp,const float *tq,
                  const float *c11,const float *c13,const float *c33,float dt2,int nnx,int nnz);

#endif
