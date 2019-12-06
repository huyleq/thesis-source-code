#include <cstdio>
#include "kernels.h"

__global__ void forwardABCD(float *p0,float *q0,const float *p1,const float *q1,const float *a1,const float *b1c1,const float *d1,const float *a2,const float *b2c2,const float *d2,float dx2,float dz2,float dt2,int nnx,int nnz){
 __shared__ float sp[BLOCK_DIM_X+2*RADIUS][BLOCK_DIM_Y+2*RADIUS]; 
 __shared__ float sq[BLOCK_DIM_X+2*RADIUS][BLOCK_DIM_Y+2*RADIUS]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;

 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  int six=threadIdx.x+RADIUS;
  int siz=threadIdx.y+RADIUS;
 
  sp[six][siz]=p1[i];
  sq[six][siz]=q1[i];
  
  if(threadIdx.x<RADIUS){
   int k=min(blockDim.x,nnx-2*RADIUS-blockIdx.x*blockDim.x);
   sp[threadIdx.x][siz]=p1[i-RADIUS];
   sp[six+k][siz]=p1[i+k];
   sq[threadIdx.x][siz]=q1[i-RADIUS];
   sq[six+k][siz]=q1[i+k];
  }
 
  if(threadIdx.y<RADIUS){
   int k=min(blockDim.y,nnz-2*RADIUS-blockIdx.y*blockDim.y);
   sp[six][threadIdx.y]=p1[i-RADIUS*nnx];
   sp[six][siz+k]=p1[i+k*nnx];
   sq[six][threadIdx.y]=q1[i-RADIUS*nnx];
   sq[six][siz+k]=q1[i+k*nnx];
  }
 __syncthreads();
 
  float tpx=(C0*sp[six][siz]+C1*(sp[six-1][siz]+sp[six+1][siz])
                            +C2*(sp[six-2][siz]+sp[six+2][siz])
                            +C3*(sp[six-3][siz]+sp[six+3][siz])
                            +C4*(sp[six-4][siz]+sp[six+4][siz]))/dx2;
  float tpz=(C0*sp[six][siz]+C1*(sp[six][siz-1]+sp[six][siz+1])
                            +C2*(sp[six][siz-2]+sp[six][siz+2])
                            +C3*(sp[six][siz-3]+sp[six][siz+3])
                            +C4*(sp[six][siz-4]+sp[six][siz+4]))/dz2;
  float tqx=(C0*sq[six][siz]+C1*(sq[six-1][siz]+sq[six+1][siz])
                            +C2*(sq[six-2][siz]+sq[six+2][siz])
                            +C3*(sq[six-3][siz]+sq[six+3][siz])
                            +C4*(sq[six-4][siz]+sq[six+4][siz]))/dx2;
  float tqz=(C0*sq[six][siz]+C1*(sq[six][siz-1]+sq[six][siz+1])
                            +C2*(sq[six][siz-2]+sq[six][siz+2])
                            +C3*(sq[six][siz-3]+sq[six][siz+3])
                            +C4*(sq[six][siz-4]+sq[six][siz+4]))/dz2;
  p0[i]=2.f*p1[i]+dt2*(a1[i]*tpx+a2[i]*tpz+b1c1[i]*tqx+b2c2[i]*tqz)-p0[i];
  q0[i]=2.f*q1[i]+dt2*(b1c1[i]*tpx+b2c2[i]*tpz+d1[i]*tqx+d2[i]*tqz)-q0[i];
 }
 return;
}

__global__ void forwardRDR(float *p0,float *q0,const float *p1,const float *q1,const float *r11,const float *r13,const float *r33,float dx2,float dz2,float dt2,int nnx,int nnz){
 __shared__ float sp[BLOCK_DIM_X+2*RADIUS][BLOCK_DIM_Y]; 
 __shared__ float sq[BLOCK_DIM_X][BLOCK_DIM_Y+2*RADIUS]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;

 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  int six=threadIdx.x+RADIUS;
  int siz=threadIdx.y+RADIUS;
 
  sp[six][threadIdx.y]=r11[i]*p1[i]+r13[i]*q1[i];//tp[i];
  sq[threadIdx.x][siz]=r13[i]*p1[i]+r33[i]*q1[i];//tq[i];
  
  if(threadIdx.x<RADIUS){
   int k=min(blockDim.x,nnx-2*RADIUS-blockIdx.x*blockDim.x);
   sp[threadIdx.x][threadIdx.y]=r11[i-RADIUS]*p1[i-RADIUS]+r13[i-RADIUS]*q1[i-RADIUS];//tp[i-RADIUS];
   sp[six+k][threadIdx.y]=r11[i+k]*p1[i+k]+r13[i+k]*q1[i+k];//tp[i+k];
//   sq[threadIdx.x][siz]=r13[i-RADIUS]*p1[i-RADIUS]+r33[i-RADIUS]*q1[i-RADIUS];//tq[i-RADIUS];
//   sq[six+k][siz]=r13[i+k]*p1[i+k]+r33[i+k]*q1[i+k];//tq[i+k];
  }
 
  if(threadIdx.y<RADIUS){
   int k=min(blockDim.y,nnz-2*RADIUS-blockIdx.y*blockDim.y);
//   sp[six][threadIdx.y]=r11[i-RADIUS*nnx]*p1[i-RADIUS*nnx]+r13[i-RADIUS*nnx]*q1[i-RADIUS*nnx];//tp[i-RADIUS*nnx];
//   sp[six][siz+k]=r11[i+k*nnx]*p1[i+k*nnx]+r13[i+k*nnx]*q1[i+k*nnx];//tp[i+k*nnx];
   sq[threadIdx.x][threadIdx.y]=r13[i-RADIUS*nnx]*p1[i-RADIUS*nnx]+r33[i-RADIUS*nnx]*q1[i-RADIUS*nnx];//tq[i-RADIUS*nnx];
   sq[threadIdx.x][siz+k]=r13[i+k*nnx]*p1[i+k*nnx]+r33[i+k*nnx]*q1[i+k*nnx];//tq[i+k*nnx];
  }
 __syncthreads();

  float tp1=(C0*sp[six][threadIdx.y]+C1*(sp[six-1][threadIdx.y]+sp[six+1][threadIdx.y])
                               +C2*(sp[six-2][threadIdx.y]+sp[six+2][threadIdx.y])
                               +C3*(sp[six-3][threadIdx.y]+sp[six+3][threadIdx.y])
                               +C4*(sp[six-4][threadIdx.y]+sp[six+4][threadIdx.y]))/dx2;
  float tq1=(C0*sq[threadIdx.x][siz]+C1*(sq[threadIdx.x][siz-1]+sq[threadIdx.x][siz+1])
                               +C2*(sq[threadIdx.x][siz-2]+sq[threadIdx.x][siz+2])
                               +C3*(sq[threadIdx.x][siz-3]+sq[threadIdx.x][siz+3])
                               +C4*(sq[threadIdx.x][siz-4]+sq[threadIdx.x][siz+4]))/dz2;
  p0[i]=2.f*p1[i]+dt2*(r11[i]*tp1+r13[i]*tq1)-p0[i];
  q0[i]=2.f*q1[i]+dt2*(r13[i]*tp1+r33[i]*tq1)-q0[i];
 }
 return;
}

__global__ void backwardDC(float *p1,float *q1,const float *p0,const float *q0,const float *c11,const float *c13,const float *c33,float dx2,float dz2,float dt2,int nnx,int nnz){
 __shared__ float sp[BLOCK_DIM_X+2*RADIUS][BLOCK_DIM_Y]; 
 __shared__ float sq[BLOCK_DIM_X][BLOCK_DIM_Y+2*RADIUS]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;

 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  int six=threadIdx.x+RADIUS;
  int siz=threadIdx.y+RADIUS;
 
  sp[six][threadIdx.y]=c11[i]*p0[i]+c13[i]*q0[i];//tp[i];
  sq[threadIdx.x][siz]=c13[i]*p0[i]+c33[i]*q0[i];//tq[i];
  
  if(threadIdx.x<RADIUS){
   int k=min(blockDim.x,nnx-2*RADIUS-blockIdx.x*blockDim.x);
   sp[threadIdx.x][threadIdx.y]=c11[i-RADIUS]*p0[i-RADIUS]+c13[i-RADIUS]*q0[i-RADIUS];//tp[i-RADIUS];
   sp[six+k][threadIdx.y]=c11[i+k]*p0[i+k]+c13[i+k]*q0[i+k];//tp[i+k];
//   sq[threadIdx.x][siz]=c13[i-RADIUS]*p0[i-RADIUS]+c33[i-RADIUS]*q0[i-RADIUS];//tq[i-RADIUS];
//   sq[six+k][siz]=c13[i+k]*p0[i+k]+c33[i+k]*q0[i+k];//tq[i+k];
  }
 
  if(threadIdx.y<RADIUS){
   int k=min(blockDim.y,nnz-2*RADIUS-blockIdx.y*blockDim.y);
//   sp[six][threadIdx.y]=c11[i-RADIUS*nnx]*p0[i-RADIUS*nnx]+c13[i-RADIUS*nnx]*q0[i-RADIUS*nnx];//tp[i-RADIUS*nnx];
//   sp[six][siz+k]=c11[i+k*nnx]*p0[i+k*nnx]+c13[i+k*nnx]*q0[i+k*nnx];//tp[i+k*nnx];
   sq[threadIdx.x][threadIdx.y]=c13[i-RADIUS*nnx]*p0[i-RADIUS*nnx]+c33[i-RADIUS*nnx]*q0[i-RADIUS*nnx];//tq[i-RADIUS*nnx];
   sq[threadIdx.x][siz+k]=c13[i+k*nnx]*p0[i+k*nnx]+c33[i+k*nnx]*q0[i+k*nnx];//tq[i+k*nnx];
  }
 __syncthreads();

  float tp0=(C0*sp[six][threadIdx.y]+C1*(sp[six-1][threadIdx.y]+sp[six+1][threadIdx.y])
                               +C2*(sp[six-2][threadIdx.y]+sp[six+2][threadIdx.y])
                               +C3*(sp[six-3][threadIdx.y]+sp[six+3][threadIdx.y])
                               +C4*(sp[six-4][threadIdx.y]+sp[six+4][threadIdx.y]))/dx2;
  float tq0=(C0*sq[threadIdx.x][siz]+C1*(sq[threadIdx.x][siz-1]+sq[threadIdx.x][siz+1])
                               +C2*(sq[threadIdx.x][siz-2]+sq[threadIdx.x][siz+2])
                               +C3*(sq[threadIdx.x][siz-3]+sq[threadIdx.x][siz+3])
                               +C4*(sq[threadIdx.x][siz-4]+sq[threadIdx.x][siz+4]))/dz2;
  p1[i]=2.f*p0[i]+dt2*tp0-p1[i];
  q1[i]=2.f*q0[i]+dt2*tq0-q1[i];
 }
 return;
}

__global__ void scatteringa(float *dp,float *dq,const float *p0,const float *q0,const float *dc11,const float *dc13,const float *dc33,float dx2,float dz2,float dt2,int nnx,int nnz){
    //this is scattering for adjoint wavefields
 __shared__ float sp[BLOCK_DIM_X+2*RADIUS][BLOCK_DIM_Y]; 
 __shared__ float sq[BLOCK_DIM_X][BLOCK_DIM_Y+2*RADIUS]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;

 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  int six=threadIdx.x+RADIUS;
  int siz=threadIdx.y+RADIUS;
 
  sp[six][threadIdx.y]=dc11[i]*p0[i]+dc13[i]*q0[i];//tp[i];
  sq[threadIdx.x][siz]=dc13[i]*p0[i]+dc33[i]*q0[i];//tq[i];
  
  if(threadIdx.x<RADIUS){
   int k=min(blockDim.x,nnx-2*RADIUS-blockIdx.x*blockDim.x);
   sp[threadIdx.x][threadIdx.y]=dc11[i-RADIUS]*p0[i-RADIUS]+dc13[i-RADIUS]*q0[i-RADIUS];//tp[i-RADIUS];
   sp[six+k][threadIdx.y]=dc11[i+k]*p0[i+k]+dc13[i+k]*q0[i+k];//tp[i+k];
//   sq[threadIdx.x][siz]=dc13[i-RADIUS]*p0[i-RADIUS]+dc33[i-RADIUS]*q0[i-RADIUS];//tq[i-RADIUS];
//   sq[six+k][siz]=dc13[i+k]*p0[i+k]+dc33[i+k]*q0[i+k];//tq[i+k];
  }
 
  if(threadIdx.y<RADIUS){
   int k=min(blockDim.y,nnz-2*RADIUS-blockIdx.y*blockDim.y);
//   sp[six][threadIdx.y]=dc11[i-RADIUS*nnx]*p0[i-RADIUS*nnx]+dc13[i-RADIUS*nnx]*q0[i-RADIUS*nnx];//tp[i-RADIUS*nnx];
//   sp[six][siz+k]=dc11[i+k*nnx]*p0[i+k*nnx]+dc13[i+k*nnx]*q0[i+k*nnx];//tp[i+k*nnx];
   sq[threadIdx.x][threadIdx.y]=dc13[i-RADIUS*nnx]*p0[i-RADIUS*nnx]+dc33[i-RADIUS*nnx]*q0[i-RADIUS*nnx];//tq[i-RADIUS*nnx];
   sq[threadIdx.x][siz+k]=dc13[i+k*nnx]*p0[i+k*nnx]+dc33[i+k*nnx]*q0[i+k*nnx];//tq[i+k*nnx];
  }
 __syncthreads();

  float tp=(C0*sp[six][threadIdx.y]+C1*(sp[six-1][threadIdx.y]+sp[six+1][threadIdx.y])
                               +C2*(sp[six-2][threadIdx.y]+sp[six+2][threadIdx.y])
                               +C3*(sp[six-3][threadIdx.y]+sp[six+3][threadIdx.y])
                               +C4*(sp[six-4][threadIdx.y]+sp[six+4][threadIdx.y]))/dx2;
  float tq=(C0*sq[threadIdx.x][siz]+C1*(sq[threadIdx.x][siz-1]+sq[threadIdx.x][siz+1])
                               +C2*(sq[threadIdx.x][siz-2]+sq[threadIdx.x][siz+2])
                               +C3*(sq[threadIdx.x][siz-3]+sq[threadIdx.x][siz+3])
                               +C4*(sq[threadIdx.x][siz-4]+sq[threadIdx.x][siz+4]))/dz2;
  dp[i]+=dt2*tp;
  dq[i]+=dt2*tq;
 }
 return;
}

__global__ void injectResidual(float *p,float *q,const float *d00,const float *d01,const float *d0,const float *d1,float f,const int *rloc,int nr,int nnx,float dt2){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  int i=rloc[0+ir*2]+rloc[1+ir*2]*nnx;
  float s=0.5f*((1.f-f)*(d0[ir]-d00[ir])+f*(d1[ir]-d01[ir]));
  p[i]+=dt2*s;
  q[i]+=dt2*s;
 }
 return;
}

__global__ void injectData(float *p,float *q,const float *data0,const float *data1,float f,const int *rloc,int nr,int nnx,float dt2){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  int i=rloc[0+ir*2]+rloc[1+ir*2]*nnx;
  float s=0.5f*((1.f-f)*data0[ir]+f*data1[ir]);
  p[i]+=dt2*s;
  q[i]+=dt2*s;
 }
 return;
}

__global__ void injectDipoleResidual(float *p,float *q,const float *d00,const float *d01,const float *d0,const float *d1,float f,const int *rloc,int nr,int nnx,float dt2){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  int i=rloc[0+ir*2]+rloc[1+ir*2]*nnx;
  float s=0.5f*((1.f-f)*(d0[ir]-d00[ir])+f*(d1[ir]-d01[ir]));
  p[i]+=dt2*s;
  q[i]+=dt2*s;
  p[i+nnx]-=dt2*s;
  q[i+nnx]-=dt2*s;
 }
 return;
}

__global__ void injectDipoleData(float *p,float *q,const float *data0,const float *data1,float f,const int *rloc,int nr,int nnx,float dt2){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  int i=rloc[0+ir*2]+rloc[1+ir*2]*nnx;
  float s=0.5f*((1.f-f)*data0[ir]+f*data1[ir]);
  p[i]+=dt2*s;
  q[i]+=dt2*s;
  p[i+nnx]-=dt2*s;
  q[i+nnx]-=dt2*s;
 }
 return;
}

__global__ void injectSource(float *p,float *q,float dt2source,int slocxz){
 p[slocxz]+=dt2source;
 q[slocxz]+=dt2source;
 return;
}

__global__ void injectDipoleSource(float *p,float *q,float dt2source,int slocxz,int nnx){
 p[slocxz]+=dt2source;
 q[slocxz]+=dt2source;
 p[slocxz+nnx]-=dt2source;
 q[slocxz+nnx]-=dt2source;
 return;
}

__global__ void extractDipoleWavelet(float *gwavelet,float *p,float *q,int slocxz,int nnx,float dt2){
 *gwavelet=dt2*(p[slocxz]-p[slocxz+nnx]+q[slocxz]-q[slocxz+nnx]);
 return;
}

__global__ void extractWavelet(float *gwavelet,float *p,float *q,int slocxz,int nnx,float dt2){
 *gwavelet=dt2*(p[slocxz]+q[slocxz]);
 return;
}

__global__ void scattering(float *p,float *q,const float *dc11,const float *dc13,const float *dc33,const float *Dp0,const float *Dq0,const float *Dp1,const float *Dq1,float f,float dt2,int nnx,int nnz){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  float Dp=(1.f-f)*Dp0[i]+f*Dp1[i];
  float Dq=(1.f-f)*Dq0[i]+f*Dq1[i];
  p[i]+=dt2*(Dp*dc11[i]+Dq*dc13[i]);
  q[i]+=dt2*(Dp*dc13[i]+Dq*dc33[i]);
 }
 return;
}

__global__ void scattering(float *dp,float *dq,const float *dc11,const float *dc13,const float *dc33,const float *Dp,const float *Dq,float dt2,int nnx,int nnz){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  dp[i]+=dt2*(dc11[i]*Dp[i]+dc13[i]*Dq[i]);
  dq[i]+=dt2*(dc13[i]*Dp[i]+dc33[i]*Dq[i]);
 }
 return;
}

__global__ void abc(float *p,float *q,const float *taper,int nnx,int nnz){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  p[i]*=taper[i];
  q[i]*=taper[i];
 }
 return;
}

__global__ void abc(float *p1,float *q1,float *p0,float *q0,const float *taper,int nnx,int nnz){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  p1[i]*=taper[i];
  q1[i]*=taper[i];
  p0[i]*=taper[i];
  q0[i]*=taper[i];
 }
 return;
}

__global__ void recordData(float *d,const float *p,const float *q,const int *rloc,int nr,int nnx){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  int i=rloc[0+ir*2]+rloc[1+ir*2]*nnx;
  d[ir]=0.5f*(p[i]+q[i]);
 }
 return;
}

__global__ void recordDipoleData(float *d,const float *p,const float *q,const int *rloc,int nr,int nnx){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  int i=rloc[0+ir*2]+rloc[1+ir*2]*nnx;
  d[ir]=0.5f*(p[i]+q[i]-p[i+nnx]-q[i+nnx]);
 }
 return;
}

__global__ void recordWavefieldSlice(float *d,const float *p,const float *q,int nnx,int nnz){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  d[i]=0.5f*(p[i]+q[i]);
 }
 return;
}

__global__ void D(float *tp,float *tq,const float *p,const float *q,float dx2,float dz2,int nnx,int nnz){
 __shared__ float sp[BLOCK_DIM_X+2*RADIUS][BLOCK_DIM_Y]; 
 __shared__ float sq[BLOCK_DIM_X][BLOCK_DIM_Y+2*RADIUS]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 
 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  int six=threadIdx.x+RADIUS;
  int siz=threadIdx.y+RADIUS;
 
  sp[six][threadIdx.y]=p[i];
  sq[threadIdx.x][siz]=q[i];
  
  if(threadIdx.x<RADIUS){
   int k=min(blockDim.x,nnx-2*RADIUS-blockIdx.x*blockDim.x);
   sp[threadIdx.x][threadIdx.y]=p[i-RADIUS];
   sp[six+k][threadIdx.y]=p[i+k];
//   sq[threadIdx.x][siz]=q[i-RADIUS];
//   sq[six+k][siz]=q[i+k];
  }
 
  if(threadIdx.y<RADIUS){
   int k=min(blockDim.y,nnz-2*RADIUS-blockIdx.y*blockDim.y);
//   sp[six][threadIdx.y]=p[i-RADIUS*nnx];
//   sp[six][siz+k]=p[i+k*nnx];
   sq[threadIdx.x][threadIdx.y]=q[i-RADIUS*nnx];
   sq[threadIdx.x][siz+k]=q[i+k*nnx];
  }
 __syncthreads();

  tp[i]=(C0*sp[six][threadIdx.y]+C1*(sp[six-1][threadIdx.y]+sp[six+1][threadIdx.y])
                           +C2*(sp[six-2][threadIdx.y]+sp[six+2][threadIdx.y])
                           +C3*(sp[six-3][threadIdx.y]+sp[six+3][threadIdx.y])
                           +C4*(sp[six-4][threadIdx.y]+sp[six+4][threadIdx.y]))/dx2;
  tq[i]=(C0*sq[threadIdx.x][siz]+C1*(sq[threadIdx.x][siz-1]+sq[threadIdx.x][siz+1])
                           +C2*(sq[threadIdx.x][siz-2]+sq[threadIdx.x][siz+2])
                           +C3*(sq[threadIdx.x][siz-3]+sq[threadIdx.x][siz+3])
                           +C4*(sq[threadIdx.x][siz-4]+sq[threadIdx.x][siz+4]))/dz2;
 }
 return;
}

__global__ void C(float *tpp,float *tqq,const float *tp,const float *tq,
                  const float *c11,const float *c13,const float *c33,int nnx,int nnz){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  tpp[i]=c11[i]*tp[i]+c13[i]*tq[i];
  tqq[i]=c13[i]*tp[i]+c33[i]*tq[i];
 }
 return;
}

__global__ void dC(float *tpp,float *tqq,const float *tp0,const float *tq0,const float *tp1,const float *tq1,float f,const float *dc11,const float *dc13,const float *dc33,int nnx,int nnz){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  float tp=(1.f-f)*tp0[i]+f*tp1[i];
  float tq=(1.f-f)*tq0[i]+f*tq1[i];
  tpp[i]=dc11[i]*tp+dc13[i]*tq;
  tqq[i]=dc13[i]*tp+dc33[i]*tq;
 }
 return;
}

__global__ void imagingCrossCor(float *image,const float *ap,const float *aq,const float *sourceWavefieldSlice0,const float *sourceWavefieldSlice1,float f,int nnx,int nnz){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  float d=(1.f-f)*sourceWavefieldSlice0[i]+f*sourceWavefieldSlice1[i];
  image[i]+=0.5f*(ap[i]+aq[i])*d;
 }
 return;
}

__global__ void extendedImagingCrossCor(float *image,const float *ap,const float *aq,const float *sourceWavefieldSlice0,const float *sourceWavefieldSlice1,float f,int nnx,int nnz,int nhx){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 if(ix>=nhx && ix<nnx-nhx && iz<nnz){
  int i=ix+iz*nnx;
  for(int ihx=-nhx;ihx<=nhx;++ihx){
   float d=(1.f-f)*sourceWavefieldSlice0[i+ihx]+f*sourceWavefieldSlice1[i+ihx];
   image[i+(ihx+nhx)*nnx*nnz]+=0.5f*(ap[i-ihx]+aq[i-ihx])*d;
  }
 }
 return;
}

__global__ void gradientCrossCor(float *dc11,float *dc13,float *dc33,const float *ap,const float *aq,const float *Dp0,const float *Dq0,const float *Dp1,const float *Dq1,float f,int nnx,int nnz){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  float Dp=(1.f-f)*Dp0[i]+f*Dp1[i];
  float Dq=(1.f-f)*Dq0[i]+f*Dq1[i];
  dc11[i]+=ap[i]*Dp;
  dc13[i]+=ap[i]*Dq+aq[i]*Dp;
  dc33[i]+=aq[i]*Dq;
 }
 return;
}

__global__ void forwardCD(float *p0,float *q0,const float *p1,const float *q1,const float *c11,const float *c13,const float *c33,float dx2,float dz2,float dt2,int nnx,int nnz){
// __shared__ float sp[BLOCK_DIM_X+2*RADIUS][BLOCK_DIM_Y+2*RADIUS]; 
// __shared__ float sq[BLOCK_DIM_X+2*RADIUS][BLOCK_DIM_Y+2*RADIUS]; 
 __shared__ float sp[BLOCK_DIM_X+2*RADIUS][BLOCK_DIM_Y]; 
 __shared__ float sq[BLOCK_DIM_X][BLOCK_DIM_Y+2*RADIUS]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;

 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  int six=threadIdx.x+RADIUS;
  int siz=threadIdx.y+RADIUS;
 
  sp[six][threadIdx.y]=p1[i];
  sq[threadIdx.x][siz]=q1[i];
  
  if(threadIdx.x<RADIUS){
   int k=min(blockDim.x,nnx-2*RADIUS-blockIdx.x*blockDim.x);
   sp[threadIdx.x][threadIdx.y]=p1[i-RADIUS];
   sp[six+k][threadIdx.y]=p1[i+k];
//   sq[threadIdx.x][siz]=q1[i-RADIUS];
//   sq[six+k][siz]=q1[i+k];
  }
 
  if(threadIdx.y<RADIUS){
   int k=min(blockDim.y,nnz-2*RADIUS-blockIdx.y*blockDim.y);
//   sp[six][threadIdx.y]=p1[i-RADIUS*nnx];
//   sp[six][siz+k]=p1[i+k*nnx];
   sq[threadIdx.x][threadIdx.y]=q1[i-RADIUS*nnx];
   sq[threadIdx.x][siz+k]=q1[i+k*nnx];
  }
 __syncthreads();
 
  float tp=(C0*sp[six][threadIdx.y]+C1*(sp[six-1][threadIdx.y]+sp[six+1][threadIdx.y])
                              +C2*(sp[six-2][threadIdx.y]+sp[six+2][threadIdx.y])
                              +C3*(sp[six-3][threadIdx.y]+sp[six+3][threadIdx.y])
                              +C4*(sp[six-4][threadIdx.y]+sp[six+4][threadIdx.y]))/dx2;
  float tq=(C0*sq[threadIdx.x][siz]+C1*(sq[threadIdx.x][siz-1]+sq[threadIdx.x][siz+1])
                              +C2*(sq[threadIdx.x][siz-2]+sq[threadIdx.x][siz+2])
                              +C3*(sq[threadIdx.x][siz-3]+sq[threadIdx.x][siz+3])
                              +C4*(sq[threadIdx.x][siz-4]+sq[threadIdx.x][siz+4]))/dz2;
  p0[i]=2.f*p1[i]+dt2*(c11[i]*tp+c13[i]*tq)-p0[i];
  q0[i]=2.f*q1[i]+dt2*(c13[i]*tp+c33[i]*tq)-q0[i];
 }
 return;
}

__global__ void backwardD(float *p1,float *q1,const float *p0,const float *q0,const float *tp,const float *tq,float dx2,float dz2,float dt2,int nnx,int nnz){
 __shared__ float sp[BLOCK_DIM_X+2*RADIUS][BLOCK_DIM_Y]; 
 __shared__ float sq[BLOCK_DIM_X][BLOCK_DIM_Y+2*RADIUS]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;

 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  int six=threadIdx.x+RADIUS;
  int siz=threadIdx.y+RADIUS;
 
  sp[six][threadIdx.y]=tp[i];
  sq[threadIdx.x][siz]=tq[i];
  
  if(threadIdx.x<RADIUS){
   int k=min(blockDim.x,nnx-2*RADIUS-blockIdx.x*blockDim.x);
   sp[threadIdx.x][threadIdx.y]=tp[i-RADIUS];
   sp[six+k][threadIdx.y]=tp[i+k];
//   sq[threadIdx.x][siz]=tq[i-RADIUS];
//   sq[six+k][siz]=tq[i+k];
  }
 
  if(threadIdx.y<RADIUS){
   int k=min(blockDim.y,nnz-2*RADIUS-blockIdx.y*blockDim.y);
//   sp[six][threadIdx.y]=tp[i-RADIUS*nnx];
//   sp[six][siz+k]=tp[i+k*nnx];
   sq[threadIdx.x][threadIdx.y]=tq[i-RADIUS*nnx];
   sq[threadIdx.x][siz+k]=tq[i+k*nnx];
  }
 __syncthreads();

  float tp0=(C0*sp[six][threadIdx.y]+C1*(sp[six-1][threadIdx.y]+sp[six+1][threadIdx.y])
                               +C2*(sp[six-2][threadIdx.y]+sp[six+2][threadIdx.y])
                               +C3*(sp[six-3][threadIdx.y]+sp[six+3][threadIdx.y])
                               +C4*(sp[six-4][threadIdx.y]+sp[six+4][threadIdx.y]))/dx2;
  float tq0=(C0*sq[threadIdx.x][siz]+C1*(sq[threadIdx.x][siz-1]+sq[threadIdx.x][siz+1])
                               +C2*(sq[threadIdx.x][siz-2]+sq[threadIdx.x][siz+2])
                               +C3*(sq[threadIdx.x][siz-3]+sq[threadIdx.x][siz+3])
                               +C4*(sq[threadIdx.x][siz-4]+sq[threadIdx.x][siz+4]))/dz2;
  p1[i]=2.f*p0[i]+dt2*tp0-p1[i];
  q1[i]=2.f*q0[i]+dt2*tq0-q1[i];
 }
 return;
}

__global__ void forwardC(float *p0,float *q0,const float *p1,const float *q1,const float *tp,const float *tq,
                  const float *c11,const float *c13,const float *c33,float dt2,int nnx,int nnz){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+RADIUS;
 int iz=threadIdx.y+blockIdx.y*blockDim.y+RADIUS;
 if(ix<nnx-RADIUS && iz<nnz-RADIUS){
  int i=ix+iz*nnx;
  p0[i]=2.f*p1[i]+dt2*(c11[i]*tp[i]+c13[i]*tq[i])-p0[i];
  q0[i]=2.f*q1[i]+dt2*(c13[i]*tp[i]+c33[i]*tq[i])-q0[i];
 }
 return;
}

