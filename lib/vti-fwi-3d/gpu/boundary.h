#ifndef BOUNDARY_H
#define BOUNDARY_H

void getBoundary(float *boundary,const float *v,int nx,int ny,int nz,int npad);

void putBoundary(const float *boundary,float *v,int nx,int ny,int nz,int npad);

void zeroBoundary(float *v,int nx,int ny,int nz,int npad);

void pad1d(float *vout,float *vin,int nx,int npad);

void pad2d(float *vout,float *vin,int nx,int ny,int npad);

void pad3d(float *vout,float *vin,int nx,int ny,int nz,int npad);

#endif
