#include "wave3d.h"

void init_abc(float *damping,int nx,int ny,int npad){
	#pragma omp parallel for num_threads(16)
	for(int iy=0;iy<ny;iy++){
		float dist;
		if(iy<npad) dist=npad-iy;
		else if(iy>ny-npad) dist=iy-ny+npad;
		else dist=0.;
		float dampingY=DAMPER+(1.-DAMPER)*cos(PI*dist/npad);
		for(int ix=0;ix<nx;ix++){
			if(ix<npad) dist=npad-ix;
			else if(ix>nx-npad) dist=ix-nx+npad;
			else dist=0.;
			float dampingX=DAMPER+(1.-DAMPER)*cos(PI*dist/npad);
			damping[ix+iy*nx]=dampingX*dampingY;
		}
	}
	
	return;
}

void init_abc(float *damping,int nx,int ny,int nz,int npad){
	#pragma omp parallel for num_threads(16)
	for(int iy=0;iy<ny;iy++){
		float dist;
		if(iy<npad) dist=npad-iy;
		else if(iy>ny-npad) dist=iy-ny+npad;
		else dist=0.;
		float dampingY=DAMPER+(1.-DAMPER)*cos(PI*dist/npad);
		for(int ix=0;ix<nx;ix++){
			if(ix<npad) dist=npad-ix;
			else if(ix>nx-npad) dist=ix-nx+npad;
			else dist=0.;
			float dampingX=DAMPER+(1.-DAMPER)*cos(PI*dist/npad);
			damping[ix+iy*nx]=dampingX*dampingY;
		}
	}
    int nxy=nx*ny;
	#pragma omp parallel for num_threads(16)
	for(int iz=0;iz<nz;iz++){
		float dist;
		if(iz<npad) dist=npad-iz;
		else if(iz>nz-npad) dist=iz-nz+npad;
		else dist=0.;
		damping[nxy+iz]=DAMPER+(1.-DAMPER)*cos(PI*dist/npad);
    }
	
	return;
}

__global__ void abc(int ib,int nx,int ny,int nz,int npad,float *nextSigmaX,float *curSigmaX1,float *nextSigmaZ,float *curSigmaZ1,const float *damping){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int ixy=ix+iy*nx;

  float dist,dampingZ;
  for(int i=0;i<HALF_STENCIL;++i){
   int iz=ib*HALF_STENCIL+i;
   int nxy=nx*ny;

   if(iz<npad) dist=npad-iz;
   else if(iz>nz-npad) dist=iz-nz+npad;
   else dist=0.;
   dampingZ=DAMPER+(1.-DAMPER)*cos(PI*dist/npad);
   
   dampingZ=damping[ix+iy*nx]*damping[nxy+iz];

   int j=ixy+i*nxy;
   nextSigmaX[j]*=dampingZ;
   curSigmaX1[j]*=dampingZ;
   nextSigmaZ[j]*=dampingZ;
   curSigmaZ1[j]*=dampingZ;
  }
 }
	
 return;
}

__global__ void abcXYZ(int ib,int nx,int ny,int nz,int npad,float *nextSigmaX,float *curSigmaX1,float *nextSigmaZ,float *curSigmaZ1,const float *damping){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int ixy=ix+iy*nx;

  float dampingXYZ;
  for(int i=0;i<HALF_STENCIL;++i){
   int iz=ib*HALF_STENCIL+i;
   int nxy=nx*ny;
   dampingXYZ=damping[ix+iy*nx]*damping[nxy+iz];
   int j=ixy+i*nxy;
   nextSigmaX[j]*=dampingXYZ;
   curSigmaX1[j]*=dampingXYZ;
   nextSigmaZ[j]*=dampingXYZ;
   curSigmaZ1[j]*=dampingXYZ;
  }
 }
	
 return;
}

__global__ void abcXY(int ib,int nx,int ny,int nz,int npad,float *nextSigmaX,float *curSigmaX1,float *nextSigmaZ,float *curSigmaZ1,const float *damping){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int ixy=ix+iy*nx;

  float dampingXY;
  for(int i=0;i<HALF_STENCIL;++i){
   int nxy=nx*ny;
   dampingXY=damping[ix+iy*nx];
   int j=ixy+i*nxy;
   nextSigmaX[j]*=dampingXY;
   curSigmaX1[j]*=dampingXY;
   nextSigmaZ[j]*=dampingXY;
   curSigmaZ1[j]*=dampingXY;
  }
 }
	
 return;
}

