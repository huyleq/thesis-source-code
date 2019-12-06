#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"

using namespace std;

__global__ void testKernel(float *nextSigmaX,float *curSigmaX0,float *curSigmaX1,float *curSigmaX2,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *v,float *eps,float *del,int nx,int ny,float dx2,float dy2,float dz2,float dt2){

 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  for(int j=0;j<HALF_STENCIL;j++){
   int ij=i+j*nx*ny;
   nextSigmaX[ij]=curSigmaX1[ij]+1;
   nextSigmaZ[ij]=curSigmaZ1[ij]+1;
  }
 }

 return;
}

__global__ void forwardKernelBottomBlock(float *nextSigmaX,float *curSigmaX0,float *curSigmaX1,float *curSigmaX2,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *v,float *eps,float *del,int nx,int ny,float dx2,float dy2,float dz2,float dt2){
 
 __shared__ float sSigmaX[BLOCK_DIM+2*HALF_STENCIL][BLOCK_DIM+2*HALF_STENCIL]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  
  int six=threadIdx.x+HALF_STENCIL;
  int siy=threadIdx.y+HALF_STENCIL;
  
  float zSigmaZ[2*HALF_STENCIL+1];
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   zSigmaZ[iz+1]=curSigmaZ0[j];
   zSigmaZ[iz+1+HALF_STENCIL]=curSigmaZ1[j];
  }

  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   
   zSigmaZ[0]=zSigmaZ[1];
   zSigmaZ[1]=zSigmaZ[2];
   zSigmaZ[2]=zSigmaZ[3];
   zSigmaZ[3]=zSigmaZ[4];
   zSigmaZ[4]=zSigmaZ[5];
   zSigmaZ[5]=zSigmaZ[6];
   zSigmaZ[6]=zSigmaZ[7];
   zSigmaZ[7]=zSigmaZ[8];
   zSigmaZ[8]=0.;
   
   __syncthreads();
   
   sSigmaX[six][siy]=curSigmaX1[j];
   
   if(threadIdx.x<HALF_STENCIL){
    int k=min(blockDim.x,nx-2*HALF_STENCIL-blockIdx.x*blockDim.x);
    sSigmaX[threadIdx.x][siy]=curSigmaX1[j-HALF_STENCIL];
	sSigmaX[six+k][siy]=curSigmaX1[j+k];
   }
   
   if(threadIdx.y<HALF_STENCIL){
    int k=min(blockDim.y,ny-2*HALF_STENCIL-blockIdx.y*blockDim.y);
    sSigmaX[six][threadIdx.y]=curSigmaX1[j-HALF_STENCIL*nx];
	sSigmaX[six][siy+k]=curSigmaX1[j+k*nx];
   }

   __syncthreads();

   float c33=v[j]*v[j];
   float c11=c33*(1.+2.*eps[j]);
   float c13=c33*sqrt(1.+2.*del[j]);
  
   float tx=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six-1][siy]+sSigmaX[six+1][siy])
                                 +C2*(sSigmaX[six-2][siy]+sSigmaX[six+2][siy])
                                 +C3*(sSigmaX[six-3][siy]+sSigmaX[six+3][siy])
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))/dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))/dy2;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))/dz2;
   
   nextSigmaX[j]=dt2*(c11*(tx+ty)+c13*tz)+2.*sSigmaX[six][siy]-prevSigmaX[j];
   nextSigmaZ[j]=dt2*(c13*(tx+ty)+c33*tz)+2.*zSigmaZ[HALF_STENCIL]-prevSigmaZ[j];
  }
 } 

 return;
}

__global__ void forwardKernelTopBlock(float *nextSigmaX,float *curSigmaX0,float *curSigmaX1,float *curSigmaX2,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *v,float *eps,float *del,int nx,int ny,float dx2,float dy2,float dz2,float dt2){
 
 __shared__ float sSigmaX[BLOCK_DIM+2*HALF_STENCIL][BLOCK_DIM+2*HALF_STENCIL]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  
  int six=threadIdx.x+HALF_STENCIL;
  int siy=threadIdx.y+HALF_STENCIL;
  
  float zSigmaZ[2*HALF_STENCIL+1];
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   zSigmaZ[iz+1]=0.;
   zSigmaZ[iz+1+HALF_STENCIL]=curSigmaZ1[j];
  }

  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   
   zSigmaZ[0]=zSigmaZ[1];
   zSigmaZ[1]=zSigmaZ[2];
   zSigmaZ[2]=zSigmaZ[3];
   zSigmaZ[3]=zSigmaZ[4];
   zSigmaZ[4]=zSigmaZ[5];
   zSigmaZ[5]=zSigmaZ[6];
   zSigmaZ[6]=zSigmaZ[7];
   zSigmaZ[7]=zSigmaZ[8];
   zSigmaZ[8]=curSigmaZ2[j];
   
   __syncthreads();
   
   sSigmaX[six][siy]=curSigmaX1[j];
   
   if(threadIdx.x<HALF_STENCIL){
    int k=min(blockDim.x,nx-2*HALF_STENCIL-blockIdx.x*blockDim.x);
    sSigmaX[threadIdx.x][siy]=curSigmaX1[j-HALF_STENCIL];
	sSigmaX[six+k][siy]=curSigmaX1[j+k];
   }
   
   if(threadIdx.y<HALF_STENCIL){
    int k=min(blockDim.y,ny-2*HALF_STENCIL-blockIdx.y*blockDim.y);
    sSigmaX[six][threadIdx.y]=curSigmaX1[j-HALF_STENCIL*nx];
	sSigmaX[six][siy+k]=curSigmaX1[j+k*nx];
   }

   __syncthreads();

   float c33=v[j]*v[j];
   float c11=c33*(1.+2.*eps[j]);
   float c13=c33*sqrt(1.+2.*del[j]);
  
   float tx=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six-1][siy]+sSigmaX[six+1][siy])
                                 +C2*(sSigmaX[six-2][siy]+sSigmaX[six+2][siy])
                                 +C3*(sSigmaX[six-3][siy]+sSigmaX[six+3][siy])
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))/dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))/dy2;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))/dz2;
   
   nextSigmaX[j]=dt2*(c11*(tx+ty)+c13*tz)+2.*sSigmaX[six][siy]-prevSigmaX[j];
   nextSigmaZ[j]=dt2*(c13*(tx+ty)+c33*tz)+2.*zSigmaZ[HALF_STENCIL]-prevSigmaZ[j];
  }
 } 

 return;
}

__global__ void forwardKernel(float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *c11,float *c13,float *c33,int nx,int ny,float dt2dx2,float dt2dy2,float dt2dz2){
 
 __shared__ float sSigmaX[BLOCK_DIM+2*HALF_STENCIL][BLOCK_DIM+2*HALF_STENCIL]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  
  int six=threadIdx.x+HALF_STENCIL;
  int siy=threadIdx.y+HALF_STENCIL;
  
  float zSigmaZ[2*HALF_STENCIL+1];
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   zSigmaZ[iz+1]=curSigmaZ0[j];
   zSigmaZ[iz+1+HALF_STENCIL]=curSigmaZ1[j];
  }

  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   
   zSigmaZ[0]=zSigmaZ[1];
   zSigmaZ[1]=zSigmaZ[2];
   zSigmaZ[2]=zSigmaZ[3];
   zSigmaZ[3]=zSigmaZ[4];
   zSigmaZ[4]=zSigmaZ[5];
   zSigmaZ[5]=zSigmaZ[6];
   zSigmaZ[6]=zSigmaZ[7];
   zSigmaZ[7]=zSigmaZ[8];
   zSigmaZ[8]=curSigmaZ2[j];
   
   __syncthreads();
   
   sSigmaX[six][siy]=curSigmaX1[j];
   
   if(threadIdx.x<HALF_STENCIL){
    int k=min(blockDim.x,nx-2*HALF_STENCIL-blockIdx.x*blockDim.x);
    sSigmaX[threadIdx.x][siy]=curSigmaX1[j-HALF_STENCIL];
	sSigmaX[six+k][siy]=curSigmaX1[j+k];
   }
   
   if(threadIdx.y<HALF_STENCIL){
    int k=min(blockDim.y,ny-2*HALF_STENCIL-blockIdx.y*blockDim.y);
    sSigmaX[six][threadIdx.y]=curSigmaX1[j-HALF_STENCIL*nx];
	sSigmaX[six][siy+k]=curSigmaX1[j+k*nx];
   }

   __syncthreads();

   float tx=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six-1][siy]+sSigmaX[six+1][siy])
                                 +C2*(sSigmaX[six-2][siy]+sSigmaX[six+2][siy])
                                 +C3*(sSigmaX[six-3][siy]+sSigmaX[six+3][siy])
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))*dt2dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))*dt2dy2;
   ty+=tx;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))*dt2dz2;
   
   nextSigmaX[j]=c11[j]*ty+c13[j]*tz+2.*sSigmaX[six][siy]-prevSigmaX[j];
   nextSigmaZ[j]=c13[j]*ty+c33[j]*tz+2.*zSigmaZ[HALF_STENCIL]-prevSigmaZ[j];
  }
 } 

 return;
}

__global__ void injectSource(float *SigmaX,float *SigmaZ,float source,int souIndexBlock){
 SigmaX[souIndexBlock]+=source;
 SigmaZ[souIndexBlock]+=source;
 return;
}

__global__ void extractAdjWfldAtSouLoc(float *gwavelet,float *SigmaXa,float *SigmaZa,int souIndexBlock,int it){
 gwavelet[it]+=SigmaXa[souIndexBlock]+SigmaZa[souIndexBlock];
 return;
}

__global__ void injectDipoleSource(float *SigmaX,float *SigmaZ,float source,int souIndexBlock,int nxy){
 SigmaX[souIndexBlock]+=source;
 SigmaZ[souIndexBlock]+=source;
 SigmaX[souIndexBlock+nxy]-=source;
 SigmaZ[souIndexBlock+nxy]-=source;
 return;
}

__global__ void recordData(float *data,float *SigmaX,float *SigmaZ,int nr,const int *recIndex){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  data[ir]=TWOTHIRD*SigmaX[recIndex[ir]]+ONETHIRD*SigmaZ[recIndex[ir]];
 }
 return;
}

__global__ void recordDipoleData(float *data,float *SigmaX,float *SigmaZ,int nr,const int *recIndex,int nxy){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  data[ir]=TWOTHIRD*(SigmaX[recIndex[ir]]-SigmaX[recIndex[ir]+nxy])+ONETHIRD*(SigmaZ[recIndex[ir]]-SigmaZ[recIndex[ir]+nxy]);
 }
 return;
}

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

bool forwardAbc(int b,int e,int k,float ***d_SigmaX,float ***d_SigmaZ,const int *nbuffSigma,const float *d_damping,float **d_v,float **d_eps,float **d_del,const int nbuffVEpsDel,const float *wavelet,int souIndexBlock,int souBlock,int nr,const int *recIndex,int recBlock,int samplingTimeStep,float *data,int nx,int ny,int nz,int nt,int npad,float dx2,float dy2,float dz2,float dt2,cudaStream_t *stream,int gpu,int NGPU){
 dim3 block(BLOCK_DIM,BLOCK_DIM);
 dim3 grid((nx-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM,(ny-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM);
 
 bool record=false;
 int nb=nz/HALF_STENCIL;
 int roundlen=max(NGPU*(NUPDATE+3)+3,nb);
 int nround=(nt-2)/(NGPU*NUPDATE);
 
 for(int i=b;i<e;++i){
  int ib=(k-3-i)%roundlen;
  int iround=(k-3-i)/roundlen;
  if(ib>=0 && ib<nb && iround>=0 && iround<nround){
   int it=iround*NUPDATE*NGPU+gpu*NUPDATE+2+i;
//   fprintf(stderr,"i %d gpu %d kgpu %d iround %d updating block ib %d at time it %d\n",i,gpu,k,iround,ib,it);
   
   if(ib==0){
    forwardKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
   }
   else if(ib==nb-1){
    forwardKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
   }
   else{
    forwardKernel<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
   }
  
  if(ib==souBlock){
   float source=dt2*wavelet[it-1];
   injectSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock);
//    injectDipoleSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock,nx*ny);
   }
  
   abc<<<grid,block,0,*stream>>>(ib,nx,ny,nz,npad,d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_damping);
   int iz=ib*HALF_STENCIL;
   if(iz<npad || iz+HALF_STENCIL-1>nz-npad) abcXYZ<<<grid,block,0,*stream>>>(ib,nx,ny,nz,npad,d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_damping);
   else abcXY<<<grid,block,0,*stream>>>(ib,nx,ny,nz,npad,d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_damping);
   
   if(ib==recBlock && it%samplingTimeStep==0 && it<nt){
    recordData<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,*stream>>>(data,d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],nr,recIndex);
 //   recordDipoleData<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,*stream>>>(data,d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],nr,recIndex,nx*ny);
    record=true;
   }
  }
 }
 
 return record;
}

void forwardRandom(int b,int e,int k,float ***d_SigmaX,float ***d_SigmaZ,const int *nbuffSigma,float **d_v,float **d_eps,float **d_del,const int nbuffVEpsDel,float *wavelet,int souIndexBlock,int souBlock,int nx,int ny,int nz,int nt,int npad,float dx2,float dy2,float dz2,float dt2,cudaStream_t *stream,int gpu,int NGPU){
 dim3 block(BLOCK_DIM,BLOCK_DIM);
 dim3 grid((nx-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM,(ny-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM);
 
 int nb=nz/HALF_STENCIL;
 int roundlen=max(NGPU*(NUPDATE+3)+3,nb);
 int nround=(nt-2)/(NGPU*NUPDATE);
 
 for(int i=b;i<e;++i){
  int ib=(k-3-i)%roundlen;
  int iround=(k-3-i)/roundlen;
  if(ib>=0 && ib<nb && iround>=0 && iround<nround){
   int it=iround*NUPDATE*NGPU+gpu*NUPDATE+2+i;
  if(ib==0){
   forwardKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else if(ib==nb-1){
   forwardKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else{
   forwardKernel<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  
  if(ib==souBlock){
   float source=dt2*wavelet[it-1];
   injectSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock);
//   injectDipoleSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock,nx*ny);
  }
 }
 }
 
 return;
}

void backwardRandom(int b,int e,int k,float ***d_SigmaX,float ***d_SigmaZ,const int *nbuffSigma,float **d_v,float **d_eps,float **d_del,const int nbuffVEpsDel,float *wavelet,int souIndexBlock,int souBlock,int nx,int ny,int nz,int npad,int nt,float dx2,float dy2,float dz2,float dt2,cudaStream_t *stream,int gpu,int NGPU){
 dim3 block(BLOCK_DIM,BLOCK_DIM);
 dim3 grid((nx-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM,(ny-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM);
 
 for(int i=b;i<e;++i){
  int nb=nz/HALF_STENCIL;
  int ib=(k-3-i)%nb;
  int it=(k-3-i)/nb*NUPDATE*NGPU+gpu*NUPDATE+2+i;
  it=nt-1-it;
  
  if(ib==0){
   forwardKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else if(ib==nb-1){
   forwardKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else{
   forwardKernel<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  
  if(ib==souBlock){
   float source=dt2*wavelet[it+1];
   injectSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock);
//   injectDipoleSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock,nx*ny);
  }
 }
 
 return;
}

__global__ void adjointKernelBottomBlock(float *prevSigmaX,float *curSigmaX0,float *curSigmaX1,float *curSigmaX2,float *nextSigmaX,float *prevSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *nextSigmaZ,float *v0,float *v1,float *v2,float *eps0,float *eps1,float *eps2,float *del0,float *del1,float *del2,int nx,int ny,float dx2,float dy2,float dz2,float dt2){
 
 __shared__ float sSigmaX[BLOCK_DIM+2*HALF_STENCIL][BLOCK_DIM+2*HALF_STENCIL]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  
  int six=threadIdx.x+HALF_STENCIL;
  int siy=threadIdx.y+HALF_STENCIL;
  
  float zSigmaZ[2*HALF_STENCIL+1];
  
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   float c33=v0[j]*v0[j];
   float c13=c33*sqrt(1.+2.*del0[j]);
   zSigmaZ[iz+1]=c13*curSigmaX0[j]+c33*curSigmaZ0[j];
   c33=v1[j]*v1[j];
   c13=c33*sqrt(1.+2.*del1[j]);
   zSigmaZ[iz+1+HALF_STENCIL]=c13*curSigmaX1[j]+c33*curSigmaZ1[j];
  }

  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   
   zSigmaZ[0]=zSigmaZ[1];
   zSigmaZ[1]=zSigmaZ[2];
   zSigmaZ[2]=zSigmaZ[3];
   zSigmaZ[3]=zSigmaZ[4];
   zSigmaZ[4]=zSigmaZ[5];
   zSigmaZ[5]=zSigmaZ[6];
   zSigmaZ[6]=zSigmaZ[7];
   zSigmaZ[7]=zSigmaZ[8];
   zSigmaZ[8]=0.;
   
   __syncthreads();
   
   float c33=v1[j]*v1[j];
   float c11=c33*(1.+2.*eps1[j]);
   float c13=c33*sqrt(1.+2.*del1[j]);
   
   sSigmaX[six][siy]=c11*curSigmaX1[j]+c13*curSigmaZ1[j];
   
   if(threadIdx.x<HALF_STENCIL){
    int k=min(blockDim.x,nx-2*HALF_STENCIL-blockIdx.x*blockDim.x);
    c33=v1[j-HALF_STENCIL]*v1[j-HALF_STENCIL];
    c11=c33*(1.+2.*eps1[j-HALF_STENCIL]);
    c13=c33*sqrt(1.+2.*del1[j-HALF_STENCIL]);
    sSigmaX[threadIdx.x][siy]=c11*curSigmaX1[j-HALF_STENCIL]+c13*curSigmaZ1[j-HALF_STENCIL];
    c33=v1[j+k]*v1[j+k];
    c11=c33*(1.+2.*eps1[j+k]);
    c13=c33*sqrt(1.+2.*del1[j+k]);
	sSigmaX[six+k][siy]=c11*curSigmaX1[j+k]+c13*curSigmaZ1[j+k];
   }
   
   if(threadIdx.y<HALF_STENCIL){
    int k=min(blockDim.y,ny-2*HALF_STENCIL-blockIdx.y*blockDim.y);
    c33=v1[j-HALF_STENCIL*nx]*v1[j-HALF_STENCIL*nx];
    c11=c33*(1.+2.*eps1[j-HALF_STENCIL*nx]);
    c13=c33*sqrt(1.+2.*del1[j-HALF_STENCIL*nx]);
    sSigmaX[six][threadIdx.y]=c11*curSigmaX1[j-HALF_STENCIL*nx]+c13*curSigmaZ1[j-HALF_STENCIL*nx];
    c33=v1[j+k*nx]*v1[j+k*nx];
    c11=c33*(1.+2.*eps1[j+k*nx]);
    c13=c33*sqrt(1.+2.*del1[j+k*nx]);
	sSigmaX[six][siy+k]=c11*curSigmaX1[j+k*nx]+c13*curSigmaZ1[j+k*nx];
   }

   __syncthreads();
  
   float tx=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six-1][siy]+sSigmaX[six+1][siy])
                                 +C2*(sSigmaX[six-2][siy]+sSigmaX[six+2][siy])
                                 +C3*(sSigmaX[six-3][siy]+sSigmaX[six+3][siy])
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))/dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))/dy2;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))/dz2;
   
   prevSigmaX[j]=dt2*(tx+ty)+2.*curSigmaX1[j]-nextSigmaX[j];
   prevSigmaZ[j]=dt2*tz+2.*curSigmaZ1[j]-nextSigmaZ[j];
  }
 } 

 return;
}

__global__ void adjointKernelTopBlock(float *prevSigmaX,float *curSigmaX0,float *curSigmaX1,float *curSigmaX2,float *nextSigmaX,float *prevSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *nextSigmaZ,float *v0,float *v1,float *v2,float *eps0,float *eps1,float *eps2,float *del0,float *del1,float *del2,int nx,int ny,float dx2,float dy2,float dz2,float dt2){
 
 __shared__ float sSigmaX[BLOCK_DIM+2*HALF_STENCIL][BLOCK_DIM+2*HALF_STENCIL]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  
  int six=threadIdx.x+HALF_STENCIL;
  int siy=threadIdx.y+HALF_STENCIL;
  
  float zSigmaZ[2*HALF_STENCIL+1];
  
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   zSigmaZ[iz+1]=0.;
   float c33=v1[j]*v1[j];
   float c13=c33*sqrt(1.+2.*del1[j]);
   zSigmaZ[iz+1+HALF_STENCIL]=c13*curSigmaX1[j]+c33*curSigmaZ1[j];
  }

  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   
   float c33=v2[j]*v2[j];
   float c13=c33*sqrt(1.+2.*del2[j]);
   
   zSigmaZ[0]=zSigmaZ[1];
   zSigmaZ[1]=zSigmaZ[2];
   zSigmaZ[2]=zSigmaZ[3];
   zSigmaZ[3]=zSigmaZ[4];
   zSigmaZ[4]=zSigmaZ[5];
   zSigmaZ[5]=zSigmaZ[6];
   zSigmaZ[6]=zSigmaZ[7];
   zSigmaZ[7]=zSigmaZ[8];
   zSigmaZ[8]=c13*curSigmaX2[j]+c33*curSigmaZ2[j];
   
   __syncthreads();
   
   c33=v1[j]*v1[j];
   float c11=c33*(1.+2.*eps1[j]);
   c13=c33*sqrt(1.+2.*del1[j]);
   
   sSigmaX[six][siy]=c11*curSigmaX1[j]+c13*curSigmaZ1[j];
   
   if(threadIdx.x<HALF_STENCIL){
    int k=min(blockDim.x,nx-2*HALF_STENCIL-blockIdx.x*blockDim.x);
    c33=v1[j-HALF_STENCIL]*v1[j-HALF_STENCIL];
    c11=c33*(1.+2.*eps1[j-HALF_STENCIL]);
    c13=c33*sqrt(1.+2.*del1[j-HALF_STENCIL]);
    sSigmaX[threadIdx.x][siy]=c11*curSigmaX1[j-HALF_STENCIL]+c13*curSigmaZ1[j-HALF_STENCIL];
    c33=v1[j+k]*v1[j+k];
    c11=c33*(1.+2.*eps1[j+k]);
    c13=c33*sqrt(1.+2.*del1[j+k]);
	sSigmaX[six+k][siy]=c11*curSigmaX1[j+k]+c13*curSigmaZ1[j+k];
   }
   
   if(threadIdx.y<HALF_STENCIL){
    int k=min(blockDim.y,ny-2*HALF_STENCIL-blockIdx.y*blockDim.y);
    c33=v1[j-HALF_STENCIL*nx]*v1[j-HALF_STENCIL*nx];
    c11=c33*(1.+2.*eps1[j-HALF_STENCIL*nx]);
    c13=c33*sqrt(1.+2.*del1[j-HALF_STENCIL*nx]);
    sSigmaX[six][threadIdx.y]=c11*curSigmaX1[j-HALF_STENCIL*nx]+c13*curSigmaZ1[j-HALF_STENCIL*nx];
    c33=v1[j+k*nx]*v1[j+k*nx];
    c11=c33*(1.+2.*eps1[j+k*nx]);
    c13=c33*sqrt(1.+2.*del1[j+k*nx]);
	sSigmaX[six][siy+k]=c11*curSigmaX1[j+k*nx]+c13*curSigmaZ1[j+k*nx];
   }

   __syncthreads();
  
   float tx=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six-1][siy]+sSigmaX[six+1][siy])
                                 +C2*(sSigmaX[six-2][siy]+sSigmaX[six+2][siy])
                                 +C3*(sSigmaX[six-3][siy]+sSigmaX[six+3][siy])
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))/dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))/dy2;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))/dz2;
   
   prevSigmaX[j]=dt2*(tx+ty)+2.*curSigmaX1[j]-nextSigmaX[j];
   prevSigmaZ[j]=dt2*tz+2.*curSigmaZ1[j]-nextSigmaZ[j];
  }
 } 

 return;
}

__global__ void adjointKernel(float *prevSigmaX,float *curSigmaX0,float *curSigmaX1,float *curSigmaX2,float *nextSigmaX,float *prevSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *nextSigmaZ,float *v0,float *v1,float *v2,float *eps0,float *eps1,float *eps2,float *del0,float *del1,float *del2,int nx,int ny,float dx2,float dy2,float dz2,float dt2){
 
 __shared__ float sSigmaX[BLOCK_DIM+2*HALF_STENCIL][BLOCK_DIM+2*HALF_STENCIL]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  
  int six=threadIdx.x+HALF_STENCIL;
  int siy=threadIdx.y+HALF_STENCIL;
  
  float zSigmaZ[2*HALF_STENCIL+1];
  
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   float c33=v0[j]*v0[j];
   float c13=c33*sqrt(1.+2.*del0[j]);
   zSigmaZ[iz+1]=c13*curSigmaX0[j]+c33*curSigmaZ0[j];
   c33=v1[j]*v1[j];
   c13=c33*sqrt(1.+2.*del1[j]);
   zSigmaZ[iz+1+HALF_STENCIL]=c13*curSigmaX1[j]+c33*curSigmaZ1[j];
  }

  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   
   float c33=v2[j]*v2[j];
   float c13=c33*sqrt(1.+2.*del2[j]);
   
   zSigmaZ[0]=zSigmaZ[1];
   zSigmaZ[1]=zSigmaZ[2];
   zSigmaZ[2]=zSigmaZ[3];
   zSigmaZ[3]=zSigmaZ[4];
   zSigmaZ[4]=zSigmaZ[5];
   zSigmaZ[5]=zSigmaZ[6];
   zSigmaZ[6]=zSigmaZ[7];
   zSigmaZ[7]=zSigmaZ[8];
   zSigmaZ[8]=c13*curSigmaX2[j]+c33*curSigmaZ2[j];
   
   __syncthreads();
   
   c33=v1[j]*v1[j];
   float c11=c33*(1.+2.*eps1[j]);
   c13=c33*sqrt(1.+2.*del1[j]);
   
   sSigmaX[six][siy]=c11*curSigmaX1[j]+c13*curSigmaZ1[j];
   
   if(threadIdx.x<HALF_STENCIL){
    int k=min(blockDim.x,nx-2*HALF_STENCIL-blockIdx.x*blockDim.x);
    c33=v1[j-HALF_STENCIL]*v1[j-HALF_STENCIL];
    c11=c33*(1.+2.*eps1[j-HALF_STENCIL]);
    c13=c33*sqrt(1.+2.*del1[j-HALF_STENCIL]);
    sSigmaX[threadIdx.x][siy]=c11*curSigmaX1[j-HALF_STENCIL]+c13*curSigmaZ1[j-HALF_STENCIL];
    c33=v1[j+k]*v1[j+k];
    c11=c33*(1.+2.*eps1[j+k]);
    c13=c33*sqrt(1.+2.*del1[j+k]);
	sSigmaX[six+k][siy]=c11*curSigmaX1[j+k]+c13*curSigmaZ1[j+k];
   }
   
   if(threadIdx.y<HALF_STENCIL){
    int k=min(blockDim.y,ny-2*HALF_STENCIL-blockIdx.y*blockDim.y);
    c33=v1[j-HALF_STENCIL*nx]*v1[j-HALF_STENCIL*nx];
    c11=c33*(1.+2.*eps1[j-HALF_STENCIL*nx]);
    c13=c33*sqrt(1.+2.*del1[j-HALF_STENCIL*nx]);
    sSigmaX[six][threadIdx.y]=c11*curSigmaX1[j-HALF_STENCIL*nx]+c13*curSigmaZ1[j-HALF_STENCIL*nx];
    c33=v1[j+k*nx]*v1[j+k*nx];
    c11=c33*(1.+2.*eps1[j+k*nx]);
    c13=c33*sqrt(1.+2.*del1[j+k*nx]);
	sSigmaX[six][siy+k]=c11*curSigmaX1[j+k*nx]+c13*curSigmaZ1[j+k*nx];
   }

   __syncthreads();
  
   float tx=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six-1][siy]+sSigmaX[six+1][siy])
                                 +C2*(sSigmaX[six-2][siy]+sSigmaX[six+2][siy])
                                 +C3*(sSigmaX[six-3][siy]+sSigmaX[six+3][siy])
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))/dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))/dy2;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))/dz2;
   
   prevSigmaX[j]=dt2*(tx+ty)+2.*curSigmaX1[j]-nextSigmaX[j];
   prevSigmaZ[j]=dt2*tz+2.*curSigmaZ1[j]-nextSigmaZ[j];
  }
 } 

 return;
}

__global__ void injectResidual(float *residual,float *SigmaX,float *SigmaZ,int nr,const int *recIndex,float dt2){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  float s=dt2*residual[ir];
  SigmaX[recIndex[ir]]+=TWOTHIRD*s;
  SigmaZ[recIndex[ir]]+=ONETHIRD*s;
 }
 return;
}

__global__ void injectDipoleResidual(float *residual,float *SigmaX,float *SigmaZ,int nr,const int *recIndex,float dt2,int nxy){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  float s=dt2*residual[ir];
  SigmaX[recIndex[ir]]+=TWOTHIRD*s;
  SigmaZ[recIndex[ir]]+=ONETHIRD*s;
  SigmaX[recIndex[ir]+nxy]-=TWOTHIRD*s;
  SigmaZ[recIndex[ir]+nxy]-=ONETHIRD*s;
 }
 return;
}

void interpolateResidual(float *fineResidual,float *coarseResidual,int timeIndex,int nnt,int nr,int samplingTimeStep){
//	fprintf(stderr,"timeIndex %d\n",timeIndex);
	float f=float(timeIndex)/float(samplingTimeStep);
	int i=f;
	if(timeIndex>-1 && i>-1){
	 if(i>=nnt){
	  #pragma omp parallel for num_threads(16)
      for(int ir=0;ir<nr;ir++){
       fineResidual[ir]=coarseResidual[(nnt-1)+ir*nnt];
      }
	 }
	 else{
      f=f-i;
	  #pragma omp parallel for num_threads(16)
      for(int ir=0;ir<nr;ir++){
       fineResidual[ir]=(1.-f)*coarseResidual[i+ir*nnt]+f*coarseResidual[(i+1)+ir*nnt];
      }
	 }
	}
	return;
}

__global__ void gradientKernelBottomBlock(float *gv,float *geps,float *gdel,float *nextSigmaX,float *curSigmaX0,float *curSigmaX1,float *curSigmaX2,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *curSigmaX1a,float *curSigmaZ1a,float *v,float *eps,float *del,int nx,int ny,float dx2,float dy2,float dz2,float dt2){
 
 __shared__ float sSigmaX[BLOCK_DIM+2*HALF_STENCIL][BLOCK_DIM+2*HALF_STENCIL]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  
  int six=threadIdx.x+HALF_STENCIL;
  int siy=threadIdx.y+HALF_STENCIL;
  
  float zSigmaZ[2*HALF_STENCIL+1];
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   zSigmaZ[iz+1]=curSigmaZ0[j];
   zSigmaZ[iz+1+HALF_STENCIL]=curSigmaZ1[j];
  }

  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   
   zSigmaZ[0]=zSigmaZ[1];
   zSigmaZ[1]=zSigmaZ[2];
   zSigmaZ[2]=zSigmaZ[3];
   zSigmaZ[3]=zSigmaZ[4];
   zSigmaZ[4]=zSigmaZ[5];
   zSigmaZ[5]=zSigmaZ[6];
   zSigmaZ[6]=zSigmaZ[7];
   zSigmaZ[7]=zSigmaZ[8];
   zSigmaZ[8]=0.;
   
   __syncthreads();
   
   sSigmaX[six][siy]=curSigmaX1[j];
   
   if(threadIdx.x<HALF_STENCIL){
    int k=min(blockDim.x,nx-2*HALF_STENCIL-blockIdx.x*blockDim.x);
    sSigmaX[threadIdx.x][siy]=curSigmaX1[j-HALF_STENCIL];
	sSigmaX[six+k][siy]=curSigmaX1[j+k];
   }
   
   if(threadIdx.y<HALF_STENCIL){
    int k=min(blockDim.y,ny-2*HALF_STENCIL-blockIdx.y*blockDim.y);
    sSigmaX[six][threadIdx.y]=curSigmaX1[j-HALF_STENCIL*nx];
	sSigmaX[six][siy+k]=curSigmaX1[j+k*nx];
   }

   __syncthreads();

   float c33=v[j]*v[j];
   float c11=c33*(1.+2.*eps[j]);
   float sqrt12del=sqrt(1.+2.*del[j]);
   float c13=c33*sqrt12del;
  
   float tx=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six-1][siy]+sSigmaX[six+1][siy])
                                 +C2*(sSigmaX[six-2][siy]+sSigmaX[six+2][siy])
                                 +C3*(sSigmaX[six-3][siy]+sSigmaX[six+3][siy])
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))/dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))/dy2;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))/dz2;
   
   //update source wavefields
   nextSigmaX[j]=dt2*(c11*(tx+ty)+c13*tz)+2.*sSigmaX[six][siy]-prevSigmaX[j];
   nextSigmaZ[j]=dt2*(c13*(tx+ty)+c33*tz)+2.*zSigmaZ[HALF_STENCIL]-prevSigmaZ[j];
   
   //imaging conditions to compute gradient
   float gc11=curSigmaX1a[j]*(tx+ty);
   float gc33=curSigmaZ1a[j]*tz;
   float gc13=curSigmaX1a[j]*tz+curSigmaZ1a[j]*(tx+ty);
   
   gv[j]+=2.*v[j]*(gc11*(1.+2.*eps[j])+gc13*sqrt12del+gc33);
   geps[j]+=gc11*2.*c33;
   gdel[j]+=gc13*c33/sqrt12del;
  }
 } 

 return;
}

__global__ void gradientKernelTopBlock(float *gv,float *geps,float *gdel,float *nextSigmaX,float *curSigmaX0,float *curSigmaX1,float *curSigmaX2,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *curSigmaX1a,float *curSigmaZ1a,float *v,float *eps,float *del,int nx,int ny,float dx2,float dy2,float dz2,float dt2){
 
 __shared__ float sSigmaX[BLOCK_DIM+2*HALF_STENCIL][BLOCK_DIM+2*HALF_STENCIL]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  
  int six=threadIdx.x+HALF_STENCIL;
  int siy=threadIdx.y+HALF_STENCIL;
  
  float zSigmaZ[2*HALF_STENCIL+1];
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   zSigmaZ[iz+1]=0.;
   zSigmaZ[iz+1+HALF_STENCIL]=curSigmaZ1[j];
  }

  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   
   zSigmaZ[0]=zSigmaZ[1];
   zSigmaZ[1]=zSigmaZ[2];
   zSigmaZ[2]=zSigmaZ[3];
   zSigmaZ[3]=zSigmaZ[4];
   zSigmaZ[4]=zSigmaZ[5];
   zSigmaZ[5]=zSigmaZ[6];
   zSigmaZ[6]=zSigmaZ[7];
   zSigmaZ[7]=zSigmaZ[8];
   zSigmaZ[8]=curSigmaZ2[j];
   
   __syncthreads();
   
   sSigmaX[six][siy]=curSigmaX1[j];
   
   if(threadIdx.x<HALF_STENCIL){
    int k=min(blockDim.x,nx-2*HALF_STENCIL-blockIdx.x*blockDim.x);
    sSigmaX[threadIdx.x][siy]=curSigmaX1[j-HALF_STENCIL];
	sSigmaX[six+k][siy]=curSigmaX1[j+k];
   }
   
   if(threadIdx.y<HALF_STENCIL){
    int k=min(blockDim.y,ny-2*HALF_STENCIL-blockIdx.y*blockDim.y);
    sSigmaX[six][threadIdx.y]=curSigmaX1[j-HALF_STENCIL*nx];
	sSigmaX[six][siy+k]=curSigmaX1[j+k*nx];
   }

   __syncthreads();

   float c33=v[j]*v[j];
   float c11=c33*(1.+2.*eps[j]);
   float sqrt12del=sqrt(1.+2.*del[j]);
   float c13=c33*sqrt12del;
  
   float tx=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six-1][siy]+sSigmaX[six+1][siy])
                                 +C2*(sSigmaX[six-2][siy]+sSigmaX[six+2][siy])
                                 +C3*(sSigmaX[six-3][siy]+sSigmaX[six+3][siy])
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))/dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))/dy2;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))/dz2;
   
   //update source wavefields
   nextSigmaX[j]=dt2*(c11*(tx+ty)+c13*tz)+2.*sSigmaX[six][siy]-prevSigmaX[j];
   nextSigmaZ[j]=dt2*(c13*(tx+ty)+c33*tz)+2.*zSigmaZ[HALF_STENCIL]-prevSigmaZ[j];
   
   //imaging conditions to compute gradient
   float gc11=curSigmaX1a[j]*(tx+ty);
   float gc33=curSigmaZ1a[j]*tz;
   float gc13=curSigmaX1a[j]*tz+curSigmaZ1a[j]*(tx+ty);
   
   gv[j]+=2.*v[j]*(gc11*(1.+2.*eps[j])+gc13*sqrt12del+gc33);
   geps[j]+=gc11*2.*c33;
   gdel[j]+=gc13*c33/sqrt12del;
  }
 } 

 return;
}

__global__ void gradientKernel(float *gv,float *geps,float *gdel,float *nextSigmaX,float *curSigmaX0,float *curSigmaX1,float *curSigmaX2,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *curSigmaX1a,float *curSigmaZ1a,float *v,float *eps,float *del,int nx,int ny,float dx2,float dy2,float dz2,float dt2){
 
 __shared__ float sSigmaX[BLOCK_DIM+2*HALF_STENCIL][BLOCK_DIM+2*HALF_STENCIL]; 
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  
  int six=threadIdx.x+HALF_STENCIL;
  int siy=threadIdx.y+HALF_STENCIL;
  
  float zSigmaZ[2*HALF_STENCIL+1];
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   zSigmaZ[iz+1]=curSigmaZ0[j];
   zSigmaZ[iz+1+HALF_STENCIL]=curSigmaZ1[j];
  }

  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   
   zSigmaZ[0]=zSigmaZ[1];
   zSigmaZ[1]=zSigmaZ[2];
   zSigmaZ[2]=zSigmaZ[3];
   zSigmaZ[3]=zSigmaZ[4];
   zSigmaZ[4]=zSigmaZ[5];
   zSigmaZ[5]=zSigmaZ[6];
   zSigmaZ[6]=zSigmaZ[7];
   zSigmaZ[7]=zSigmaZ[8];
   zSigmaZ[8]=curSigmaZ2[j];
   
   __syncthreads();
   
   sSigmaX[six][siy]=curSigmaX1[j];
   
   if(threadIdx.x<HALF_STENCIL){
    int k=min(blockDim.x,nx-2*HALF_STENCIL-blockIdx.x*blockDim.x);
    sSigmaX[threadIdx.x][siy]=curSigmaX1[j-HALF_STENCIL];
	sSigmaX[six+k][siy]=curSigmaX1[j+k];
   }
   
   if(threadIdx.y<HALF_STENCIL){
    int k=min(blockDim.y,ny-2*HALF_STENCIL-blockIdx.y*blockDim.y);
    sSigmaX[six][threadIdx.y]=curSigmaX1[j-HALF_STENCIL*nx];
	sSigmaX[six][siy+k]=curSigmaX1[j+k*nx];
   }

   __syncthreads();

   float c33=v[j]*v[j];
   float c11=c33*(1.+2.*eps[j]);
   float sqrt12del=sqrt(1.+2.*del[j]);
   float c13=c33*sqrt12del;
  
   float tx=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six-1][siy]+sSigmaX[six+1][siy])
                                 +C2*(sSigmaX[six-2][siy]+sSigmaX[six+2][siy])
                                 +C3*(sSigmaX[six-3][siy]+sSigmaX[six+3][siy])
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))/dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))/dy2;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))/dz2;
   
   //update source wavefields
   nextSigmaX[j]=dt2*(c11*(tx+ty)+c13*tz)+2.*sSigmaX[six][siy]-prevSigmaX[j];
   nextSigmaZ[j]=dt2*(c13*(tx+ty)+c33*tz)+2.*zSigmaZ[HALF_STENCIL]-prevSigmaZ[j];
   
   //imaging conditions to compute gradient
   float gc11=curSigmaX1a[j]*(tx+ty);
   float gc33=curSigmaZ1a[j]*tz;
   float gc13=curSigmaX1a[j]*tz+curSigmaZ1a[j]*(tx+ty);
   
   gv[j]+=2.*v[j]*(gc11*(1.+2.*eps[j])+gc13*sqrt12del+gc33);
   geps[j]+=gc11*2.*c33;
   gdel[j]+=gc13*c33/sqrt12del;
  }
 } 

 return;
}

void gradient(float **gv,float **geps,float **gdel,int ndg,int b,int e,int k,float ***d_SigmaX,float ***d_SigmaZ,float ***d_SigmaXa,float ***d_SigmaZa,const int *nbuffSigma,const float *d_damping,float **d_v,float **d_eps,float **d_del,const int nbuffVEpsDel,float *wavelet,int souIndexBlock,int souBlock,float *data,int nr,const int *recIndex,int recBlock,int nx,int ny,int nz,int npad,int nt,float dx2,float dy2,float dz2,float dt2,cudaStream_t *stream,int gpu,int NGPU){
 dim3 block(BLOCK_DIM,BLOCK_DIM);
 dim3 grid((nx-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM,(ny-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM);
 
 int nb=nz/HALF_STENCIL;
 int roundlen=max(NGPU*(NUPDATE+3)+3,nb);
 int nround=(nt-2)/(NGPU*NUPDATE);
 
 for(int i=b;i<e;++i){
  int ib=(k-3-i)%roundlen;
  int iround=(k-3-i)/roundlen;
  if(ib>=0 && ib<nb && iround>=0 && iround<nround){
   int it=iround*NUPDATE*NGPU+gpu*NUPDATE+2+i;
  it=nt-1-it;
  if(ib==0){
   gradientKernelTopBlock<<<grid,block,0,*stream>>>(gv[(k-i)%ndg],geps[(k-i)%ndg],gdel[(k-i)%ndg],d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);

   forwardKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
//   adjointKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-3)%nbuffVEpsDel],d_v[(k-i-2)%nbuffVEpsDel],d_v[(k-i-1)%nbuffVEpsDel],d_eps[(k-i-3)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-1)%nbuffVEpsDel],d_del[(k-i-3)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],d_del[(k-i-1)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else if(ib==nb-1){
   gradientKernelBottomBlock<<<grid,block,0,*stream>>>(gv[(k-i)%ndg],geps[(k-i)%ndg],gdel[(k-i)%ndg],d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);

   forwardKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
//   adjointKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-3)%nbuffVEpsDel],d_v[(k-i-2)%nbuffVEpsDel],d_v[(k-i-1)%nbuffVEpsDel],d_eps[(k-i-3)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-1)%nbuffVEpsDel],d_del[(k-i-3)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],d_del[(k-i-1)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else{
   gradientKernel<<<grid,block,0,*stream>>>(gv[(k-i)%ndg],geps[(k-i)%ndg],gdel[(k-i)%ndg],d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);

   forwardKernel<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
//   adjointKernel<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-3)%nbuffVEpsDel],d_v[(k-i-2)%nbuffVEpsDel],d_v[(k-i-1)%nbuffVEpsDel],d_eps[(k-i-3)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-1)%nbuffVEpsDel],d_del[(k-i-3)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],d_del[(k-i-1)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  
  if(ib==souBlock){
   float source=dt2*wavelet[it+1];
   injectSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock);
//   injectDipoleSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock,nx*ny);
  }
 
  if(ib==recBlock){
   injectResidual<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,*stream>>>(data,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],nr,recIndex,dt2);
//   injectDipoleResidual<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,*stream>>>(data,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],nr,recIndex,dt2,nx*ny);
  }
  
  int iz=ib*HALF_STENCIL;
  if(iz<npad || iz+HALF_STENCIL-1>nz-npad) abcXYZ<<<grid,block,0,*stream>>>(ib,nx,ny,nz,npad,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_damping);
  else abcXY<<<grid,block,0,*stream>>>(ib,nx,ny,nz,npad,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_damping);
 }
 }
 
 return;
}

void gradientWavelet(float *gwavelet,int b,int e,int k,float ***d_SigmaXa,float ***d_SigmaZa,const int *nbuffSigma,const float *d_damping,float **d_v,float **d_eps,float **d_del,const int nbuffVEpsDel,int souIndexBlock,int souBlock,float *data,int nr,const int *recIndex,int recBlock,int nx,int ny,int nz,int npad,int nt,float dx2,float dy2,float dz2,float dt2,cudaStream_t *stream,int gpu,int NGPU){
 dim3 block(BLOCK_DIM,BLOCK_DIM);
 dim3 grid((nx-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM,(ny-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM);
 
 for(int i=b;i<e;++i){
  int nb=nz/HALF_STENCIL;
  int ib=(k-3-i)%nb;
  int it=(k-3-i)/nb*NUPDATE*NGPU+gpu*NUPDATE+2+i;
  it=nt-1-it;
  
  if(ib==0){
   forwardKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else if(ib==nb-1){
   forwardKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else{
   forwardKernel<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  
  if(ib==souBlock){ 
   extractAdjWfldAtSouLoc<<<1,1,0,*stream>>>(gwavelet,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],souIndexBlock,it);
  }
 
  if(ib==recBlock){
   injectResidual<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,*stream>>>(data,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],nr,recIndex,dt2);
//   injectDipoleResidual<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,*stream>>>(data,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],nr,recIndex,dt2,nx*ny);
  }
  
  int iz=ib*HALF_STENCIL;
  if(iz<npad || iz+HALF_STENCIL-1>nz-npad) abcXYZ<<<grid,block,0,*stream>>>(ib,nx,ny,nz,npad,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_damping);
  else abcXY<<<grid,block,0,*stream>>>(ib,nx,ny,nz,npad,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_damping);
 }
 
 return;
}

void sumGradientTime(float *gv,float *geps,float *gdel,float *h_gv,float *h_geps,float *h_gdel,size_t nElemBlock){
	#pragma omp parallel for num_threads(16)
	for(size_t i=0;i<nElemBlock;i++){
		gv[i]+=h_gv[i];
		geps[i]+=h_geps[i];
		gdel[i]+=h_gdel[i];
	}
	return;
}

__global__ void imagingKernel(float *image,float *nextSigmaX,float *nextSigmaZ,float *nextSigmaXa,float *nextSigmaZa,int nx,int ny){
 
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   image[j]+=(TWOTHIRD*nextSigmaX[j]+ONETHIRD*nextSigmaZ[j])*(TWOTHIRD*nextSigmaXa[j]+ONETHIRD*nextSigmaZa[j]);
  } 
 }

 return;
}

void imaging(float **d_image,int ndg,int b,int e,int k,float ***d_SigmaX,float ***d_SigmaZ,float ***d_SigmaXa,float ***d_SigmaZa,const int *nbuffSigma,const float *d_damping,float **d_v,float **d_eps,float **d_del,const int nbuffVEpsDel,float *wavelet,int souIndexBlock,int souBlock,float *data,int nr,const int *recIndex,int recBlock,int nx,int ny,int nz,int npad,int nt,float dx2,float dy2,float dz2,float dt2,cudaStream_t *stream,int gpu,int NGPU){
 dim3 block(BLOCK_DIM,BLOCK_DIM);
 dim3 grid((nx-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM,(ny-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM);
 
 int nb=nz/HALF_STENCIL;
 int roundlen=max(NGPU*(NUPDATE+3)+3,nb);
 int nround=(nt-2)/(NGPU*NUPDATE);
 
 for(int i=b;i<e;++i){
  int ib=(k-3-i)%roundlen;
  int iround=(k-3-i)/roundlen;
  if(ib>=0 && ib<nb && iround>=0 && iround<nround){
   int it=iround*NUPDATE*NGPU+gpu*NUPDATE+2+i;
  it=nt-1-it;
  if(ib==0){
   forwardKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
   
   forwardKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);

//   adjointKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-3)%nbuffVEpsDel],d_v[(k-i-2)%nbuffVEpsDel],d_v[(k-i-1)%nbuffVEpsDel],d_eps[(k-i-3)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-1)%nbuffVEpsDel],d_del[(k-i-3)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],d_del[(k-i-1)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else if(ib==nb-1){
   forwardKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);

   forwardKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);

//   adjointKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-3)%nbuffVEpsDel],d_v[(k-i-2)%nbuffVEpsDel],d_v[(k-i-1)%nbuffVEpsDel],d_eps[(k-i-3)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-1)%nbuffVEpsDel],d_del[(k-i-3)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],d_del[(k-i-1)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else{
   forwardKernel<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);

   forwardKernel<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);

//   adjointKernel<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-3)%nbuffVEpsDel],d_v[(k-i-2)%nbuffVEpsDel],d_v[(k-i-1)%nbuffVEpsDel],d_eps[(k-i-3)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-1)%nbuffVEpsDel],d_del[(k-i-3)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],d_del[(k-i-1)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  
  if(ib==souBlock){
   float source=dt2*wavelet[it+1];
   injectSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock);
//   injectDipoleSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock,nx*ny);
  }
 
  if(ib==recBlock){
   injectResidual<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,*stream>>>(data,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],nr,recIndex,dt2);
//   injectDipoleResidual<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,*stream>>>(data,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],nr,recIndex,dt2,nx*ny);
  }
  
  int iz=ib*HALF_STENCIL;
  if(iz<npad || iz+HALF_STENCIL-1>nz-npad) abcXYZ<<<grid,block,0,*stream>>>(ib,nx,ny,nz,npad,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_damping);
  else abcXY<<<grid,block,0,*stream>>>(ib,nx,ny,nz,npad,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_damping);

  imagingKernel<<<grid,block,0,*stream>>>(d_image[(k-i)%ndg],d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],nx,ny);
 }
 }
 
 return;
}

void sumImageTime(float *image,float *h_image,size_t nElemBlock){
	#pragma omp parallel for num_threads(16)
	for(size_t i=0;i<nElemBlock;i++){
		image[i]+=h_image[i];
	}
	return;
}

__global__ void extendedImagingKernel(float *image,float *nextSigmaX,float *nextSigmaZ,float *nextSigmaXa,float *nextSigmaZa,int nx,int ny){
 int nElemBlock=HALF_STENCIL*nx*ny;
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix>=NLAG && ix<nx-NLAG && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   float temp=TWOTHIRD*nextSigmaX[j]+ONETHIRD*nextSigmaZ[j];
   for(int lag=-NLAG;lag<=NLAG;lag++){
    image[j+(lag+NLAG)*nElemBlock]+=temp*(TWOTHIRD*nextSigmaXa[j+lag]+ONETHIRD*nextSigmaZa[j+lag]);
   }
  } 
 }

 return;
}

void extendedImaging(float **image,int ndg,int b,int e,int k,float ***d_SigmaX,float ***d_SigmaZ,float ***d_SigmaXa,float ***d_SigmaZa,const int *nbuffSigma,const float *d_damping,float **d_v,float **d_eps,float **d_del,const int nbuffVEpsDel,float *wavelet,int souIndexBlock,int souBlock,float *data,int nr,const int *recIndex,int recBlock,int nx,int ny,int nz,int npad,int nt,float dx2,float dy2,float dz2,float dt2,cudaStream_t *stream,int gpu,int NGPU){
 dim3 block(BLOCK_DIM,BLOCK_DIM);
 dim3 grid((nx-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM,(ny-2*HALF_STENCIL+BLOCK_DIM-1)/BLOCK_DIM);
 
 int nb=nz/HALF_STENCIL;
 int roundlen=max(NGPU*(NUPDATE+3)+3,nb);
 int nround=(nt-2)/(NGPU*NUPDATE);
 
 for(int i=b;i<e;++i){
  int ib=(k-3-i)%roundlen;
  int iround=(k-3-i)/roundlen;
  if(ib>=0 && ib<nb && iround>=0 && iround<nround){
   int it=iround*NUPDATE*NGPU+gpu*NUPDATE+2+i;
  it=nt-1-it;
  if(ib==0){
   forwardKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);

   forwardKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
//   adjointKernelTopBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-3)%nbuffVEpsDel],d_v[(k-i-2)%nbuffVEpsDel],d_v[(k-i-1)%nbuffVEpsDel],d_eps[(k-i-3)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-1)%nbuffVEpsDel],d_del[(k-i-3)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],d_del[(k-i-1)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else if(ib==nb-1){
   forwardKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);

   forwardKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
//   adjointKernelBottomBlock<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-3)%nbuffVEpsDel],d_v[(k-i-2)%nbuffVEpsDel],d_v[(k-i-1)%nbuffVEpsDel],d_eps[(k-i-3)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-1)%nbuffVEpsDel],d_del[(k-i-3)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],d_del[(k-i-1)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  else{
   forwardKernel<<<grid,block,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaX[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaX[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaX[i][(k-3)%nbuffSigma[i]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZ[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZ[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);

   forwardKernel<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
//   adjointKernel<<<grid,block,0,*stream>>>(d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaXa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaXa[i][(k-3)%nbuffSigma[i]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-3)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+1][(k-1)%nbuffSigma[i+1]],d_SigmaZa[i][(k-3)%nbuffSigma[i]],d_v[(k-i-3)%nbuffVEpsDel],d_v[(k-i-2)%nbuffVEpsDel],d_v[(k-i-1)%nbuffVEpsDel],d_eps[(k-i-3)%nbuffVEpsDel],d_eps[(k-i-2)%nbuffVEpsDel],d_eps[(k-i-1)%nbuffVEpsDel],d_del[(k-i-3)%nbuffVEpsDel],d_del[(k-i-2)%nbuffVEpsDel],d_del[(k-i-1)%nbuffVEpsDel],nx,ny,dx2,dy2,dz2,dt2);
  }
  
  if(ib==souBlock){
   float source=dt2*wavelet[it+1];
   injectSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock);
//   injectDipoleSource<<<1,1,0,*stream>>>(d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],source,souIndexBlock,nx*ny);
  }
 
  if(ib==recBlock){
   injectResidual<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,*stream>>>(data,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],nr,recIndex,dt2);
//   injectDipoleResidual<<<(nr+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,*stream>>>(data,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],nr,recIndex,dt2,nx*ny);
  }
  
  int iz=ib*HALF_STENCIL;
  if(iz<npad || iz+HALF_STENCIL-1>nz-npad) abcXYZ<<<grid,block,0,*stream>>>(ib,nx,ny,nz,npad,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_damping);
  else abcXY<<<grid,block,0,*stream>>>(ib,nx,ny,nz,npad,d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+1][(k-2)%nbuffSigma[i+1]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+1][(k-2)%nbuffSigma[i+1]],d_damping);

  extendedImagingKernel<<<grid,block,0,*stream>>>(image[(k-i)%ndg],d_SigmaX[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZ[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaXa[i+2][(k-1)%nbuffSigma[i+2]],d_SigmaZa[i+2][(k-1)%nbuffSigma[i+2]],nx,ny);
 }
 }
 
 return;
}

