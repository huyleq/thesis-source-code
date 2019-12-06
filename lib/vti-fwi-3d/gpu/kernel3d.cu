#include "wave3d.h"

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

__global__ void forwardKernelTopBlock(float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *c11,float *c13,float *c33,int nx,int ny,float dt2dx2,float dt2dy2,float dt2dz2){
 
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

__global__ void forwardKernelBottomBlock(float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *c11,float *c13,float *c33,int nx,int ny,float dt2dx2,float dt2dy2,float dt2dz2){
 
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

__global__ void gradientKernel(float *gc11,float *gc13,float *gc33,float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *curLambdaX,float *curLambdaZ,float *c11,float *c13,float *c33,int nx,int ny,float dx2,float dy2,float dz2,float dt2){
 
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
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))/dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))/dy2;
   ty+=tx;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))/dz2;
   
   nextSigmaX[j]=dt2*(c11[j]*ty+c13[j]*tz)+2.*sSigmaX[six][siy]-prevSigmaX[j];
   nextSigmaZ[j]=dt2*(c13[j]*ty+c33[j]*tz)+2.*zSigmaZ[HALF_STENCIL]-prevSigmaZ[j];
   
   //imaging conditions to compute gradient
   gc11[j]+=curLambdaX[j]*ty;
   gc33[j]+=curLambdaZ[j]*tz;
   gc13[j]+=curLambdaX[j]*tz+curLambdaZ[j]*ty;
  }
 } 

 return;
}

__global__ void gradientKernelTopBlock(float *gc11,float *gc13,float *gc33,float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *curLambdaX,float *curLambdaZ,float *c11,float *c13,float *c33,int nx,int ny,float dx2,float dy2,float dz2,float dt2){
 
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

   float tx=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six-1][siy]+sSigmaX[six+1][siy])
                                 +C2*(sSigmaX[six-2][siy]+sSigmaX[six+2][siy])
                                 +C3*(sSigmaX[six-3][siy]+sSigmaX[six+3][siy])
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))/dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))/dy2;
   ty+=tx;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))/dz2;
   
   nextSigmaX[j]=dt2*(c11[j]*ty+c13[j]*tz)+2.*sSigmaX[six][siy]-prevSigmaX[j];
   nextSigmaZ[j]=dt2*(c13[j]*ty+c33[j]*tz)+2.*zSigmaZ[HALF_STENCIL]-prevSigmaZ[j];
   
   //imaging conditions to compute gradient
   gc11[j]+=curLambdaX[j]*ty;
   gc33[j]+=curLambdaZ[j]*tz;
   gc13[j]+=curLambdaX[j]*tz+curLambdaZ[j]*ty;
  }
 } 

 return;
}

__global__ void gradientKernelBottomBlock(float *gc11,float *gc13,float *gc33,float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *curLambdaX,float *curLambdaZ,float *c11,float *c13,float *c33,int nx,int ny,float dx2,float dy2,float dz2,float dt2){
 
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

   float tx=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six-1][siy]+sSigmaX[six+1][siy])
                                 +C2*(sSigmaX[six-2][siy]+sSigmaX[six+2][siy])
                                 +C3*(sSigmaX[six-3][siy]+sSigmaX[six+3][siy])
                                 +C4*(sSigmaX[six-4][siy]+sSigmaX[six+4][siy]))/dx2;
   float ty=(C0*sSigmaX[six][siy]+C1*(sSigmaX[six][siy-1]+sSigmaX[six][siy+1])
                                 +C2*(sSigmaX[six][siy-2]+sSigmaX[six][siy+2])
                                 +C3*(sSigmaX[six][siy-3]+sSigmaX[six][siy+3])
                                 +C4*(sSigmaX[six][siy-4]+sSigmaX[six][siy+4]))/dy2;
   ty+=tx;
   float tz=(C0*zSigmaZ[HALF_STENCIL]+C1*(zSigmaZ[HALF_STENCIL-1]+zSigmaZ[HALF_STENCIL+1])
                                     +C2*(zSigmaZ[HALF_STENCIL-2]+zSigmaZ[HALF_STENCIL+2])
                                     +C3*(zSigmaZ[HALF_STENCIL-3]+zSigmaZ[HALF_STENCIL+3])
                                     +C4*(zSigmaZ[HALF_STENCIL-4]+zSigmaZ[HALF_STENCIL+4]))/dz2;
   
   nextSigmaX[j]=dt2*(c11[j]*ty+c13[j]*tz)+2.*sSigmaX[six][siy]-prevSigmaX[j];
   nextSigmaZ[j]=dt2*(c13[j]*ty+c33[j]*tz)+2.*zSigmaZ[HALF_STENCIL]-prevSigmaZ[j];
   
   //imaging conditions to compute gradient
   gc11[j]+=curLambdaX[j]*ty;
   gc33[j]+=curLambdaZ[j]*tz;
   gc13[j]+=curLambdaX[j]*tz+curLambdaZ[j]*ty;
  }
 } 

 return;
}

__global__ void imagingKernel(float *image,float *curSigmaX,float *curSigmaZ,float *curLambdaX,float *curLambdaZ,int nx,int ny){
 int ix=threadIdx.x+blockIdx.x*blockDim.x+HALF_STENCIL;
 int iy=threadIdx.y+blockIdx.y*blockDim.y+HALF_STENCIL;

 if(ix<nx-HALF_STENCIL && iy<ny-HALF_STENCIL){
  int i=ix+iy*nx;
  for(int iz=0;iz<HALF_STENCIL;++iz){
   int j=i+iz*nx*ny;
   image[j]+=(TWOTHIRD*curSigmaX[j]+ONETHIRD*curSigmaZ[j])*(TWOTHIRD*curLambdaX[j]+ONETHIRD*curLambdaZ[j]);
  } 
 }
 return;
}

