#include <cstring>
#include <cstdlib>
#include <cstdio>

#include "mylib.h"
#include "boundary.h"

using namespace std;

void getBoundary(float *boundary,const float *v,int nx,int ny,int nz,int npad){
    size_t currentB=0,currentV=0,n=nx*ny*npad;
    memcpy(boundary+currentB,v+currentV,n*sizeof(float));
    currentB+=n;
    currentV+=n;
    for(int iz=0;iz<nz-2*npad;iz++){
        for(int iy=0;iy<npad;iy++){
            n=nx;
            memcpy(boundary+currentB,v+currentV,n*sizeof(float));
            currentB+=n;
            currentV+=n;
        }
        for(int iy=0;iy<ny-2*npad;iy++){
            n=npad;
            memcpy(boundary+currentB,v+currentV,n*sizeof(float));
            currentB+=n;
            currentV=currentV+nx-npad;
            memcpy(boundary+currentB,v+currentV,n*sizeof(float));
            currentB+=n;
            currentV+=n;
        }
        for(int iy=0;iy<npad;iy++){
            n=nx;
            memcpy(boundary+currentB,v+currentV,n*sizeof(float));
            currentB+=n;
            currentV+=n;
        }
    }
    n=nx*ny*npad;
    memcpy(boundary+currentB,v+currentV,n*sizeof(float));
//    currentB+=n;
//    currentV+=n;
//    long long nboundary=nx*ny*nz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);
//    fprintf(stderr,"currentB %u currentV %u\n",currentB,currentV);
//    fprintf(stderr,"nboundary %u nxyz %u\n",nboundary,nx*ny*nz);
    return;
}

void putBoundary(const float *boundary,float *v,int nx,int ny,int nz,int npad){
    size_t currentB=0,currentV=0,n=nx*ny*npad;
    memcpy(v+currentV,boundary+currentB,n*sizeof(float));
    currentB+=n;
    currentV+=n;
    for(int iz=0;iz<nz-2*npad;iz++){
        for(int iy=0;iy<npad;iy++){
            n=nx;
            memcpy(v+currentV,boundary+currentB,n*sizeof(float));
            currentB+=n;
            currentV+=n;
        }
        for(int iy=0;iy<ny-2*npad;iy++){
            n=npad;
            memcpy(v+currentV,boundary+currentB,n*sizeof(float));
            currentB+=n;
            currentV=currentV+nx-npad;
            memcpy(v+currentV,boundary+currentB,n*sizeof(float));
            currentB+=n;
            currentV+=n;
        }
        for(int iy=0;iy<npad;iy++){
            n=nx;
            memcpy(v+currentV,boundary+currentB,n*sizeof(float));
            currentB+=n;
            currentV+=n;
        }
    }
    n=nx*ny*npad;
    memcpy(v+currentV,boundary+currentB,n*sizeof(float));
//    currentB+=n;
//    currentV+=n;
//    long long nboundary=nx*ny*nz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);
//    fprintf(stderr,"currentB %u currentV %u\n",currentB,currentV);
//    fprintf(stderr,"nboundary %u nxyz %u\n",nboundary,nx*ny*nz);
    return;
}

void zeroBoundary(float *v,int nx,int ny,int nz,int npad){
    size_t currentV=0,n=nx*ny*npad;
    memset(v+currentV,0,n*sizeof(float));
    currentV+=n;
    for(int iz=0;iz<nz-2*npad;iz++){
        for(int iy=0;iy<npad;iy++){
            n=nx;
            memset(v+currentV,0,n*sizeof(float));
            currentV+=n;
        }
        for(int iy=0;iy<ny-2*npad;iy++){
            n=npad;
            memset(v+currentV,0,n*sizeof(float));
            currentV=currentV+nx-npad;
            memset(v+currentV,0,n*sizeof(float));
            currentV+=n;
        }
        for(int iy=0;iy<npad;iy++){
            n=nx;
            memset(v+currentV,0,n*sizeof(float));
            currentV+=n;
        }
    }
    n=nx*ny*npad;
    memset(v+currentV,0,n*sizeof(float));
//    currentB+=n;
//    currentV+=n;
//    long long nboundary=nx*ny*nz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);
//    fprintf(stderr,"currentB %u currentV %u\n",currentB,currentV);
//    fprintf(stderr,"nboundary %u nxyz %u\n",nboundary,nx*ny*nz);
    return;
}

void pad1d(float *vout,float *vin,int nx,int npad){
    int nx0=nx-2*npad;
    set(vout,vin[0],npad);
    memcpy(vout+npad,vin,nx0*sizeof(float));
    set(vout+npad+nx0,vin[nx0-1],npad);
    return;
}

void pad2d(float *vout,float *vin,int nx,int ny,int npad){
    int nx0=nx-2*npad,ny0=ny-2*npad;
    float *temp=new float[nx];
    
    pad1d(temp,vin,nx,npad);
    #pragma omp parallel for
    for(int iy=0;iy<=npad;iy++) memcpy(vout+iy*nx,temp,nx*sizeof(float));
    
    for(int iy=npad+1;iy<npad+ny0;iy++){
        pad1d(temp,vin+(iy-npad)*nx0,nx,npad);
        memcpy(vout+iy*nx,temp,nx*sizeof(float));
    }

    #pragma omp parallel for
    for(int iy=npad+ny0;iy<ny;iy++) memcpy(vout+iy*nx,temp,nx*sizeof(float));
    
    delete []temp;
    return;
}

void pad3d(float *vout,float *vin,int nx,int ny,int nz,int npad){
    int nx0=nx-2*npad,ny0=ny-2*npad,nz0=nz-2*npad;
    int nxy=nx*ny,nxy0=nx0*ny0;
    float *temp=new float[nxy];
    
    pad2d(temp,vin,nx,ny,npad);
    #pragma omp parallel for
    for(int iz=0;iz<=npad;iz++) memcpy(vout+iz*nxy,temp,nxy*sizeof(float));
    
    for(int iz=npad+1;iz<npad+nz0;iz++){
        pad2d(temp,vin+(iz-npad)*nxy0,nx,ny,npad);
        memcpy(vout+iz*nxy,temp,nxy*sizeof(float));
    }

    #pragma omp parallel for
    for(int iz=npad+nz0;iz<nz;iz++) memcpy(vout+iz*nxy,temp,nxy*sizeof(float));
    
    delete []temp;
    return;
}
