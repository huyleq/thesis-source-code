#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <string>

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>

#include "myio.h"
#include "mylib.h"
#include "laplacian3d.h"

#define HALF_STENCIL 4
#define DAMPER 0.95
#define PI 3.14159265359

using namespace std;

class Block{
    public:
    Block(int beginX,int endX,int beginY,int endY,int beginZ,int endZ):_beginX(beginX),_endX(endX),_beginY(beginY),_endY(endY),_beginZ(beginZ),_endZ(endZ){};
    int _beginX,_endX,_beginY,_endY,_beginZ,_endZ;
};

void abc(float *sigma,const float *damping,int nx,int ny,int nz,int npad){
    #pragma omp parallel for num_threads(16)
    for(int iz=0;iz<npad;iz++){
        for(int iy=0;iy<ny;iy++){
            for(int ix=0;ix<nx;ix++){
                sigma[ix+iy*nx+iz*nx*ny]*=damping[iz]; 
                sigma[ix+iy*nx+(nz-1-iz)*nx*ny]*=damping[iz]; 
            }
        }
    }
    #pragma omp parallel for num_threads(16)
    for(int iz=0;iz<nz;iz++){
        for(int iy=0;iy<npad;iy++){
            for(int ix=0;ix<nx;ix++){
                sigma[ix+iy*nx+iz*nx*ny]*=damping[iy]; 
                sigma[ix+(ny-1-iy)*nx+iz*nx*ny]*=damping[iy]; 
            }
        }
    }
    #pragma omp parallel for num_threads(16)
    for(int iz=0;iz<nz;iz++){
        for(int iy=0;iy<ny;iy++){
            for(int ix=0;ix<npad;ix++){
                sigma[ix+iy*nx+iz*nx*ny]*=damping[ix]; 
                sigma[nx-1-ix+iy*nx+iz*nx*ny]*=damping[ix]; 
            }
        }
    }
    return;
}

int main(int argc,char **argv){
    myio_init(argc,argv);

    int nx,ny,nz,nt,npad;
    float ox,oy,oz,ot,dx,dy,dz,dt;

    get_param("n1",nx,"o1",ox,"d1",dx);
    get_param("n2",ny,"o2",oy,"d2",dy);
    get_param("n3",nz,"o3",oz,"d3",dz);
    
    get_param("npad",npad);
    get_param("nt",nt,"ot",ot,"dt",dt);
    
    long long nxy=nx*ny;
    long long nxyz=nxy*nz;
    
    float *wavelet=new float[nt]();
    read("wavelet",wavelet,nt);
    
    float samplingRate;
    get_param("samplingRate",samplingRate);
    int samplingTimeStep=std::round(samplingRate/dt);
    int nnt=(nt-1)/samplingTimeStep+1;

    float *wavefieldX=new float[nx*nz*nnt]();
    float *wavefieldZ=new float[nx*nz*nnt]();

    float *damping=new float[npad];
    for(int i=0;i<npad;i++) damping[i]=DAMPER+(1.-DAMPER)*cos(PI*(npad-i)/npad);

    float dx2=dx*dx,dy2=dy*dy,dz2=dz*dz,dt2=dt*dt;

    float *prevSigmaX=new float[nxyz];
    float *curSigmaX=new float[nxyz];
    float *prevSigmaZ=new float[nxyz];
    float *curSigmaZ=new float[nxyz];

    int blockSizeX,blockSizeY,blockSizeZ;
    get_param("blockSizeX",blockSizeX,"blockSizeY",blockSizeY,"blockSizeZ",blockSizeZ);

    vector<int> beginX(1,HALF_STENCIL),endX(1,HALF_STENCIL+blockSizeX);
    int nleft=nx-2*HALF_STENCIL-blockSizeX,i=0;
    while(nleft>0){
        int blockSize=min(nleft,blockSizeX);
        beginX.push_back(endX[i]);
        endX.push_back(endX[i]+blockSize);
        ++i;
        nleft-=blockSize;
    }
    
    vector<int> beginY(1,HALF_STENCIL),endY(1,HALF_STENCIL+blockSizeY);
    nleft=ny-2*HALF_STENCIL-blockSizeY;i=0;
    while(nleft>0){
        int blockSize=min(nleft,blockSizeY);
        beginY.push_back(endY[i]);
        endY.push_back(endY[i]+blockSize);
        ++i;
        nleft-=blockSize;
    }
    
    vector<int> beginZ(1,HALF_STENCIL),endZ(1,HALF_STENCIL+blockSizeZ);
    nleft=nz-2*HALF_STENCIL-blockSizeZ;i=0;
    while(nleft>0){
        int blockSize=min(nleft,blockSizeZ);
        beginZ.push_back(endZ[i]);
        endZ.push_back(endZ[i]+blockSize);
        ++i;
        nleft-=blockSize;
    }
    
    vector<Block> blocks;
    for(int iz=0;iz<beginZ.size();iz++){
        for(int iy=0;iy<beginY.size();iy++){
            for(int ix=0;ix<beginX.size();ix++){
                blocks.push_back(Block(beginX[ix],endX[ix],beginY[iy],endY[iy],beginZ[iz],endZ[iz]));
            }
        }
    }

    float *v=new float[nxyz]; set(v,2000.,nxyz);
    float *eps=new float[nxyz]();
    float *del=new float[nxyz]();
    
    memset(curSigmaX,0,nxyz*sizeof(float));
    memset(prevSigmaX,0,nxyz*sizeof(float));
    memset(curSigmaZ,0,nxyz*sizeof(float));
    memset(prevSigmaZ,0,nxyz*sizeof(float));

    int souIndex=nx/2+ny/2*nx+nz/2*nxy;
    float s=wavelet[0]*dt2;
    curSigmaX[souIndex]+=s;
    curSigmaZ[souIndex]+=s;
    
    for(int it=2;it<nt;it++){
        tbb::parallel_for(tbb::blocked_range<int>(0,(int)blocks.size()),
   	                   [&](const tbb::blocked_range<int>&r){
       	    for(int ib=r.begin();ib!=r.end();++ib){
                for(int iz=blocks[ib]._beginZ;iz<blocks[ib]._endZ;iz++){
                    for(int iy=blocks[ib]._beginY;iy<blocks[ib]._endY;iy++){
                        int i=blocks[ib]._beginX+iy*nx+iz*nxy;
   	                    forward(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,v+i,eps+i,del+i,prevSigmaX+i,curSigmaX+i,prevSigmaZ+i,curSigmaZ+i);
                    }                                                           
                }                                                                        
            }                                                                            
        });                                                                     

        float *pt=prevSigmaX;prevSigmaX=curSigmaX;curSigmaX=pt;
        pt=prevSigmaZ;prevSigmaZ=curSigmaZ;curSigmaZ=pt;
        
        s=wavelet[it-1]*dt2;
        curSigmaX[souIndex]+=s;
        curSigmaZ[souIndex]+=s;
        
        abc(prevSigmaX,damping,nx,ny,nz,npad);
        abc(prevSigmaZ,damping,nx,ny,nz,npad);
        abc(curSigmaX,damping,nx,ny,nz,npad);
        abc(curSigmaZ,damping,nx,ny,nz,npad);

        if(it%samplingTimeStep==0){
            int it1=it/samplingTimeStep;
            #pragma omp parallel for num_threads(16)
            for(int iz=0;iz<nz;iz++){
                memcpy(wavefieldX+iz*nx+it1*nx*nz,curSigmaX+ny/2*nx+iz*nxy,nx*sizeof(float));
                memcpy(wavefieldZ+iz*nx+it1*nx*nz,curSigmaZ+ny/2*nx+iz*nxy,nx*sizeof(float));
            }
        }
    }

    write("wavefieldX",wavefieldX,nx*nz*nnt);
    to_header("wavefieldX","n1",nx,"o1",ox,"d1",dx);
    to_header("wavefieldX","n2",nz,"o2",oz,"d2",dz);
    to_header("wavefieldX","n3",nnt,"o3",ot,"d3",samplingRate);
    
    write("wavefieldZ",wavefieldZ,nx*nz*nnt);
    to_header("wavefieldZ","n1",nx,"o1",ox,"d1",dx);
    to_header("wavefieldZ","n2",nz,"o2",oz,"d2",dz);
    to_header("wavefieldZ","n3",nnt,"o3",ot,"d3",samplingRate);
    
    delete []wavelet;delete []damping;
    delete []prevSigmaX;delete []curSigmaX;
    delete []prevSigmaZ;delete []curSigmaZ;
    delete []wavefieldX;delete []wavefieldZ;
    
    myio_close();
    return 0;
}

