#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

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

    from_header("v","n1",nx,"o1",ox,"d1",dx);
    from_header("v","n2",ny,"o2",oy,"d2",dy);
    from_header("v","n3",nz,"o3",oz,"d3",dz);
    get_param("npad",npad);
    get_param("nt",nt,"ot",ot,"dt",dt);
    
    long long nxy=nx*ny;
    long long nxz=nx*nz;
    long long nxyz=nxy*nz;
    
    float *wavelet=new float[nt]();
    read("wavelet",wavelet,nt);

    float samplingRate;
    get_param("samplingRate",samplingRate);
    int samplingTimeStep=std::round(samplingRate/dt);
    int nnt=(nt-1)/samplingTimeStep+1;

    float *damping=new float[npad];
    for(int i=0;i<npad;i++) damping[i]=DAMPER+(1.-DAMPER)*cos(PI*(npad-i)/npad);

    float dx2=dx*dx,dy2=dy*dy,dz2=dz*dz,dt2=dt*dt;

    float *v=new float[nxyz];
    read("v",v,nxyz);
    float *eps=new float[nxyz];
    read("eps",eps,nxyz);
    float *del=new float[nxyz];
    read("del",del,nxyz);
    
//    #pragma omp parallel for num_threads(16)
//    for(size_t i=0;i<nxyz;i++){
//        v[i]=v[i]*v[i];
//        eps[i]=v[i]*(1.+2.*eps[i]);
//        del[i]=v[i]*sqrt(1.+2.*del[i]);
//    }
//    float *c11=eps,*c13=del,*c33=v;
    
    int ns,nr;
    from_header("souloc","n2",ns);
    float *souloc=new float[5*ns];
    read("souloc",souloc,5*ns);
    
    from_header("recloc","n2",nr);
    float *recloc=new float[3*nr];
    read("recloc",recloc,3*nr);
    
    float *prevSigmaX=new float[nxyz];
    float *curSigmaX=new float[nxyz];
    float *prevSigmaZ=new float[nxyz];
    float *curSigmaZ=new float[nxyz];

    float *prevSigmaXa=new float[nxyz];
    float *curSigmaXa=new float[nxyz];
    float *prevSigmaZa=new float[nxyz];
    float *curSigmaZa=new float[nxyz];
    
    float *image0=new float[nxyz]();

//    float *forwardSouWavefield=new float[nx*nz*nt]();
//    float *backwardSouWavefield=new float[nx*nz*nt]();
//    float *backwardRecWavefield=new float[nx*nz*nt]();
//    float *backwardImageWavefield=new float[nx*nz*nt]();
//    float *a=new float[nxz]();

    vector<int> beginX(1,HALF_STENCIL),endX(1,HALF_STENCIL+BLOCKSIZE);
    int nleft=nx-2*HALF_STENCIL-BLOCKSIZE,i=0;
    while(nleft>0){
        int blockSize=min(nleft,BLOCKSIZE);
        beginX.push_back(endX[i]);
        endX.push_back(endX[i]+blockSize);
        ++i;
        nleft-=blockSize;
    }
    
    vector<int> beginY(1,HALF_STENCIL),endY(1,HALF_STENCIL+BLOCKSIZE);
    nleft=ny-2*HALF_STENCIL-BLOCKSIZE;i=0;
    while(nleft>0){
        int blockSize=min(nleft,BLOCKSIZE);
        beginY.push_back(endY[i]);
        endY.push_back(endY[i]+blockSize);
        ++i;
        nleft-=blockSize;
    }
    
    vector<int> beginZ(1,HALF_STENCIL),endZ(1,HALF_STENCIL+BLOCKSIZE);
    nleft=nz-2*HALF_STENCIL-BLOCKSIZE;i=0;
    while(nleft>0){
        int blockSize=min(nleft,BLOCKSIZE);
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

    for(int is=0;is<ns;is++){
        fprintf(stderr,"shot %d\n",is);
	    int souIndexX=(souloc[5*is]-ox)/dx;
	    int souIndexY=(souloc[5*is+1]-oy)/dy;
	    int souIndexZ=(souloc[5*is+2]-oz)/dz;
	    int souIndex=souIndexX+souIndexY*nx+souIndexZ*nxy;

        memset(curSigmaX,0,nxyz*sizeof(float));
        memset(prevSigmaX,0,nxyz*sizeof(float));
        memset(curSigmaZ,0,nxyz*sizeof(float));
        memset(prevSigmaZ,0,nxyz*sizeof(float));

        float s=wavelet[0]*dt2;
        curSigmaX[souIndex]+=s;
        curSigmaZ[souIndex]+=s;
//        curSigmaX[souIndex+nxy]-=s;
//        curSigmaZ[souIndex+nxy]-=s;
        
//        #pragma omp parallel for num_threads(16)
//        for(int iz=0;iz<nz;iz++){
//            for(int ix=0;ix<nx;ix++){
//                int i=ix+ny/2*nx+iz*nxy;
//                a[ix+iz*nx]=(2.*curSigmaX[i]+curSigmaZ[i])/3.;
//            }
//        }
//        memcpy(forwardSouWavefield+nxz,a,nxz*sizeof(float));

        for(int it=2;it<nt;it++){
            tbb::parallel_for(tbb::blocked_range<int>(0,(int)blocks.size()),
       	                   [&](const tbb::blocked_range<int>&r){
           	    for(int ib=r.begin();ib!=r.end();++ib){
                    for(int iz=blocks[ib]._beginZ;iz<blocks[ib]._endZ;iz++){
                        for(int iy=blocks[ib]._beginY;iy<blocks[ib]._endY;iy++){
                            int i=blocks[ib]._beginX+iy*nx+iz*nxy;
       	                    forward(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,v+i,eps+i,del+i,prevSigmaX+i,curSigmaX+i,prevSigmaZ+i,curSigmaZ+i);
//       	                    forwardCij(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,c11+i,c13+i,c33+i,prevSigmaX+i,curSigmaX+i,prevSigmaZ+i,curSigmaZ+i);
                        }                                                           
                    }                                                                        
                }                                                                            
            });                                                                     
    
            float *pt=prevSigmaX;prevSigmaX=curSigmaX;curSigmaX=pt;
            pt=prevSigmaZ;prevSigmaZ=curSigmaZ;curSigmaZ=pt;
    
            s=wavelet[it-1]*dt2;
            curSigmaX[souIndex]+=s;
            curSigmaZ[souIndex]+=s;
//            curSigmaX[souIndex+nxy]-=s;
//            curSigmaZ[souIndex+nxy]-=s;

//            #pragma omp parallel for num_threads(16)
//            for(int iz=0;iz<nz;iz++){
//                for(int ix=0;ix<nx;ix++){
//                    int i=ix+ny/2*nx+iz*nxy;
//                    a[ix+iz*nx]=(2.*curSigmaX[i]+curSigmaZ[i])/3.;
//                }
//                memcpy(forwardSouWavefield+it*nxz,a,nxz*sizeof(float));
//            }
        }
            
        float *pt=prevSigmaX;prevSigmaX=curSigmaX;curSigmaX=pt;
        pt=prevSigmaZ;prevSigmaZ=curSigmaZ;curSigmaZ=pt;
    
        memset(curSigmaXa,0,nxyz*sizeof(float));
        memset(prevSigmaXa,0,nxyz*sizeof(float));
        memset(curSigmaZa,0,nxyz*sizeof(float));
        memset(prevSigmaZa,0,nxyz*sizeof(float));

        int nr1=souloc[5*is+3];
        int *recIndex=new int[nr1];
        int irbegin=souloc[5*is+4];
        float *data=new float[nnt*nr1];
        read("data",data,nnt*nr1,nnt*irbegin);

        #pragma omp parallel for num_threads(16)
        for(int ir=0;ir<nr1;ir++){
            int ir1=irbegin+ir;
            int recIndexX=(recloc[3*ir1]-ox)/dx;
            int recIndexY=(recloc[3*ir1+1]-oy)/dy;
            int recIndexZ=(recloc[3*ir1+2]-oz)/dz;
            recIndex[ir]=recIndexX+recIndexY*nx+recIndexZ*nxy;
            float temp=dt2*data[(nnt-1)+ir*nnt];
            curSigmaXa[recIndex[ir]]+=2./3.*temp;
            curSigmaZa[recIndex[ir]]+=1./3.*temp;
//            curSigmaXa[recIndex[ir]+nxy]-=2./3.*temp;
//            curSigmaZa[recIndex[ir]+nxy]-=1./3.*temp;
        }

        #pragma omp parallel for num_threads(16)
        for(size_t i=0;i<nxyz;i++){
            image0[i]+=(2.*curSigmaX[i]+curSigmaZ[i])*(2.*curSigmaXa[i]+curSigmaZa[i])/9.;
        }

//        #pragma omp parallel for num_threads(16)
//        for(int iz=0;iz<nz;iz++){
//            for(int ix=0;ix<nx;ix++){
//                int i=ix+ny/2*nx+iz*nxy;
//                a[ix+iz*nx]=(2.*curSigmaX[i]+curSigmaZ[i])*(2.*curSigmaXa[i]+curSigmaZa[i])/9.;
//            }
//        }
//        memcpy(backwardImageWavefield+nxz,a,nxz*sizeof(float));
//
//        #pragma omp parallel for num_threads(16)
//        for(int iz=0;iz<nz;iz++){
//            for(int ix=0;ix<nx;ix++){
//                int i=ix+ny/2*nx+iz*nxy;
//                a[ix+iz*nx]=(2.*prevSigmaX[i]+prevSigmaZ[i])/3.;
//            }
//        }
//        memcpy(backwardSouWavefield,a,nxz*sizeof(float));
//
//        #pragma omp parallel for num_threads(16)
//        for(int iz=0;iz<nz;iz++){
//            for(int ix=0;ix<nx;ix++){
//                int i=ix+ny/2*nx+iz*nxy;
//                a[ix+iz*nx]=(2.*curSigmaX[i]+curSigmaZ[i])/3.;
//            }
//        }
//        memcpy(backwardSouWavefield+nxz,a,nxz*sizeof(float));
//
//        #pragma omp parallel for num_threads(16)
//        for(int iz=0;iz<nz;iz++){
//            for(int ix=0;ix<nx;ix++){
//                int i=ix+ny/2*nx+iz*nxy;
//                a[ix+iz*nx]=(2.*curSigmaXa[i]+curSigmaZa[i])/3.;
//            }
//        }
//        memcpy(backwardRecWavefield+nxz,a,nxz*sizeof(float));

        for(int it=2;it<nt;it++){
            tbb::parallel_for(tbb::blocked_range<int>(0,(int)blocks.size()),
       	                   [&](const tbb::blocked_range<int>&r){
           	    for(int ib=r.begin();ib!=r.end();++ib){
                    for(int iz=blocks[ib]._beginZ;iz<blocks[ib]._endZ;iz++){
                        for(int iy=blocks[ib]._beginY;iy<blocks[ib]._endY;iy++){
                            int i=blocks[ib]._beginX+iy*nx+iz*nxy;
       	                    forward(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,v+i,eps+i,del+i,prevSigmaX+i,curSigmaX+i,prevSigmaZ+i,curSigmaZ+i);
       	                    forward(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,v+i,eps+i,del+i,prevSigmaXa+i,curSigmaXa+i,prevSigmaZa+i,curSigmaZa+i);
//       	                    adjoint(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,v+i,eps+i,del+i,prevSigmaXa+i,curSigmaXa+i,prevSigmaZa+i,curSigmaZa+i,cc);
//       	                    forwardCij(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,c11+i,c13+i,c33+i,prevSigmaX+i,curSigmaX+i,prevSigmaZ+i,curSigmaZ+i);
//       	                    adjointCij(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,c11+i,c13+i,c33+i,prevSigmaXa+i,curSigmaXa+i,prevSigmaZa+i,curSigmaZa+i,cc);
                        }                                                           
                    }                                                                        
                }                                                                            
            });                                                                     

            float *pt=prevSigmaX;prevSigmaX=curSigmaX;curSigmaX=pt;
            pt=prevSigmaZ;prevSigmaZ=curSigmaZ;curSigmaZ=pt;
            pt=prevSigmaXa;prevSigmaXa=curSigmaXa;curSigmaXa=pt;
            pt=prevSigmaZa;prevSigmaZa=curSigmaZa;curSigmaZa=pt;
    
            int timeIndex=nt-it;
            
            s=wavelet[timeIndex]*dt2;
            curSigmaX[souIndex]+=s;
            curSigmaZ[souIndex]+=s;
//            curSigmaX[souIndex+nxy]-=s;
//            curSigmaZ[souIndex+nxy]-=s;

            float f=float(timeIndex)/float(samplingTimeStep);
            int j=f;
            f=f-j;
            #pragma omp parallel for num_threads(16)
            for(int ir=0;ir<nr1;ir++){
                float temp=(1.-f)*data[j+ir*nnt]+f*data[(j+1)+ir*nnt];
                temp*=dt2;
                curSigmaXa[recIndex[ir]]+=2./3.*temp;
                curSigmaZa[recIndex[ir]]+=1./3.*temp;
//                curSigmaXa[recIndex[ir]+nxy]-=2./3.*temp;
//                curSigmaZa[recIndex[ir]+nxy]-=1./3.*temp;
            }
            
            abc(prevSigmaXa,damping,nx,ny,nz,npad);
            abc(prevSigmaZa,damping,nx,ny,nz,npad);
            abc(curSigmaXa,damping,nx,ny,nz,npad);
            abc(curSigmaZa,damping,nx,ny,nz,npad);

            #pragma omp parallel for num_threads(16)
            for(size_t i=0;i<nxyz;i++){
                image0[i]+=(2.*curSigmaX[i]+curSigmaZ[i])*(2.*curSigmaXa[i]+curSigmaZa[i])/9.;
            }
//
//            #pragma omp parallel for num_threads(16)
//            for(int iz=0;iz<nz;iz++){
//                for(int ix=0;ix<nx;ix++){
//                    int i=ix+ny/2*nx+iz*nxy;
//                    a[ix+iz*nx]=(2.*curSigmaX[i]+curSigmaZ[i])*(2.*curSigmaXa[i]+curSigmaZa[i])/9.;
//                }
//            }
//            memcpy(backwardImageWavefield+it*nxz,a,nxz*sizeof(float));
//    
//            #pragma omp parallel for num_threads(16)
//            for(int iz=0;iz<nz;iz++){
//                for(int ix=0;ix<nx;ix++){
//                    int i=ix+ny/2*nx+iz*nxy;
//                    a[ix+iz*nx]=(2.*curSigmaX[i]+curSigmaZ[i])/3.;
//                }
//            }
//            memcpy(backwardSouWavefield+it*nxz,a,nxz*sizeof(float));
//    
//            #pragma omp parallel for num_threads(16)
//            for(int iz=0;iz<nz;iz++){
//                for(int ix=0;ix<nx;ix++){
//                    int i=ix+ny/2*nx+iz*nxy;
//                    a[ix+iz*nx]=(2.*curSigmaXa[i]+curSigmaZa[i])/3.;
//                }
//            }
//            memcpy(backwardRecWavefield+it*nxz,a,nxz*sizeof(float));
        }

        delete []recIndex;
        delete []data;
    }
   
    float *image=new float[nxyz]();
    #pragma omp parallel for num_threads(16)
    for(int iz=1;iz<nz-1;iz++){
        for(int iy=1;iy<ny-1;iy++){
            #pragma omp simd
            for(int ix=1;ix<nx-1;ix++){
                size_t i=ix+iy*nx+iz*nxy;
                image[i]=image0[i+1]+image0[i-1]+image0[i+nx]+image0[i-nx]+image0[i+nxy]+image0[i-nxy]-6.f*image0[i];
            }
        }
    }

    write("image",image,nxyz);
    to_header("image","n1",nx,"o1",ox,"d1",dx);
    to_header("image","n2",ny,"o2",oy,"d2",dy);
    to_header("image","n3",nz,"o3",oz,"d3",dz);

//    write("forwardSouWavefield",forwardSouWavefield,nx*nz*nt);
//    to_header("forwardSouWavefield","n1",nx,"o1",ox,"d1",dx);
//    to_header("forwardSouWavefield","n2",nz,"o2",oz,"d2",dz);
//    to_header("forwardSouWavefield","n3",nt,"o3",ot,"d3",dt);
//
//    write("backwardSouWavefield",backwardSouWavefield,nx*nz*nt);
//    to_header("backwardSouWavefield","n1",nx,"o1",ox,"d1",dx);
//    to_header("backwardSouWavefield","n2",nz,"o2",oz,"d2",dz);
//    to_header("backwardSouWavefield","n3",nt,"o3",ot,"d3",dt);
//
//    write("backwardRecWavefield",backwardRecWavefield,nx*nz*nt);
//    to_header("backwardRecWavefield","n1",nx,"o1",ox,"d1",dx);
//    to_header("backwardRecWavefield","n2",nz,"o2",oz,"d2",dz);
//    to_header("backwardRecWavefield","n3",nt,"o3",ot,"d3",dt);
//
//    write("backwardImageWavefield",backwardImageWavefield,nx*nz*nt);
//    to_header("backwardImageWavefield","n1",nx,"o1",ox,"d1",dx);
//    to_header("backwardImageWavefield","n2",nz,"o2",oz,"d2",dz);
//    to_header("backwardImageWavefield","n3",nt,"o3",ot,"d3",dt);

    delete []wavelet;delete []damping;
    delete []prevSigmaX;delete []curSigmaX;
    delete []prevSigmaZ;delete []curSigmaZ;
    delete []prevSigmaXa;delete []curSigmaXa;
    delete []prevSigmaZa;delete []curSigmaZa;
    delete []image;delete []image0;
    delete []v;delete []eps;delete []del;
    
//    delete []forwardSouWavefield;delete []backwardSouWavefield;
//    delete []backwardRecWavefield;delete []backwardImageWavefield;
//    delete []a;

    myio_close();
    return 0;
}

