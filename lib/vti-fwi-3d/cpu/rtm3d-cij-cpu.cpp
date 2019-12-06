#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

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

    int nx,ny,nz,npad,nt;
    float ox,oy,oz,ot,dx,dy,dz,dt;
    
    from_header("cij","n1",nx,"o1",ox,"d1",dx);
    from_header("cij","n2",ny,"o2",oy,"d2",dy);
    from_header("cij","n3",nz,"o3",oz,"d3",dz);
    get_param("npad",npad);
    get_param("nt",nt,"ot",ot,"dt",dt);
   
    long long nxy=nx*ny,nxyz=nxy*nz,nn=3*nxyz;
    long long nboundary=nxyz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);
    
    float *wavelet=new float[nt]();
    read("wavelet",wavelet,nt);
   
    float samplingRate;
    get_param("samplingRate",samplingRate);
    int samplingTimeStep=std::round(samplingRate/dt);
    int nnt=(nt-1)/samplingTimeStep+1;
    
    float *cij=new float[nn];
    float *c11=cij,*c13=cij+nxyz,*c33=cij+2*nxyz;
    read("cij",cij,nn);
   
    float *image=new float[nxyz];
    
    int ns,nr;
    from_header("souloc","n2",ns);
    float *souloc=new float[5*ns];
    read("souloc",souloc,5*ns);
   
    vector<int> shotid;
    if(!get_array("shotid",shotid)){
     vector<int> shotrange;
     if(!get_array("shotrange",shotrange)){
       shotrange.push_back(0);
       shotrange.push_back(ns);
     }
     vector<int> badshot;
     get_array("badshot",badshot);
     for(int i=shotrange[0];i<shotrange[1];i++){
      if(find(badshot.begin(),badshot.end(),i)==badshot.end()) shotid.push_back(i);
     }
    }
   
    from_header("recloc","n2",nr);
    float *recloc=new float[3*nr];
    read("recloc",recloc,3*nr);
 
    float *damping=new float[npad];
    for(int i=0;i<npad;i++) damping[i]=DAMPER+(1.-DAMPER)*cos(PI*(npad-i)/npad);

    float dx2=dx*dx,dy2=dy*dy,dz2=dz*dz,dt2=dt*dt;

    float *prevSigmaX=new float[nxyz];
    float *curSigmaX=new float[nxyz];
    float *prevSigmaZ=new float[nxyz];
    float *curSigmaZ=new float[nxyz];

    float *prevSigmaXa=new float[nxyz];
    float *curSigmaXa=new float[nxyz];
    float *prevSigmaZa=new float[nxyz];
    float *curSigmaZa=new float[nxyz];
    
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

    double objfunc=0.;

    chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
    for(vector<int>::iterator it=shotid.begin();it!=shotid.end();it++){
        int is=*it;
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
       	                    forwardCij(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,c11+i,c13+i,c33+i,prevSigmaX+i,curSigmaX+i,prevSigmaZ+i,curSigmaZ+i);
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
            image[i]+=(2.*curSigmaX[i]+curSigmaZ[i])*(2.*curSigmaXa[i]+curSigmaZa[i])/9.;
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
       	                    forwardCij(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,c11+i,c13+i,c33+i,prevSigmaX+i,curSigmaX+i,prevSigmaZ+i,curSigmaZ+i);
       	                    forwardCij(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,c11+i,c13+i,c33+i,prevSigmaXa+i,curSigmaXa+i,prevSigmaZa+i,curSigmaZa+i);
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
                image[i]+=(2.*curSigmaX[i]+curSigmaZ[i])*(2.*curSigmaXa[i]+curSigmaZa[i])/9.;
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
    
    chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
    chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
    cout<<"total time "<<time.count()/60.<<" minutes"<<endl;

//    int nrtotal=souloc[5*(ns-1)+3]+souloc[5*(ns-1)+4];
//    to_header("modeleddata","n1",nnt,"o1",ot,"d1",samplingRate);
//    to_header("modeleddata","n2",nrtotal,"o2",0.,"d2",1);
//    to_header("residual","n1",nnt,"o1",ot,"d1",samplingRate);
//    to_header("residual","n2",nrtotal,"o2",0.,"d2",1);
 
    if(write("image",image,nxyz)){
     to_header("image","n1",nx,"o1",ox,"d1",dx);
     to_header("image","n2",ny,"o2",oy,"d2",dy);
     to_header("image","n3",nz,"o3",oz,"d3",dz);
    }
  
    delete []wavelet;delete []damping;
    delete []prevSigmaX;delete []curSigmaX;
    delete []prevSigmaZ;delete []curSigmaZ;
    delete []prevSigmaXa;delete []curSigmaXa;
    delete []prevSigmaZa;delete []curSigmaZa;
    delete []cij;delete []image;
    
    myio_close();
    fprintf(stderr,"jobstate=C\n");
    return 0;
}

