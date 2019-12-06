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
#include "boundary.h"
#include "conversions.h"

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

//    float cc[5]={C0,C1,C2,C3,C4};

    int nx,ny,nz,nt,npad;
    float ox,oy,oz,ot,dx,dy,dz,dt;

    from_header("v","n1",nx,"o1",ox,"d1",dx);
    from_header("v","n2",ny,"o2",oy,"d2",dy);
    from_header("v","n3",nz,"o3",oz,"d3",dz);
    get_param("npad",npad);
    get_param("nt",nt,"ot",ot,"dt",dt);
    
    long long nxy=nx*ny;
    long long nxyz=nxy*nz;
    long long nboundary=nxyz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);
    
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

    float *randomboundary=new float[nboundary];
    read("randomboundary",randomboundary,nboundary);
    float *padboundary=new float[nboundary];
    read("padboundary",padboundary,nboundary);
    
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
    
    float *gv=new float[nxyz]();
    float *geps=new float[nxyz]();
    float *gdel=new float[nxyz]();
    float *gc11=new float[nxyz]();
    float *gc13=new float[nxyz]();
    float *gc33=new float[nxyz]();

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

    double objfunc=0.;

    for(int is=0;is<ns;is++){
        fprintf(stderr,"shot %d\n",is);
        putBoundary(padboundary,v,nx,ny,nz,npad);

        int nr1=souloc[5*is+3];
        int irbegin=souloc[5*is+4];
        
        int *recIndex=new int[nr1];
        #pragma omp parallel for num_threads(16)
        for(int ir=0;ir<nr1;ir++){
            int ir1=irbegin+ir;
            int recIndexX=(recloc[3*ir1]-ox)/dx;
            int recIndexY=(recloc[3*ir1+1]-oy)/dy;
            int recIndexZ=(recloc[3*ir1+2]-oz)/dz;
            recIndex[ir]=recIndexX+recIndexY*nx+recIndexZ*nxy;
        }    
        
        float *bgdata=new float[nnt*nr1]();

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
//            curSigmaX[souIndex+nxy]-=s;
//            curSigmaZ[souIndex+nxy]-=s;
            
            abc(prevSigmaX,damping,nx,ny,nz,npad);
            abc(prevSigmaZ,damping,nx,ny,nz,npad);
            abc(curSigmaX,damping,nx,ny,nz,npad);
            abc(curSigmaZ,damping,nx,ny,nz,npad);
    
            if(it%samplingTimeStep==0){
                int it1=it/samplingTimeStep;
                for(int ir=0;ir<nr1;ir++){
                    bgdata[it1+ir*nnt]=(2.*curSigmaX[recIndex[ir]]+curSigmaZ[recIndex[ir]])/3.; 
//                    bgdata[it1+ir*nnt]=(2.*(curSigmaX[recIndex[ir]]-curSigmaX[recIndex[ir]+nxy])+(curSigmaZ[recIndex[ir]]-curSigmaZ[recIndex[ir]+nxy]))/3.; 
                }    
            }
        }

//     write("modeleddata",bgdata,nnt*nr1,ios_base::app);
//
//     if(is==0){
//      write("abforwardwfld",curSigmaX,nxyz);
//      to_header("abforwardwfld","n1",nx,"o1",ox,"d1",dx);
//      to_header("abforwardwfld","n2",ny,"o2",oy,"d2",dy);
//      to_header("abforwardwfld","n3",nz,"o3",oz,"d3",dz);
//     }

        putBoundary(randomboundary,v,nx,ny,nz,npad);

        memset(curSigmaX,0,nxyz*sizeof(float));
        memset(prevSigmaX,0,nxyz*sizeof(float));
        memset(curSigmaZ,0,nxyz*sizeof(float));
        memset(prevSigmaZ,0,nxyz*sizeof(float));

        s=wavelet[0]*dt2;
        curSigmaX[souIndex]+=s;
        curSigmaZ[souIndex]+=s;
//        curSigmaX[souIndex+nxy]-=s;
//        curSigmaZ[souIndex+nxy]-=s;
        
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
//            curSigmaX[souIndex+nxy]-=s;
//            curSigmaZ[souIndex+nxy]-=s;
        }
            
        float *pt=prevSigmaX;prevSigmaX=curSigmaX;curSigmaX=pt;
        pt=prevSigmaZ;prevSigmaZ=curSigmaZ;curSigmaZ=pt;
    
        memset(curSigmaXa,0,nxyz*sizeof(float));
        memset(prevSigmaXa,0,nxyz*sizeof(float));
        memset(curSigmaZa,0,nxyz*sizeof(float));
        memset(prevSigmaZa,0,nxyz*sizeof(float));

        float *data=new float[nnt*nr1];
        read("data",data,nnt*nr1,nnt*irbegin);
        #pragma omp parallel for reduction(+:objfunc) num_threads(16)
        for(size_t i=0;i<nnt*nr1;i++){
            data[i]=bgdata[i]-data[i];
            objfunc+=data[i]*data[i];
        }

//     write("residual",data,nnt*nr1,ios_base::app);
//	
//     if(is==0){
//      write("randforwardwfld",curSigmaX,nxyz);
//      to_header("randforwardwfld","n1",nx,"o1",ox,"d1",dx);
//      to_header("randforwardwfld","n2",ny,"o2",oy,"d2",dy);
//      to_header("randforwardwfld","n3",nz,"o3",oz,"d3",dz);
//     }

        #pragma omp parallel for num_threads(16)
        for(int ir=0;ir<nr1;ir++){
            float temp=dt2*data[(nnt-1)+ir*nnt];
            curSigmaXa[recIndex[ir]]+=2./3.*temp;
            curSigmaZa[recIndex[ir]]+=1./3.*temp;
//            curSigmaXa[recIndex[ir]+nxy]-=2./3.*temp;
//            curSigmaZa[recIndex[ir]+nxy]-=1./3.*temp;
        }

        for(int it=2;it<nt;it++){
            tbb::parallel_for(tbb::blocked_range<int>(0,(int)blocks.size()),
       	                   [&](const tbb::blocked_range<int>&r){
           	    for(int ib=r.begin();ib!=r.end();++ib){
                    for(int iz=blocks[ib]._beginZ;iz<blocks[ib]._endZ;iz++){
                        for(int iy=blocks[ib]._beginY;iy<blocks[ib]._endY;iy++){
                            int i=blocks[ib]._beginX+iy*nx+iz*nxy;
//       	                    gradient(gv+i,geps+i,gdel+i,blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,v+i,eps+i,del+i,prevSigmaX+i,curSigmaX+i,prevSigmaZ+i,curSigmaZ+i,curSigmaXa+i,curSigmaZa+i);
      	                    gradientCij(gc11+i,gc13+i,gc33+i,blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,v+i,eps+i,del+i,prevSigmaX+i,curSigmaX+i,prevSigmaZ+i,curSigmaZ+i,curSigmaXa+i,curSigmaZa+i);
//       	                    adjoint(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,v+i,eps+i,del+i,prevSigmaXa+i,curSigmaXa+i,prevSigmaZa+i,curSigmaZa+i,cc);
       	                    forward(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,dx2,dy2,dz2,dt2,v+i,eps+i,del+i,prevSigmaXa+i,curSigmaXa+i,prevSigmaZa+i,curSigmaZa+i);
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
        }

//     if(is==0){
//      write("adjwfld",curSigmaXa,nxyz);
//      to_header("adjwfld","n1",nx,"o1",ox,"d1",dx);
//      to_header("adjwfld","n2",ny,"o2",oy,"d2",dy);
//      to_header("adjwfld","n3",nz,"o3",oz,"d3",dz);
//     }

        delete []recIndex;
        delete []data;
        delete []bgdata;
    }

    int nrtotal=souloc[5*(ns-1)+3]+souloc[5*(ns-1)+4];
//    to_header("modeleddata","n1",nnt,"o1",ot,"d1",samplingRate);
//    to_header("modeleddata","n2",nrtotal,"o2",0.,"d2",1);
//    to_header("residual","n1",nnt,"o1",ot,"d1",samplingRate);
//    to_header("residual","n2",nrtotal,"o2",0.,"d2",1);
 
    fprintf(stderr,"objfunc is %10.16f\n",0.5*objfunc);
   
    zeroBoundary(gc11,nx,ny,nz,npad);
    zeroBoundary(gc13,nx,ny,nz,npad);
    zeroBoundary(gc33,nx,ny,nz,npad);

    float wbottom;get_param("wbottom",wbottom);
    int nwbottom=(wbottom-oz)/dz+1-npad;
    memset(gc11+npad*nxy,0,nwbottom*nxy*sizeof(float));
    memset(gc13+npad*nxy,0,nwbottom*nxy*sizeof(float));
    memset(gc33+npad*nxy,0,nwbottom*nxy*sizeof(float));
    
    GradCij2GradVEpsDel(gv,geps,gdel,gc11,gc13,gc33,v,eps,del,1.,1.,1.,nxyz);
    write("gv",gv,nxyz);
    to_header("gv","n1",nx,"o1",ox,"d1",dx);
    to_header("gv","n2",ny,"o2",oy,"d2",dy);
    to_header("gv","n3",nz,"o3",oz,"d3",dz);

    write("geps",geps,nxyz);
    to_header("geps","n1",nx,"o1",ox,"d1",dx);
    to_header("geps","n2",ny,"o2",oy,"d2",dy);
    to_header("geps","n3",nz,"o3",oz,"d3",dz);

    write("gdel",gdel,nxyz);
    to_header("gdel","n1",nx,"o1",ox,"d1",dx);
    to_header("gdel","n2",ny,"o2",oy,"d2",dy);
    to_header("gdel","n3",nz,"o3",oz,"d3",dz);

    write("gc11",gc11,nxyz);
    to_header("gc11","n1",nx,"o1",ox,"d1",dx);
    to_header("gc11","n2",ny,"o2",oy,"d2",dy);
    to_header("gc11","n3",nz,"o3",oz,"d3",dz);

    write("gc13",gc13,nxyz);
    to_header("gc13","n1",nx,"o1",ox,"d1",dx);
    to_header("gc13","n2",ny,"o2",oy,"d2",dy);
    to_header("gc13","n3",nz,"o3",oz,"d3",dz);

    write("gc33",gc33,nxyz);
    to_header("gc33","n1",nx,"o1",ox,"d1",dx);
    to_header("gc33","n2",ny,"o2",oy,"d2",dy);
    to_header("gc33","n3",nz,"o3",oz,"d3",dz);

    delete []wavelet;delete []damping;
    delete []prevSigmaX;delete []curSigmaX;
    delete []prevSigmaZ;delete []curSigmaZ;
    delete []prevSigmaXa;delete []curSigmaXa;
    delete []prevSigmaZa;delete []curSigmaZa;
    delete []gv;delete []geps;delete []gdel;
    delete []gc11;delete []gc13;delete []gc33;
    delete []v;delete []eps;delete []del;
    delete []randomboundary;delete []padboundary;
    
    myio_close();
    return 0;
}

