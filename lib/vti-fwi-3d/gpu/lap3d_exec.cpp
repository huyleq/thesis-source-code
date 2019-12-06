#include <cstdio>
#include <cstdlib>

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>

#include "myio.h"
#include "lap3d.h"

#define HALF_STENCIL 1
#define BLOCKSIZE 16

using namespace std;

class Block{
    public:
    Block(int beginX,int endX,int beginY,int endY,int beginZ,int endZ):_beginX(beginX),_endX(endX),_beginY(beginY),_endY(endY),_beginZ(beginZ),_endZ(endZ){};
    int _beginX,_endX,_beginY,_endY,_beginZ,_endZ;
};

int main(int argc,char **argv){
    myio_init(argc,argv);

    int nx,ny,nz;
    float ox,oy,oz,dx,dy,dz;
    
    from_header("u","n1",nx,"o1",ox,"d1",dx);
    from_header("u","n2",ny,"o2",oy,"d2",dy);
    from_header("u","n3",nz,"o3",oz,"d3",dz);
   
    long long nxy=nx*ny,nxyz=nxy*nz;
    float *u=new float[nxyz]; read("u",u,nxyz);
    float *lapu=new float[nxyz]();
    
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

    tbb::parallel_for(tbb::blocked_range<int>(0,(int)blocks.size()),
                    [&](const tbb::blocked_range<int>&r){
   	    for(int ib=r.begin();ib!=r.end();++ib){
            for(int iz=blocks[ib]._beginZ;iz<blocks[ib]._endZ;iz++){
                for(int iy=blocks[ib]._beginY;iy<blocks[ib]._endY;iy++){
                    int i=blocks[ib]._beginX+iy*nx+iz*nxy;
                     lap3d(blocks[ib]._endX-blocks[ib]._beginX,nx,nxy,lapu+i,u+i);
                }                                                           
            }                                                                        
        }                                                                            
    });                                                                     
    
    write("lapu",lapu,nxyz);
    to_header("lapu","n1",nx,"o1",ox,"d1",dx);
    to_header("lapu","n2",ny,"o2",oy,"d2",dy);
    to_header("lapu","n3",nz,"o3",oz,"d3",dz);
   
    delete []u;delete []lapu;
    myio_close();
    return 0;
}

