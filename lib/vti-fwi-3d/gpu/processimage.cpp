#include <iostream>
#include <cstring>
#include <chrono>

#include "myio.h"
#include "mylib.h"
#include "boundary.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);
 
 int nx,ny,nz,npad,nt;
 float ox,oy,oz,ot,dx,dy,dz,dt;
 
 from_header("image","n1",nx,"o1",ox,"d1",dx);
 from_header("image","n2",ny,"o2",oy,"d2",dy);
 from_header("image","n3",nz,"o3",oz,"d3",dz);
 get_param("npad",npad);
 
 long long nxy=nx*ny;
 long long nxyz=nxy*nz;
 long long nboundary=nxyz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);
 
 float *image0=new float[nxyz];
 read("image",image0,nxyz);

 float wbottom; get_param("wbottom",wbottom);
 
 zeroBoundary(image0,nx,ny,nz,npad);
 int nwbottom=(wbottom-oz)/dz+1-npad;
 memset(image0+npad*nxy,0,nwbottom*nxy*sizeof(float));
 
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

 write("imageout",image,nxyz);
 to_header("imageout","n1",nx,"o1",ox,"d1",dx);
 to_header("imageout","n2",ny,"o2",oy,"d2",dy);
 to_header("imageout","n3",nz,"o3",oz,"d3",dz);

 delete []image0;
 delete []image;
 
 myio_close();
 return 0;
}

