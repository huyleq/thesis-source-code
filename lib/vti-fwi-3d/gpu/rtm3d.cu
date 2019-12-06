#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);
 
 int nx,ny,nz,npad,nt;
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
 
 float *v=new float[nxyz];
 read("v",v,nxyz);
 float *eps=new float[nxyz]();
 read("eps",eps,nxyz);
 float *del=new float[nxyz]();
 read("del",del,nxyz);

 float *randomboundary=new float[nboundary]; 
 read("randomboundary",randomboundary,nboundary);

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
 
 float *image0=new float[nxyz];

 float wbottom; get_param("wbottom",wbottom);
 
 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 rtmVEpsDel(image0,souloc,ns,shotid,recloc,wavelet,v,eps,del,randomboundary,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,wbottom);

 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
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

 delete []wavelet;
 delete []v;delete []eps;delete []del;
 delete []image0;
 delete []image;
 delete []randomboundary;

 delete []souloc;
 delete []recloc;
 
 myio_close();
 return 0;
}

