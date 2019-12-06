#include <iostream>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);
 
 int nx,ny,nz,npad,nt;
 float ox,oy,oz,ot,dx,dy,dz,dt;
 
 from_header("cij","n1",nx,"o1",ox,"d1",dx);
 from_header("cij","n2",ny,"o2",oy,"d2",dy);
 from_header("cij","n3",nz,"o3",oz,"d3",dz);
 get_param("npad",npad);
 get_param("nt",nt,"ot",ot,"dt",dt);

 long long nxy=nx*ny;
 long long nxyz=nxy*nz;
 
 float *wavelet=new float[nt]();
 read("wavelet",wavelet,nt);

 float samplingRate;
 get_param("samplingRate",samplingRate);
 int samplingTimeStep=std::round(samplingRate/dt);
 
 float *cij=new float[3*nxyz];
 read("cij",cij,3*nxyz);
 float *c11=cij,*c13=cij+nxyz,*c33=cij+2*nxyz;

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
 
 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 float *image=new float[nxyz];

 rtmCij3d_f(image,souloc,ns,shotid,recloc,wavelet,c11,c13,c33,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate);

 write("image",image,nxyz);
 to_header("image","n1",nx,"o1",ox,"d1",dx);
 to_header("image","n2",ny,"o2",oy,"d2",dy);
 to_header("image","n3",nz,"o3",oz,"d3",dz);

 delete []image;

 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 delete []wavelet;
 delete []cij;

 delete []souloc;
 delete []recloc;
 
 myio_close();
 return 0;
}

