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

 string wantgrad=get_s("wantgrad");
 
 long long nxy=nx*ny,nxyz=nxy*nz,nn=3*nxyz;
 long long nboundary=nxyz-(nx-2*npad)*(ny-2*npad)*(nz-2*npad);
 
 float *wavelet=new float[nt]();
 read("wavelet",wavelet,nt);

 float samplingRate;
 get_param("samplingRate",samplingRate);
 int samplingTimeStep=std::round(samplingRate/dt);
 
 float *cij=new float[3*nxyz];
 read("cij",cij,3*nxyz);

 float *padboundaryCij=new float[3*nboundary];
 read("padboundary",padboundaryCij,3*nboundary);

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
 float *recloc=new float[4*nr];
 read("recloc",recloc,4*nr);
 
 vector<int> GPUs;
 get_array("gpu",GPUs);
 int ngpugroup=1;
 get_param("ngpugroup",ngpugroup);
 
 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 float objfunc=0.f;

 if(wantgrad.compare("n")==0){
  if(shotid.size()!=0){
   if(ngpugroup>1) objFuncCij3d(&objfunc,souloc,ns,shotid,recloc,wavelet,cij,padboundaryCij,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,GPUs,ngpugroup);
   else objFuncCij3d_f(&objfunc,souloc,ns,shotid,recloc,wavelet,cij,padboundaryCij,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,GPUs);
  }
 }
 else{
  float *randboundaryCij=new float[3*nboundary]; 
  read("randomboundary",randboundaryCij,3*nboundary);
  
  float *fgcij=new float[nn+1](),*gcij=fgcij+1;

  if(shotid.size()!=0){
   if(ngpugroup>1) objFuncGradientCij3d(fgcij,souloc,ns,shotid,recloc,wavelet,cij,padboundaryCij,randboundaryCij,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,GPUs,ngpugroup);
   else objFuncGradientCij3d_f(fgcij,souloc,ns,shotid,recloc,wavelet,cij,padboundaryCij,randboundaryCij,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,GPUs);
  }

  objfunc=fgcij[0];
  if(write("fgcij",fgcij,nn+1)){
   to_header("fgcij","n1",nx,"o1",ox,"d1",dx);
   to_header("fgcij","n2",ny,"o2",oy,"d2",dy);
   to_header("fgcij","n3",nz,"o3",oz,"d3",dz);
   to_header("fgcij","n4",3,"o4",0,"d4",1);
  }

  if(write("gcij",gcij,nn)){
   to_header("gcij","n1",nx,"o1",ox,"d1",dx);
   to_header("gcij","n2",ny,"o2",oy,"d2",dy);
   to_header("gcij","n3",nz,"o3",oz,"d3",dz);
   to_header("gcij","n4",3,"o4",0,"d4",1);
  }

  delete []randboundaryCij;delete []fgcij;
 }

 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 fprintf(stderr,"objfunc=%10.16f\n",objfunc); 
 
 delete []wavelet;
 delete []cij;
 delete []padboundaryCij;

 delete []souloc;
 delete []recloc;
 
 fprintf(stderr,"jobstate=COMPLETED\n");
 myio_close();
 return 0;
}

