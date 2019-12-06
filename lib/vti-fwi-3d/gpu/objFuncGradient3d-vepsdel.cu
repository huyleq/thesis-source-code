#include <iostream>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"
#include "check.h"

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
 float *padboundary=new float[nboundary];
 read("padboundary",padboundary,nboundary);

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
 
 float *gv=new float[nxyz]();
 float *geps=new float[nxyz]();
 float *gdel=new float[nxyz]();

 float wbottom=0.; get_param("wbottom",wbottom);
 float *m=new float[nxyz];
 
 float v0=1.,eps0=1.;
 get_param("v0",v0,"eps0",eps0);
 scale(v,1./v0,nxyz);
 scale(eps,1./eps0,nxyz);
 
 checkEpsDel(eps,del,eps0,1.,nxyz,m);
 
 vector<int> GPUs;
 get_array("gpu",GPUs);
 int ngpugroup=1;
 get_param("ngpugroup",ngpugroup);
 
 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 double objfunc=objFuncGradientVEpsDel(gv,geps,gdel,souloc,ns,shotid,recloc,wavelet,v,eps,del,padboundary,randomboundary,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,v0,eps0,wbottom,GPUs,ngpugroup);

 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 fprintf(stderr,"objfunc is %10.16f\n",objfunc); 
 
 float agv=fabs(gv[0]),av=fabs(v[0]);
 float age=fabs(geps[0]),ae=fabs(eps[0]);
 float agd=fabs(gdel[0]),ad=fabs(del[0]);
 for(int i=0;i<nxyz;i++){
     if(fabs(gv[i])>agv) agv=fabs(gv[i]);
     if(fabs(v[i])>av) av=fabs(v[i]);
     if(fabs(geps[i])>age) age=fabs(geps[i]);
     if(fabs(eps[i])>ae) ae=fabs(eps[i]);
     if(fabs(gdel[i])>agd) agd=fabs(gdel[i]);
     if(fabs(del[i])>ad) ad=fabs(del[i]);
 }

 if(ae==0.) get_param("maxeps",ae);
 if(ad==0.) get_param("maxdel",ad);

 v0=sqrt((agd/ad)/(agv/av));
 eps0=sqrt((agd/ad)/(age/ae));

 cout<<"v0 should be "<<v0<<endl;
 cout<<"eps0 should be "<<eps0<<endl;
 cout<<"del0 should be 1"<<endl;

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

 delete []wavelet;
 delete []v;delete []eps;delete []del;
 delete []gv;
 delete []geps;
 delete []gdel;
 delete []randomboundary; delete []padboundary;
 delete []m;

 delete []souloc;
 delete []recloc;
 
 myio_close();
 return 0;
}

