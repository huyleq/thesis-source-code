#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>
#include <new>

#include "myio.h"
#include "mylib.h"

#include "wave3d.h"
#include "check.h"
#include "conversions.h"
#include "ModelingOp3d.h"
#include "LinearSolver.h"

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
 
 long long nxy=nx*ny,nxyz=nxy*nz,nn=3*nxyz;
 
 float samplingRate;
 get_param("samplingRate",samplingRate);
 int samplingTimeStep=std::round(samplingRate/dt);
 
 float *vepsdel=new float[3*nxyz];
 float *v=vepsdel,*eps=vepsdel+nxyz,*del=vepsdel+2*nxyz;
 read("v",v,nxyz);
 if(!read("eps",eps,nxyz)) memset(eps,0,nxyz*sizeof(float));
 if(!read("del",del,nxyz)) memset(del,0,nxyz*sizeof(float));

 float *m=new float[nxyz]();
 float *cij=new float[nn];
 float *c11=cij,*c13=cij+nxyz,*c33=cij+2*nxyz;
 
 checkEpsDel(eps,del,1.,1.,nxyz,m);
 VEpsDel2Cij(c11,c13,c33,v,eps,del,1.f,1.f,1.f,nxyz);
 delete []vepsdel;

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

 int ngpugroup=1; get_param("ngpugroup",ngpugroup);
 vector<int> GPUs; get_array("gpu",GPUs);

 float *wavelet=new float[nt]();
 if(!read("wavelet",wavelet,nt)) memset(wavelet,0,nt*sizeof(float));

 int nnt=(nt-1)/samplingTimeStep+1;
 int nrtotal=souloc[5*(ns-1)+3]+souloc[5*(ns-1)+4];

 size_t n=(long long)nnt*(long long)nrtotal;
 float *data=new float[n];

// rand(data,n); scale(data,1e-2,n);
 
 ModelingOp3d Mop(souloc,ns,shotid,recloc,c11,c13,c33,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,GPUs,ngpugroup);
 
// Mop.forward(true,wavelet,data);

// float *wavelet1=new float[nt]();
//
// float *data1=new float[n];
// read("data1",data1,n);
//
// Mop.adjoint(false,wavelet1,data1);
//
// fprintf(stderr,"yAx=%.10f xAty=%.10f\n",dot_product(data,data1,n),dot_product(wavelet,wavelet1,nt));
// 
// write("gwavelet",wavelet1,nt);
// to_header("gwavelet","n1",nt,"o1",ot,"d1",dt);
//
// delete []wavelet1;delete []data1;

// write("data",data,n);
// to_header("data","n1",nnt,"o1",ot,"d1",samplingRate);
// to_header("data","n2",nrtotal,"o2",0,"d2",1);

 read("data",data,n);
 
 int niter;
 get_param("niter",niter);

 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 simpleSolver(&Mop,wavelet,data,niter);

 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 write("iwavelet",wavelet,nt);
 to_header("iwavelet","n1",nt,"o1",ot,"d1",dt);

 delete []cij;delete []m;
 
 delete []wavelet;delete []souloc;delete []recloc;
 delete []data;

 myio_close();
 return 0;
}
