#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"
#include "check.h"
#include "lbfgs.h"

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
 
 float *vepsdel=new float[3*nxyz];
 float *v=vepsdel,*eps=vepsdel+nxyz,*del=vepsdel+2*nxyz;
 read("v",v,nxyz);
 if(!read("eps",eps,nxyz)) memset(eps,0,nxyz*sizeof(float));
 if(!read("del",del,nxyz)) memset(del,0,nxyz*sizeof(float));

 float *randomboundary=new float[nboundary]; 
 read("randomboundary",randomboundary,nboundary);
 float *padboundary=new float[nboundary];
 read("padboundary",padboundary,nboundary);

 int ns,nr;
 from_header("souloc","n2",ns);
 float *souloc=new float[5*ns];
 read("souloc",souloc,5*ns);

 vector<int> shotid;
 bool providedShotId=get_array("shotid",shotid);
 if(!providedShotId){
  vector<int> badshot;
  get_array("badshot",badshot);
  for(int i=0;i<ns;i++){
   if(find(badshot.begin(),badshot.end(),i)==badshot.end()) shotid.push_back(i);
  }
 }

 from_header("recloc","n2",nr);
 float *recloc=new float[3*nr];
 read("recloc",recloc,3*nr);
 
 float wbottom=0.; get_param("wbottom",wbottom);
 float *m=new float[nxyz];
 
 float v0=1.,eps0=1.;
 get_param("v0",v0,"eps0",eps0);

 scale(v,v,1./v0,nxyz);
 scale(padboundary,padboundary,1./v0,nboundary);
 scale(randomboundary,randomboundary,1./v0,nboundary);
 scale(eps,eps,1./eps0,nxyz);

 float *gvepsdel=new float[3*nxyz]();
 float *gv=gvepsdel,*geps=gvepsdel+nxyz,*gdel=gvepsdel+2*nxyz;

 int nfg; get_param("nfg",nfg);

 int nn=3*nxyz,mm=5,diagco=0,icall=0,iflag=0;
 float f;
 float *diag=new float[nn]();
 float *w=new float[nn*(2*mm+1)+2*mm]();
 int *isave=new int[nisave]();
 float *dsave=new float[ndsave]();
 
 vector<int> GPUs;
 get_array("gpu",GPUs);
 int ngpugroup=1;
 get_param("ngpugroup",ngpugroup);
 
 to_header("iv","n1",nx,"o1",ox,"d1",dx);
 to_header("iv","n2",ny,"o2",oy,"d2",dy);
 to_header("iv","n3",nz,"o3",oz,"d3",dz);
 
 to_header("ieps","n1",nx,"o1",ox,"d1",dx);
 to_header("ieps","n2",ny,"o2",oy,"d2",dy);
 to_header("ieps","n3",nz,"o3",oz,"d3",dz);
 
 to_header("idel","n1",nx,"o1",ox,"d1",dx);
 to_header("idel","n2",ny,"o2",oy,"d2",dy);
 to_header("idel","n3",nz,"o3",oz,"d3",dz);
 
 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 while(true){
  icall++;
  checkEpsDel(eps,del,eps0,1.,nxyz,m);
  
  f=objFuncGradientVEpsDel(gv,geps,gdel,souloc,ns,shotid,recloc,wavelet,v,eps,del,padboundary,randomboundary,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate,v0,eps0,wbottom,GPUs,ngpugroup);

  fprintf(stderr,"icall %d objfunc %.16f\n",icall,f);

  write("objfunc",&f,1,std::ios_base::app);
  to_header("objfunc","n1",icall,"o1",0.,"d1",1.);
  
  write("iv",v,nxyz,std::ios_base::app);
  to_header("iv","n4",icall,"o4",0.,"d4",1.);
  
  write("ieps",eps,nxyz,std::ios_base::app);
  to_header("ieps","n4",icall,"o4",0.,"d4",1.);
  
  write("idel",del,nxyz,std::ios_base::app);
  to_header("idel","n4",icall,"o4",0.,"d4",1.);
  
  lbfgs(nn,mm,vepsdel,f,gvepsdel,diagco,diag,w,iflag,isave,dsave);
  
  if(iflag<=0 || icall>nfg){
   fprintf(stderr,"iflag %d\n",iflag);
   break;
  }
 }
 
 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 delete []vepsdel;delete []gvepsdel;delete []diag;delete []w;
 delete []isave;delete []dsave; 
 delete []wavelet;delete []souloc;delete []recloc;
 delete []m;
 delete []padboundary; delete []randomboundary;

 myio_close();
 return 0;
}
