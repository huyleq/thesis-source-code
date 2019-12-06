#include <cstdio>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"
#include "conversions.h"
#include "check.h"
#include "lbfgs.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);

 int nx,ny,nz,npad;
 float ox,oy,oz,dx,dy,dz;
 
 from_header("v","n1",nx,"o1",ox,"d1",dx);
 from_header("v","n2",ny,"o2",oy,"d2",dy);
 from_header("v","n3",nz,"o3",oz,"d3",dz);
 get_param("npad",npad);
 
 long long nxyz=nx*ny*nz;

 string command=get_s("command");

 int ns;
 from_header("souloc","n2",ns);

 vector<int> shotid;
 bool providedShotId=get_array("shotid",shotid);
 if(!providedShotId){
  vector<int> badshot;
  get_array("badshot",badshot);
  for(int i=0;i<ns;i++){
   if(find(badshot.begin(),badshot.end(),i)==badshot.end()) shotid.push_back(i);
  }
 }

 int max_shot_per_job;
 float pct=1.;
 get_param("max_shot_per_job",max_shot_per_job,"pct",pct);
    
 int nfg; get_param("nfg",nfg);

 long long nn=3*nxyz;
 int mm=5,diagco=0,icall=0,iflag=0;
 float f;
 float *vepsdel=new float[nn]; 
 float *v=vepsdel,*eps=vepsdel+nxyz,*del=vepsdel+2*nxyz;
 float *m=new float[nxyz]();
 float *cij=new float[nn];
 float *c11=cij,*c13=cij+nxyz,*c33=cij+2*nxyz;
 float *gvepsdel=new float[nn];
// float *gv=gvepsdel,*geps=gvepsdel+nxyz,*gdel=gvepsdel+2*nxyz;
 float *diag=new float[nn]();
 float *w=new float[nn*(2*mm+1)+2*mm]();
 int *isave=new int[nisave]();
 float *dsave=new float[ndsave]();
 
 float v0=1.,eps0=1.,wbottom=0.;
 get_param("v0",v0,"eps0",eps0,"wbottom",wbottom);
 read("v",v,nxyz);
 if(!read("eps",eps,nxyz)) memset(eps,0,nxyz*sizeof(float));
 if(!read("del",del,nxyz)) memset(del,0,nxyz*sizeof(float));
 scale(v,v,1./v0,nxyz);
 scale(eps,eps,1./eps0,nxyz);

 to_header("iv","n1",nx,"o1",ox,"d1",dx);
 to_header("iv","n2",ny,"o2",oy,"d2",dy);
 to_header("iv","n3",nz,"o3",oz,"d3",dz);
 
 to_header("ieps","n1",nx,"o1",ox,"d1",dx);
 to_header("ieps","n2",ny,"o2",oy,"d2",dy);
 to_header("ieps","n3",nz,"o3",oz,"d3",dz);
 
 to_header("idel","n1",nx,"o1",ox,"d1",dx);
 to_header("idel","n2",ny,"o2",oy,"d2",dy);
 to_header("idel","n3",nz,"o3",oz,"d3",dz);
 
// to_header("gv","n1",nx,"o1",ox,"d1",dx);
// to_header("gv","n2",ny,"o2",oy,"d2",dy);
// to_header("gv","n3",nz,"o3",oz,"d3",dz);
// 
// to_header("geps","n1",nx,"o1",ox,"d1",dx);
// to_header("geps","n2",ny,"o2",oy,"d2",dy);
// to_header("geps","n3",nz,"o3",oz,"d3",dz);
// 
// to_header("gdel","n1",nx,"o1",ox,"d1",dx);
// to_header("gdel","n2",ny,"o2",oy,"d2",dy);
// to_header("gdel","n3",nz,"o3",oz,"d3",dz);
 
 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 while(true){
  checkEpsDel(eps,del,eps0,1.,nxyz,m);
  VEpsDel2Cij(c11,c13,c33,v,eps,del,v0,eps0,1.,nxyz);
  
  string cijfile="cij_icall_"+to_string(icall)+".H";
  writeToHeader(cijfile,cij,nn);
  
  ofstream ofs;
  if(!open_file(ofs,cijfile,ofstream::app)){
      cout<<"cannot open file "<<cijfile<<endl;
  }
  else{
      ofs<<"n1="<<nx<<" o1="<<ox<<" d1="<<dx<<endl;
      ofs<<"n2="<<ny<<" o2="<<oy<<" d2="<<dy<<endl;
      ofs<<"n3="<<nz<<" o3="<<oz<<" d3="<<dz<<endl;
      ofs<<"n4="<<3<<" o4="<<0<<" d4="<<1<<endl;
  }
  close_file(ofs);

  string command1=command+" cij="+cijfile;

  f=objFuncGradientVEpsDel_cluster(gvepsdel,vepsdel,nx,ny,nz,npad,oz,dz,wbottom,v0,eps0,shotid,pct,max_shot_per_job,icall,command1);

  icall++;
  
  fprintf(stderr,"icall %d objfunc %.16f\n",icall,f);

  write("iv",v,nxyz,std::ios_base::app);
  to_header("iv","n4",icall,"o4",0.,"d4",1.);
  
  write("ieps",eps,nxyz,std::ios_base::app);
  to_header("ieps","n4",icall,"o4",0.,"d4",1.);
  
  write("idel",del,nxyz,std::ios_base::app);
  to_header("idel","n4",icall,"o4",0.,"d4",1.);
  
//  write("gv",gv,nxyz,std::ios_base::app);
//  to_header("gv","n4",icall,"o4",0.,"d4",1.);
//  
//  write("geps",geps,nxyz,std::ios_base::app);
//  to_header("geps","n4",icall,"o4",0.,"d4",1.);
//  
//  write("gdel",gdel,nxyz,std::ios_base::app);
//  to_header("gdel","n4",icall,"o4",0.,"d4",1.);
  
  lbfgs(nn,mm,vepsdel,f,gvepsdel,diagco,diag,w,iflag,isave,dsave);

  if(iflag<=0 || icall>nfg){
   fprintf(stderr,"iflag %d\n",iflag);
   break;
  }
 }
 
 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 delete []vepsdel;delete []cij;delete []gvepsdel;delete []diag;delete []w;
 delete []m;
 delete []isave;delete []dsave;
 delete []diag;delete []w;

 myio_close();
 return 0;
}
