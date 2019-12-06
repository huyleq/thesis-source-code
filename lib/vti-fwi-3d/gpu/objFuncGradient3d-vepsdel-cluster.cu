#include <cstdio>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"
#include "conversions.h"
#include "boundary.h"
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
 
 long long nxy=nx*ny,nxyz=nxy*nz;

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

 int max_shot_per_job=1;
 float pct=1.;
 get_param("max_shot_per_job",max_shot_per_job,"pct",pct);
    
 long long nn=3*nxyz;
 float f;
 float *vepsdel=new float[nn]; 
 float *v=vepsdel,*eps=vepsdel+nxyz,*del=vepsdel+2*nxyz;
 float *m=new float[nxyz]();
 float *cij=new float[nn];
 float *c11=cij,*c13=cij+nxyz,*c33=cij+2*nxyz;
 float *gvepsdel=new float[nn];
 float *gv=gvepsdel,*geps=gvepsdel+nxyz,*gdel=gvepsdel+2*nxyz;
 
 float v0=1.,eps0=1.,wbottom=0.;
 get_param("v0",v0,"eps0",eps0,"wbottom",wbottom);
 read("v",v,nxyz);
 scale(v,v,1./v0,nxyz);
 read("eps",eps,nxyz);
 scale(eps,eps,1./eps0,nxyz);
 read("del",del,nxyz);

 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 int icall;
 get_param("icall",icall);

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

// f=objFuncGradientVEpsDel_cluster(gvepsdel,vepsdel,nx,ny,nz,npad,oz,dz,wbottom,v0,eps0,shotid,pct,max_shot_per_job,icall,command1);

 float *fgcij=new float[nn+1];
 objFuncGradientCij_cluster(fgcij,nx,ny,nz,shotid,pct,max_shot_per_job,icall,command1);
 
 f=fgcij[0];
 
 float *gcij=fgcij+1;
 float *gc11=gcij,*gc13=gcij+nxyz,*gc33=gcij+2*nxyz;
 
 zeroBoundary(gc11,nx,ny,nz,npad);
 zeroBoundary(gc13,nx,ny,nz,npad);
 zeroBoundary(gc33,nx,ny,nz,npad);
 
 int nwbottom=(wbottom-oz)/dz+1-npad;
 memset(gc11+npad*nxy,0,nwbottom*nxy*sizeof(float));
 memset(gc13+npad*nxy,0,nwbottom*nxy*sizeof(float));
 memset(gc33+npad*nxy,0,nwbottom*nxy*sizeof(float));
 
 GradCij2GradVEpsDel(gv,geps,gdel,gc11,gc13,gc33,v,eps,del,v0,eps0,1.,nxyz);
 delete []fgcij;

 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 fprintf(stderr,"objfunc is %10.16f\n",f); 
 
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

 delete []vepsdel;delete []cij;delete []gvepsdel;
 delete []m;

 myio_close();
 return 0;
}
