#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

#include "myio.h"
#include "mylib.h"
#include "cluster.h"
#include "conversions.h"
#include "boundary.h"
#include "check.h"
#include "gaussian_filter.h"
#include "cees_job_submit.h"


using namespace std;

extern "C"{
 void setulb_(long long *n,long long *m,float *x,float *l,float *u,long long *nbd,float *f,float *g,float *factr,float *pgtol,float *wa,long long *iwa,char task[],long long *iprint,char csave[],long long lsave[],long long isave[],float dsave[],unsigned int tasklen,unsigned int csavelen);
}

int main(int argc,char **argv){
 myio_init(argc,argv);

 int nx,ny,nz,npad;
 float ox,oy,oz,dx,dy,dz;
 
 from_header("v","n1",nx,"o1",ox,"d1",dx);
 from_header("v","n2",ny,"o2",oy,"d2",dy);
 from_header("v","n3",nz,"o3",oz,"d3",dz);
 get_param("npad",npad);
 
 long long nxy=nx*ny,nxyz=nxy*nz,nn=3*nxyz;
 
 float *m=new float[nxyz]();
 
 long long mm=5,iprint=-1,lsave[4],isave[44];
 float factr=0.,pgtol=0.,f,dsave[29];
 char task[60],csave[60];
 long long *nbd=new long long[nn]();
 long long *iwa=new long long[3*nn]();
 float *l=new float[nn](); 
 bool lower=read("lv",l,nxyz);
 float *u=new float[nn]();;
 bool upper=read("uv",u,nxyz);
 if(lower && !upper) set(nbd,1,nxyz);
 if(lower && upper) set(nbd,2,nxyz);
 if(!lower && upper) set(nbd,3,nxyz);
 float *wa=new float[2*mm*nn+5*nn+11*mm*mm+8*mm]();

// float *lv=l,*leps=l+nxyz,*ldel=l+2*nxyz;
// float *uv=u,*ueps=u+nxyz,*udel=u+2*nxyz;

 float *vepsdel=new float[nn]; 
 float *v=vepsdel,*eps=vepsdel+nxyz,*del=vepsdel+2*nxyz;
 float *gvepsdel=new float[nn];
 float *gv=gvepsdel,*geps=gvepsdel+nxyz,*gdel=gvepsdel+2*nxyz;
 
 float *cij=new float[nn];
 float *c11=cij,*c13=cij+nxyz,*c33=cij+2*nxyz;
 float *fgcij=new float[nn+1];
 float *gcij=fgcij+1;
 float *gc11=gcij,*gc13=gcij+nxyz,*gc33=gcij+2*nxyz;
 float *mask=new float[nxyz];
 
 int nfg; get_param("nfg",nfg);
 
 float v0=1.,eps0=1.,wbottom=0.;
 get_param("v0",v0,"eps0",eps0,"wbottom",wbottom);
 read("v",v,nxyz);
 scale(v,v,1./v0,nxyz);
 if(lower) scale(l,l,1./v0,nxyz);
 if(upper) scale(u,u,1./v0,nxyz);
 if(!read("eps",eps,nxyz)) memset(eps,0,nxyz*sizeof(float));
 scale(eps,eps,1./eps0,nxyz);
 if(!read("del",del,nxyz)) memset(del,0,nxyz*sizeof(float));
 if(!read("mask",mask,nxyz)) set(mask,1.f,nxyz);
 multiply(nbd,nbd,mask,nxyz);

 float max_depth=3500.f,max_sigma=87.5f;
 get_param("max_depth",max_depth,"max_sigma",max_sigma);
 int max_iz=(max_depth-oz)/dz+1;

 to_header("iv","n1",nx,"o1",ox,"d1",dx);
 to_header("iv","n2",ny,"o2",oy,"d2",dy);
 to_header("iv","n3",nz,"o3",oz,"d3",dz);
 
 to_header("ieps","n1",nx,"o1",ox,"d1",dx);
 to_header("ieps","n2",ny,"o2",oy,"d2",dy);
 to_header("ieps","n3",nz,"o3",oz,"d3",dz);
 
 to_header("idel","n1",nx,"o1",ox,"d1",dx);
 to_header("idel","n2",ny,"o2",oy,"d2",dy);
 to_header("idel","n3",nz,"o3",oz,"d3",dz);
 
 int ns;
 from_header("souloc","n2",ns);

 vector<int> shotid;
 bool providedShotId=get_array("shotid",shotid);
 if(!providedShotId){
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

 string command=get_s("exec");
 command+=" par="+get_s("par")+" wavelet="+get_s("wavelet");
 command+=" souloc="+get_s("souloc")+" recloc="+get_s("recloc");
 command+=" padboundary="+get_s("padboundary")+" randomboundary="+get_s("randomboundary");
 command+=" data="+get_s("data")+" datapath="+get_s("datapath");
 string workdir=get_s("workdir");

 int max_shot_per_job=1;
 float pct=1.;
 get_param("max_shot_per_job",max_shot_per_job,"pct",pct);
    
 strcpy(task,"START");
 for(int i=5;i<60;++i) task[i]=' ';

 int icall=0,nnew=0;

 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 while((task[0]=='F' && task[1]=='G') || 
       (task[0]=='N' && task[1]=='E' && task[2]=='W' && task[3]=='_' && task[4]=='X') || 
       (task[0]=='S' && task[1]=='T' && task[2]=='A' && task[3]=='R' && task[4]=='T')){
  
  setulb_(&nn,&mm,vepsdel,l,u,nbd,&f,gvepsdel,&factr,&pgtol,wa,iwa,task,&iprint,csave,lsave,isave,dsave,60,60);

  if(task[0]=='F' && task[1]=='G'){
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
  
   objFuncGradientCij_cees_cluster(fgcij,nx,ny,nz,shotid,pct,max_shot_per_job,icall,command1,workdir);

   f=fgcij[0];
   fprintf(stderr,"icall=%d f=%.10f\n",icall,f);
  
   zeroBoundary(gc11,nx,ny,nz,npad);
   zeroBoundary(gc13,nx,ny,nz,npad);
   zeroBoundary(gc33,nx,ny,nz,npad);
   
   int nwbottom=(wbottom-oz)/dz+1-npad;
   memset(gc11+npad*nxy,0,nwbottom*nxy*sizeof(float));
   memset(gc13+npad*nxy,0,nwbottom*nxy*sizeof(float));
   memset(gc33+npad*nxy,0,nwbottom*nxy*sizeof(float));
   
   GradCij2GradVEpsDel(gv,geps,gdel,gc11,gc13,gc33,v,eps,del,v0,eps0,1.,nxyz);
   
   smooth_gradient(gv,nx,ny,nz,npad,nwbottom,max_iz,max_sigma,dx);
   smooth_gradient(geps,nx,ny,nz,npad,nwbottom,max_iz,max_sigma,dx);
   smooth_gradient(gdel,nx,ny,nz,npad,nwbottom,max_iz,max_sigma,dx);

   multiply(gv,gv,mask,nxyz);
   multiply(geps,geps,mask,nxyz);
   multiply(gdel,gdel,mask,nxyz);

   icall++;
  } 
  else{
   if(task[0]=='N' && task[1]=='E' && task[2]=='W' && task[3]=='_' && task[4]=='X'){
	nnew++;

    fprintf(stderr,"New x %d iterate %d nfg=%d f=%.10f |proj g|=%.10f\n",nnew,isave[29],isave[33],f,dsave[12]);
    
    write("objfunc",&f,1,std::ios_base::app);
    to_header("objfunc","n1",nnew,"o1",0.,"d1",1.);

    write("iv",v,nxyz,std::ios_base::app);
    to_header("iv","n4",nnew,"o4",0.,"d4",1.);

    write("ieps",eps,nxyz,std::ios_base::app);
    to_header("ieps","n4",nnew,"o4",0.,"d4",1.);
    
    write("idel",del,nxyz,std::ios_base::app);
    to_header("idel","n4",nnew,"o4",0.,"d4",1.);
 
	if(isave[33]>=nfg){
	 fprintf(stderr,"STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT");
	 break; 
	}
   
    if(dsave[12]<=1e-10*(1+fabs(f))){
	 fprintf(stderr,"STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL");
	 break; 
	}
   }
  }
 }
 
 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 delete []vepsdel;delete []cij;delete []gvepsdel;delete []fgcij;delete []mask;
 delete []m;
 delete []nbd;delete []iwa;delete []l;delete []u;delete []wa;
 
 myio_close();

 return 0;
}
