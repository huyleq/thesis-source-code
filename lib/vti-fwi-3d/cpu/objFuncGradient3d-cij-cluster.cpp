#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

#include "myio.h"
#include "cluster.h"
#include "cees_job_submit.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);

 int nx,ny,nz;
 float ox,oy,oz,dx,dy,dz;
 
 from_header("cij","n1",nx,"o1",ox,"d1",dx);
 from_header("cij","n2",ny,"o2",oy,"d2",dy);
 from_header("cij","n3",nz,"o3",oz,"d3",dz);
 
 long long nxy=nx*ny,nxyz=nxy*nz,nn=3*nxyz;

 string command=get_s("exec");
 command+=" par="+get_s("par")+" wavelet="+get_s("wavelet");
 command+=" souloc="+get_s("souloc")+" recloc="+get_s("recloc");
 command+=" padboundary="+get_s("padboundary")+" randomboundary="+get_s("randomboundary");
 command+=" cij="+get_s("cij")+" data="+get_s("data")+" datapath="+get_s("datapath");
 string workdir=get_s("workdir");

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

 int max_shot_per_job=1;
 float pct=1.;
 get_param("max_shot_per_job",max_shot_per_job,"pct",pct);
    
 int icall;
 get_param("icall",icall);

 float *fgcij=new float[nn+1];
 float *gcij=fgcij+1;
 
 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 objFuncGradientCij_cees_cluster(fgcij,nx,ny,nz,shotid,pct,max_shot_per_job,icall,command,workdir);
 
 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
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

 fprintf(stderr,"objfunc is %10.16f\n",fgcij[0]); 
 delete []fgcij;
 
 myio_close();
 return 0;
}
