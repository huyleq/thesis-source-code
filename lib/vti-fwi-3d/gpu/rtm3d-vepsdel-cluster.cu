#include <cstdio>
#include <chrono>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"
#include "conversions.h"
#include "check.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);

 int nx,ny,nz,npad;
 float ox,oy,oz,dx,dy,dz;
 
 from_header("vrandom","n1",nx,"o1",ox,"d1",dx);
 from_header("vrandom","n2",ny,"o2",oy,"d2",dy);
 from_header("vrandom","n3",nz,"o3",oz,"d3",dz);
 get_param("npad",npad);
 
 long long nxyz=nx*ny*nz;

 string command=get_s("command");

 vector<int> shotid;
 vector<int> shotrange; get_array("shotrange",shotrange);
 vector<int> badshot; get_array("badshot",badshot);
 for(int i=shotrange[0];i<shotrange[1];i++){
  if(find(badshot.begin(),badshot.end(),i)==badshot.end()) shotid.push_back(i);
 }

 int max_shot_per_job=1;
 get_param("max_shot_per_job",max_shot_per_job);
    
 long long nn=3*nxyz;
 float *vepsdel=new float[nn]; 
 float *v=vepsdel,*eps=vepsdel+nxyz,*del=vepsdel+2*nxyz;
 float *m=new float[nxyz]();
 float *cij=new float[nn];
 float *c11=cij,*c13=cij+nxyz,*c33=cij+2*nxyz;
 float *image=new float[nxyz];
 
 float wbottom;
 get_param("wbottom",wbottom);
 read("vrandom",v,nxyz);
 if(!read("eps",eps,nxyz)) memset(eps,0,nxyz*sizeof(float));
 if(!read("del",del,nxyz)) memset(del,0,nxyz*sizeof(float));

 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 int icall;
 get_param("icall",icall);

 checkEpsDel(eps,del,1.,1.,nxyz,m);
 VEpsDel2Cij(c11,c13,c33,v,eps,del,1.,1.,1.,nxyz);
 
 string cijfile="cij_icall_"+to_string(icall)+".H";
 writeToHeader(cijfile,cij,nn);
 
 delete []vepsdel;delete []cij;
 delete []m;
 
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

 rtm3d(image,nx,ny,nz,npad,oz,dz,wbottom,shotid,max_shot_per_job,icall,command1);

 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 write("image",image,nxyz);
 to_header("image","n1",nx,"o1",ox,"d1",dx);
 to_header("image","n2",ny,"o2",oy,"d2",dy);
 to_header("image","n3",nz,"o3",oz,"d3",dz);
 delete []image;

 myio_close();
 return 0;
}
