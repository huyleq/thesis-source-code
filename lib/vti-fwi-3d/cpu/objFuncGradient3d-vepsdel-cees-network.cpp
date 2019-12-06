#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <libssh/libssh.h>
#include <libssh/sftp.h>

#include "myio.h"
#include "mylib.h"
#include "cluster.h"
#include "conversions.h"
#include "boundary.h"
#include "check.h"
#include "sshtunneling.h"
#include "objFuncGradientCij3d_network.h"
#include "cees_job_submit.h"

using namespace std;

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
 
 float v0=1.,eps0=1.,wbottom=0.;
 get_param("v0",v0,"eps0",eps0,"wbottom",wbottom);
 read("v",v,nxyz);
 scale(v,v,1./v0,nxyz);
 if(!read("eps",eps,nxyz)) memset(eps,0,nxyz*sizeof(float));
 scale(eps,eps,1./eps0,nxyz);
 if(!read("del",del,nxyz)) memset(del,0,nxyz*sizeof(float));
 if(!read("mask",mask,nxyz)) set(mask,1.f,nxyz);

 float max_depth=4000.f,max_sigma=100.f;
 get_param("max_depth",max_depth,"max_sigma",max_sigma);
 int max_iz=(max_depth-oz)/dz+1;

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
    
 int icall; get_param("icall",icall);

 // remote server part 
 vector<string> server; get_sarray("server",server);
 int nserver=server.size();

 vector<string> remoteworkdir; get_sarray("remoteworkdir",remoteworkdir);
 vector<string> remotedatapath; get_sarray("remotedatapath",remotedatapath);
 vector<string> remotescript; get_sarray("remotescript",remotescript,";");
 vector<string> remotecommand; get_sarray("remotecommand",remotecommand,";");
 vector<string> remotescript1,scriptpath,gradpath,outpath,remotecommand1;

 float **fgcij_server;
 ssh_session *my_ssh_session;
 sftp_session *my_sftp_session;
 float *time_in_minute;
 
 if(nserver>0){
  fgcij_server=new float*[nserver]();
  my_ssh_session=new ssh_session[nserver]();
  my_sftp_session=new sftp_session[nserver]();
  time_in_minute=new float[nserver]();
 }

 string homedir(getenv("HOME"));
 
 for(int i=0;i<nserver;i++){
     fgcij_server[i]=new float[nn+1];

     scriptpath.push_back(remoteworkdir[i]+"scripts/");
     gradpath.push_back(remoteworkdir[i]+"grads/");
     outpath.push_back(remoteworkdir[i]+"output/");
     remotescript1.push_back("#!/usr/bin/env tcsh\n\ncd "+remoteworkdir[i]+"\n\n"+remotescript[i]+" datapath="+remotedatapath[i]);
     if(remotecommand[i].compare("/bin/bash")==0) remotecommand1.push_back(remotecommand[i]);
     else remotecommand1.push_back(remoteworkdir[i]+remotecommand[i]);
     
//     cout<<"remote workdir "<<workdir[i]<<endl;
//     cout<<"remote datapath "<<datapath[i]<<endl;
//     cout<<"remote scriptpath "<<scriptpath[i]<<endl;
//     cout<<"remote gradpath "<<gradpath[i]<<endl;
//     cout<<"remote outpath "<<outpath[i]<<endl;
//     cout<<"script1 "<<script1[i]<<endl;
//     cout<<"command1 "<<command1[i]<<endl;
    
     // Open session and set options
     my_ssh_session[i]=ssh_new();
     if (my_ssh_session[i] == NULL) exit(-1);
     cout<<"conneting to "<<server[i]<<endl;
     string ipaddr=get_s(homedir+"/.ipaddr",server[i]);
     ssh_options_set(my_ssh_session[i], SSH_OPTIONS_HOST, ipaddr.c_str());
     
     // Connect to server
     int rc = ssh_connect(my_ssh_session[i]);
     if (rc != SSH_OK){
       fprintf(stderr, "Error connecting to %s. Error code %d: %s\n",server[i].c_str(),ssh_get_error_code(my_ssh_session[i]),ssh_get_error(my_ssh_session[i]));
       ssh_free(my_ssh_session[i]);
       exit(-1);
     }
    
     // Authenticate ourselves
     string pass=get_s(homedir+"/.pass",server[i]);
     rc = ssh_userauth_password(my_ssh_session[i], NULL, pass.c_str());
     if (rc != SSH_AUTH_SUCCESS){
       fprintf(stderr, "Error authenticating with password when connectin to %s: %s\n",server[i].c_str(),ssh_get_error(my_ssh_session[i]));
       ssh_disconnect(my_ssh_session[i]);
       ssh_free(my_ssh_session[i]);
       exit(-1);
     }
    
     my_sftp_session[i] = sftp_new(my_ssh_session[i]);
     if (my_sftp_session[i] == NULL) {
       fprintf(stderr, "Error allocating SFTP session: %s\n",ssh_get_error(my_ssh_session[i]));
       return SSH_ERROR;
     }
     rc = sftp_init(my_sftp_session[i]);
     if (rc != SSH_OK) {
       fprintf(stderr, "Error initializing SFTP session: %d.\n",sftp_get_error(my_sftp_session[i]));
       sftp_free(my_sftp_session[i]);
       return rc;
     }
 }

 vector<thread> threads;

 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 checkEpsDel(eps,del,eps0,1.,nxyz,m);
 VEpsDel2Cij(c11,c13,c33,v,eps,del,v0,eps0,1.,nxyz);
 
 string cijfile="cij_icall_"+to_string(icall)+".H";

 for(int i=0;i<nserver;i++) threads.push_back(thread(objFuncGradientCij3d_network,fgcij_server[i],cij,nx,ny,nz,ox,oy,oz,dx,dy,dz,std::ref(cijfile),std::ref(remotescript1[i]),std::ref(scriptpath[i]),std::ref(gradpath[i]),std::ref(outpath[i]),std::ref(remotedatapath[i]),std::ref(remotecommand1[i]),icall,std::ref(my_ssh_session[i]),std::ref(my_sftp_session[i]),std::ref(time_in_minute[i])));
 
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

 command+=" cij="+cijfile;

 objFuncGradientCij_cees_cluster(fgcij,nx,ny,nz,shotid,pct,max_shot_per_job,icall,command,workdir);

 for(int i=0;i<nserver;i++) threads[i].join();

 for(int i=0;i<nserver;i++){
    #pragma omp parallel for
    for(size_t j=0;j<nn+1;j++) fgcij[j]+=fgcij_server[i][j];
 }
 
 zeroBoundary(gc11,nx,ny,nz,npad);
 zeroBoundary(gc13,nx,ny,nz,npad);
 zeroBoundary(gc33,nx,ny,nz,npad);
 
 int nwbottom=(wbottom-oz)/dz+1-npad;
 memset(gc11+npad*nxy,0,nwbottom*nxy*sizeof(float));
 memset(gc13+npad*nxy,0,nwbottom*nxy*sizeof(float));
 memset(gc33+npad*nxy,0,nwbottom*nxy*sizeof(float));
 
 GradCij2GradVEpsDel(gv,geps,gdel,gc11,gc13,gc33,v,eps,del,v0,eps0,1.,nxyz);
 
 multiply(gv,gv,mask,nxyz);
 multiply(geps,geps,mask,nxyz);
 multiply(gdel,gdel,mask,nxyz);

 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 float f=fgcij[0];
 fprintf(stderr,"objfunc=%.10f\n",f);

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

 for(int i=0;i<nserver;i++){
     delete []fgcij_server[i];
     delete []time_in_minute;
 }
 if(nserver>0){
  delete []fgcij_server;
  delete []my_ssh_session;delete []my_sftp_session;
 }
 
 delete []vepsdel;delete []cij;delete []gvepsdel;delete []fgcij;delete []mask;
 delete []m;
 
 myio_close();

 return 0;
}
