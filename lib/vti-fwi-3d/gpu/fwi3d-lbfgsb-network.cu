#include <cstdio>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <libssh/libssh.h>
#include <libssh/sftp.h>

#include "sshtunneling.h"

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"
#include "objFuncGradientCij3d_network.h"
#include "conversions.h"
#include "boundary.h"
#include "check.h"
#include "lbfgs.h"

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
 multiply(nbd,nbd,mask,nxyz);
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

 to_header("iv","n1",nx,"o1",ox,"d1",dx);
 to_header("iv","n2",ny,"o2",oy,"d2",dy);
 to_header("iv","n3",nz,"o3",oz,"d3",dz);
 
 to_header("ieps","n1",nx,"o1",ox,"d1",dx);
 to_header("ieps","n2",ny,"o2",oy,"d2",dy);
 to_header("ieps","n3",nz,"o3",oz,"d3",dz);
 
 to_header("idel","n1",nx,"o1",ox,"d1",dx);
 to_header("idel","n2",ny,"o2",oy,"d2",dy);
 to_header("idel","n3",nz,"o3",oz,"d3",dz);
 
 // remote server part 
 vector<string> server; get_sarray("server",server);
 int nserver=server.size();

 float **fgcij_server;
 ssh_session *my_ssh_session;
 sftp_session *my_sftp_session;
 
 if(nserver>0){
  fgcij_server=new float*[nserver]();
  my_ssh_session=new ssh_session[nserver]();
  my_sftp_session=new sftp_session[nserver]();
 }
 vector<string> workdir; get_sarray("remoteworkdir",workdir);
 vector<string> datapath; get_sarray("remotedatapath",datapath);
 vector<string> script; get_sarray("remotescript",script,";");
 vector<string> command; get_sarray("remotecommand",command,";");
 vector<string> script1,scriptpath,gradpath,outpath,command1;

 string homedir(getenv("HOME"));
 
 for(int i=0;i<nserver;i++){
     fgcij_server[i]=new float[nn+1];

     scriptpath.push_back(workdir[i]+"scripts/");
     gradpath.push_back(workdir[i]+"grads/");
     outpath.push_back(workdir[i]+"output/");
     script1.push_back("#!/usr/bin/env tcsh\n\ncd "+workdir[i]+"\n\n"+script[i]+" datapath="+datapath[i]);
     if(command[i].compare("/bin/bash")==0) command1.push_back(command[i]);
     else command1.push_back(workdir[i]+command[i]);
//     cout<<"remote workdir "<<workdir[i]<<endl;
//     cout<<"remote datapath "<<datapath[i]<<endl;
//     cout<<"remote scriptpath "<<scriptpath[i]<<endl;
//     cout<<"remote gradpath "<<gradpath[i]<<endl;
//     cout<<"remote outpath "<<outpath[i]<<endl;
//     cout<<"script1 "<<script1[i]<<endl;
//     cout<<"command1 "<<command1[i]<<endl;
 }

 vector<thread> threads;

 // cluster part 
 string cluster_command=get_s("command");
// int nshottotal; get_param("nshottotal",nshottotal);
 vector<int> badshot; get_array("badshot",badshot);
 vector<int> shotrange; get_array("shotrange",shotrange);
 vector<int> shotid;
 for(int i=shotrange[0];i<shotrange[1];i++){
  if(find(badshot.begin(),badshot.end(),i)==badshot.end()) shotid.push_back(i);
 }

 int max_shot_per_job=1;
 float pct=1.;
 get_param("max_shot_per_job",max_shot_per_job,"pct",pct);
    
 strcpy(task,"START");
 for(int i=5;i<60;++i) task[i]=' ';

 int icall=0,nnew=0;

 vector<float> time_in_min; get_array("time_in_min",time_in_min);
 vector<int> partition(nserver+1);

 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 while((task[0]=='F' && task[1]=='G') || 
       (task[0]=='N' && task[1]=='E' && task[2]=='W' && task[3]=='_' && task[4]=='X') || 
       (task[0]=='S' && task[1]=='T' && task[2]=='A' && task[3]=='R' && task[4]=='T')){
  
  setulb_(&nn,&mm,vepsdel,l,u,nbd,&f,gvepsdel,&factr,&pgtol,wa,iwa,task,&iprint,csave,lsave,isave,dsave,60,60);

  if(task[0]=='F' && task[1]=='G'){

//    server_job_partition(nshottotal,time_in_min,partition);
//    int rangeb=0,rangee;

    for(int i=0;i<nserver;i++){
//        rangee=rangeb+partition[i];
//        fprintf(stderr,"%s does %d shot from %d to %d\n",server[i].c_str(),partition[i],rangeb,rangee);
//        script1[i]+=" shotrange="+to_string(rangeb)+","+to_string(rangee);
//        rangeb=rangee;

       // Open session and set options
       my_ssh_session[i]=ssh_new();
       if (my_ssh_session[i] == NULL) exit(-1);
       cout<<"conneting to "<<server[i]<<endl;
       string ipaddr=get_s(homedir+"/.ipaddr",server[i]);
       ssh_options_set(my_ssh_session[i], SSH_OPTIONS_HOST, ipaddr.c_str());
       
       // Connect to server
       int rc = ssh_connect(my_ssh_session[i]);
       if (rc != SSH_OK){
         fprintf(stderr, "Error connecting to %s: %s\n",server[i].c_str(),ssh_get_error(my_ssh_session[i]));
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

//   rangee=rangeb+partition[nserver];
//   fprintf(stderr,"xstream does %d shot from %d to %d\n",partition[nserver],rangeb,rangee);
//   vector<int> shotid;
//   for(int i=rangeb;i<rangee;i++){
//    if(find(badshot.begin(),badshot.end(),i)==badshot.end()) shotid.push_back(i);
//   }

   checkEpsDel(eps,del,eps0,1.,nxyz,m);
   VEpsDel2Cij(c11,c13,c33,v,eps,del,v0,eps0,1.,nxyz);
   
   string cijfile="cij_icall_"+to_string(icall)+".H";
  
   for(int i=0;i<nserver;i++) threads.push_back(thread(objFuncGradientCij3d_network,fgcij_server[i],cij,nx,ny,nz,ox,oy,oz,dx,dy,dz,std::ref(cijfile),std::ref(script1[i]),std::ref(scriptpath[i]),std::ref(gradpath[i]),std::ref(outpath[i]),std::ref(datapath[i]),std::ref(command1[i]),icall,std::ref(my_ssh_session[i]),std::ref(my_sftp_session[i]),std::ref(time_in_min[i])));
   
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
  
   string cluster_command1=cluster_command+" cij="+cijfile;
  
   if(shotid.size()>0){
     objFuncGradientCij_cluster(fgcij,nx,ny,nz,shotid,pct,max_shot_per_job,icall,cluster_command1,time_in_min[nserver]);
     fprintf(stderr,"xs took %f minutes\n",time_in_min[nserver]);
   }
   
   for(int i=0;i<nserver;i++){
     threads[i].join();
     fprintf(stderr,"%s took %f minutes\n",server[i].c_str(),time_in_min[i]);
   }
   if(!threads.empty()) threads.erase(threads.begin(),threads.end());
  
   for(int i=0;i<nserver;i++){
      #pragma omp parallel for
      for(size_t j=0;j<nn+1;j++) fgcij[j]+=fgcij_server[i][j];
   }
   
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
   
   multiply(gv,gv,mask,nxyz);
   multiply(geps,geps,mask,nxyz);
   multiply(gdel,gdel,mask,nxyz);

   icall++;

   for(int i=0;i<nserver;i++){
     sftp_free(my_sftp_session[i]);
     ssh_disconnect(my_ssh_session[i]);
     ssh_free(my_ssh_session[i]);
   }

//   for(int i=0;i<nserver+1;i++) time_in_min[i]=time_in_min[i]/partition[i];
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
 
 for(int i=0;i<nserver;i++){
     delete []fgcij_server[i];
 }
 if(nserver>0){
  delete []fgcij_server;
  delete []my_ssh_session;delete []my_sftp_session;
 }
 delete []vepsdel;delete []cij;delete []gvepsdel;delete []fgcij;delete []mask;
 delete []m;
 delete []nbd;delete []iwa;delete []l;delete []u;delete []wa;

 myio_close();
 return 0;
}
