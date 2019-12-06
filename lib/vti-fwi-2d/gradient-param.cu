#include <cmath>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>

#include "myio.h"
#include "mylib.h"
#include "init.h"
#include "wave.h"

using namespace std;

int main(int argc,char **argc11){
 myio_init(argc,argc11);

int nnx,nnz,nx,nz,nt,npad;
 float ox,oz,ot,dx,dz,dt;
 init2d(nnx,nnz,nx,nz,nt,dx,dz,dt,ox,oz,ot,npad);
 int nnxz=nnx*nnz;

 float *wavelet=new float[nt];
 read("wavelet",wavelet,nt);
 
 int ns;
 from_header("souloc","n2",ns);
 float *souloc=new float[ns*4]();
 read("souloc",souloc,ns*4);
 int *sloc=new int[ns*4]();
 #pragma omp parallel for num_threads(16)
 for(int is=0;is<ns;is++){
  sloc[is*4+0]=(souloc[is*4+0]-ox)/dx+0.5+npad;
  sloc[is*4+1]=(souloc[is*4+1]-oz)/dz+0.5+npad;
  sloc[is*4+2]=souloc[is*4+2];
  sloc[is*4+3]=souloc[is*4+3];
 }

 int nr;
 from_header("recloc","n2",nr);
 float *recloc=new float[nr*2]();
 read("recloc",recloc,nr*2);
 int *rloc=new int[nr*2]();
 #pragma omp parallel for num_threads(16)
 for(int ir=0;ir<nr;ir++){
  rloc[ir*2+0]=(recloc[ir*2+0]-ox)/dx+0.5+npad;
  rloc[ir*2+1]=(recloc[ir*2+1]-oz)/dz+0.5+npad;
 }

 float rate;
 get_param("rate",rate);
 int ratio=rate/dt+0.5;
 int ntNeg=std::round(abs(ot)/dt);
 int nnt_data=(nt-ntNeg-1)/ratio+1;
 float *data=new float[nr*nnt_data]();
 read("data",data,nr*nnt_data);

 float *taper=new float[nnxz]();
 init_abc(taper,nx,nz,npad);
 
 float wbottom; get_param("wbottom",wbottom);

 int padded; get_param("padded",padded);
 
 float *m=new float[nnxz]();

 string parameter=get_s("parameter");

 if(parameter.compare("vepsdel")==0){
  float *v=new float[nnxz]();
  float *eps=new float[nnxz]();
  float *del=new float[nnxz]();
  
  if(padded==0){
   init_model("v",v,nx,nz,npad); 
   init_model("eps",eps,nx,nz,npad); 
   init_model("del",del,nx,nz,npad); 
  } 
  else{ 
   read("v",v,nnxz);
   read("eps",eps,nnxz);
   read("del",del,nnxz);
  }
 
  float v0,eps0,del0;
  get_param("v0",v0,"eps0",eps0,"del0",del0);
 
  scale(v,v,1./v0,nnxz);
  scale(eps,eps,1./eps0,nnxz);
  scale(del,del,1./del0,nnxz);
  
  float *gv=new float[nnxz]();
  float *geps=new float[nnxz]();
  float *gdel=new float[nnxz]();
  
  chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
  float val=objFuncGradientVEpsDel(gv,geps,gdel,data,v,eps,del,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  
  chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
  chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
  cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
  cout<<"Objfunc "<<val<<endl;

  float agv=fabs(gv[0]),av=fabs(v[0]);
  float age=fabs(geps[0]),ae=fabs(eps[0]);
  float agd=fabs(gdel[0]),ad=fabs(del[0]);
  for(int i=0;i<nnxz;i++){
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
 
  write("gv",gv,nnxz);
  to_header("gv","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gv","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("geps",geps,nnxz);
  to_header("geps","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("geps","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gdel",gdel,nnxz);
  to_header("gdel","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gdel","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []v; delete []eps; delete []del;
  delete []gv; delete []geps; delete []gdel;
 }
 else if(parameter.compare("vnetadel")==0){
  float *vn=new float[nnxz]();
  float *eta=new float[nnxz]();
  float *del=new float[nnxz]();
  
  if(padded==0){
   init_model("vn",vn,nx,nz,npad); 
   init_model("eta",eta,nx,nz,npad); 
   init_model("del",del,nx,nz,npad); 
  } 
  else{ 
   read("vn",vn,nnxz);
   read("eta",eta,nnxz);
   read("del",del,nnxz);
  }
 
  float vn0,eta0,del0;
  get_param("vn0",vn0,"eta0",eta0,"del0",del0);
 
  scale(vn,vn,1./vn0,nnxz);
  scale(eta,eta,1./eta0,nnxz);
  scale(del,del,1./del0,nnxz);
  
  float *gvn=new float[nnxz]();
  float *geta=new float[nnxz]();
  float *gdel=new float[nnxz]();
  
  chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
  float val=objFuncGradientVnEtaDel(gvn,geta,gdel,data,vn,eta,del,vn0,eta0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  
  chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
  chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
  cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
  cout<<"Objfunc "<<val<<endl;

  float agv=fabs(gvn[0]),av=fabs(vn[0]);
  float age=fabs(geta[0]),ae=fabs(eta[0]);
  float agd=fabs(gdel[0]),ad=fabs(del[0]);
  for(int i=0;i<nnxz;i++){
      if(fabs(gvn[i])>agv) agv=fabs(gvn[i]);
      if(fabs(vn[i])>av) av=fabs(vn[i]);
      if(fabs(geta[i])>age) age=fabs(geta[i]);
      if(fabs(eta[i])>ae) ae=fabs(eta[i]);
      if(fabs(gdel[i])>agd) agd=fabs(gdel[i]);
      if(fabs(del[i])>ad) ad=fabs(del[i]);
  }

  if(ae==0.) get_param("maxeta",ae);
  if(ad==0.) get_param("maxdel",ad);

  vn0=sqrt((agd/ad)/(agv/av));
  eta0=sqrt((agd/ad)/(age/ae));

  cout<<"vn0 should be "<<vn0<<endl;
  cout<<"eta0 should be "<<eta0<<endl;
  cout<<"del0 should be 1"<<endl;
 
  write("gvn",gvn,nnxz);
  to_header("gvn","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gvn","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("geta",geta,nnxz);
  to_header("geta","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("geta","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gdel",gdel,nnxz);
  to_header("gdel","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gdel","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []vn; delete []eta; delete []del;
  delete []gvn; delete []geta; delete []gdel;
 }
 else if(parameter.compare("vhepseta")==0){
  float *vh=new float[nnxz]();
  float *eps=new float[nnxz]();
  float *eta=new float[nnxz]();
  
  if(padded==0){
   init_model("vh",vh,nx,nz,npad); 
   init_model("eps",eps,nx,nz,npad); 
   init_model("eta",eta,nx,nz,npad); 
  } 
  else{ 
   read("vh",vh,nnxz);
   read("eps",eps,nnxz);
   read("eta",eta,nnxz);
  }
 
  float vh0,eps0,eta0;
  get_param("vh0",vh0,"eps0",eps0,"eta0",eta0);
 
  scale(vh,vh,1./vh0,nnxz);
  scale(eps,eps,1./eps0,nnxz);
  scale(eta,eta,1./eta0,nnxz);
  
  float *gvh=new float[nnxz]();
  float *geps=new float[nnxz]();
  float *geta=new float[nnxz]();
  
  chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
  float val=objFuncGradientVhEpsEta(gvh,geps,geta,data,vh,eps,eta,vh0,eps0,eta0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  
  chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
  chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
  cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
  cout<<"Objfunc "<<val<<endl;

  float agv=fabs(gvh[0]),av=fabs(vh[0]);
  float age=fabs(geps[0]),ae=fabs(eps[0]);
  float agd=fabs(geta[0]),ad=fabs(eta[0]);
  for(int i=0;i<nnxz;i++){
      if(fabs(gvh[i])>agv) agv=fabs(gvh[i]);
      if(fabs(vh[i])>av) av=fabs(vh[i]);
      if(fabs(geps[i])>age) age=fabs(geps[i]);
      if(fabs(eps[i])>ae) ae=fabs(eps[i]);
      if(fabs(geta[i])>agd) agd=fabs(geta[i]);
      if(fabs(eta[i])>ad) ad=fabs(eta[i]);
  }

  if(ae==0.) get_param("maxeps",ae);
  if(ad==0.) get_param("maxeta",ad);

  vh0=sqrt((agd/ad)/(agv/av));
  eps0=sqrt((agd/ad)/(age/ae));

  cout<<"vh0 should be "<<vh0<<endl;
  cout<<"eps0 should be "<<eps0<<endl;
  cout<<"eta0 should be 1"<<endl;
 
  write("gvh",gvh,nnxz);
  to_header("gvh","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gvh","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("geps",geps,nnxz);
  to_header("geps","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("geps","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("geta",geta,nnxz);
  to_header("geta","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("geta","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []vh; delete []eps; delete []eta;
  delete []gvh; delete []geps; delete []geta;
 }
 else if(parameter.compare("vhepsdel")==0){
  float *vh=new float[nnxz]();
  float *eps=new float[nnxz]();
  float *del=new float[nnxz]();
  
  if(padded==0){
   init_model("vh",vh,nx,nz,npad); 
   init_model("eps",eps,nx,nz,npad); 
   init_model("del",del,nx,nz,npad); 
  } 
  else{ 
   read("vh",vh,nnxz);
   read("eps",eps,nnxz);
   read("del",del,nnxz);
  }
 
  float vh0,eps0,del0;
  get_param("vh0",vh0,"eps0",eps0,"del0",del0);
 
  scale(vh,vh,1./vh0,nnxz);
  scale(eps,eps,1./eps0,nnxz);
  scale(del,del,1./del0,nnxz);
  
  float *gvh=new float[nnxz]();
  float *geps=new float[nnxz]();
  float *gdel=new float[nnxz]();
  
  chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
  float val=objFuncGradientVhEpsDel(gvh,geps,gdel,data,vh,eps,del,vh0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  
  chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
  chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
  cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
  cout<<"Objfunc "<<val<<endl;

  float agv=fabs(gvh[0]),av=fabs(vh[0]);
  float age=fabs(geps[0]),ae=fabs(eps[0]);
  float agd=fabs(gdel[0]),ad=fabs(del[0]);
  for(int i=0;i<nnxz;i++){
      if(fabs(gvh[i])>agv) agv=fabs(gvh[i]);
      if(fabs(vh[i])>av) av=fabs(vh[i]);
      if(fabs(geps[i])>age) age=fabs(geps[i]);
      if(fabs(eps[i])>ae) ae=fabs(eps[i]);
      if(fabs(gdel[i])>agd) agd=fabs(gdel[i]);
      if(fabs(del[i])>ad) ad=fabs(del[i]);
  }

  if(ae==0.) get_param("maxeps",ae);
  if(ad==0.) get_param("maxdel",ad);

  vh0=sqrt((agd/ad)/(agv/av));
  eps0=sqrt((agd/ad)/(age/ae));

  cout<<"vh0 should be "<<vh0<<endl;
  cout<<"eps0 should be "<<eps0<<endl;
  cout<<"del0 should be 1"<<endl;
 
  write("gvh",gvh,nnxz);
  to_header("gvh","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gvh","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("geps",geps,nnxz);
  to_header("geps","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("geps","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gdel",gdel,nnxz);
  to_header("gdel","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gdel","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []vh; delete []eps; delete []del;
  delete []gvh; delete []geps; delete []gdel;
 }
 else if(parameter.compare("vvhdel")==0){
  float *v=new float[nnxz]();
  float *vh=new float[nnxz]();
  float *del=new float[nnxz]();
  
  if(padded==0){
   init_model("v",v,nx,nz,npad); 
   init_model("vh",vh,nx,nz,npad); 
   init_model("del",del,nx,nz,npad); 
  } 
  else{ 
   read("v",v,nnxz);
   read("vh",vh,nnxz);
   read("del",del,nnxz);
  }
 
  float v0,vh0,del0;
  get_param("v0",v0,"vh0",vh0,"del0",del0);
 
  scale(v,v,1./v0,nnxz);
  scale(vh,vh,1./vh0,nnxz);
  scale(del,del,1./del0,nnxz);
  
  float *gv=new float[nnxz]();
  float *gvh=new float[nnxz]();
  float *gdel=new float[nnxz]();
  
  chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
  float val=objFuncGradientVVhDel(gv,gvh,gdel,data,v,vh,del,v0,vh0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  
  chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
  chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
  cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
  cout<<"Objfunc "<<val<<endl;

  float agv=fabs(gv[0]),av=fabs(v[0]);
  float age=fabs(gvh[0]),ae=fabs(vh[0]);
  float agd=fabs(gdel[0]),ad=fabs(del[0]);
  for(int i=0;i<nnxz;i++){
      if(fabs(gv[i])>agv) agv=fabs(gv[i]);
      if(fabs(v[i])>av) av=fabs(v[i]);
      if(fabs(gvh[i])>age) age=fabs(gvh[i]);
      if(fabs(vh[i])>ae) ae=fabs(vh[i]);
      if(fabs(gdel[i])>agd) agd=fabs(gdel[i]);
      if(fabs(del[i])>ad) ad=fabs(del[i]);
  }

  if(ae==0.) get_param("maxvh",ae);
  if(ad==0.) get_param("maxdel",ad);

  v0=sqrt((agd/ad)/(agv/av));
  vh0=sqrt((agd/ad)/(age/ae));

  cout<<"v0 should be "<<v0<<endl;
  cout<<"vh0 should be "<<vh0<<endl;
  cout<<"del0 should be 1"<<endl;
 
  write("gv",gv,nnxz);
  to_header("gv","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gv","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gvh",gvh,nnxz);
  to_header("gvh","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gvh","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gdel",gdel,nnxz);
  to_header("gdel","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gdel","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []v; delete []vh; delete []del;
  delete []gv; delete []gvh; delete []gdel;
 }
 else if(parameter.compare("vnvhdel")==0){
  float *vn=new float[nnxz]();
  float *vh=new float[nnxz]();
  float *del=new float[nnxz]();
  
  if(padded==0){
   init_model("vn",vn,nx,nz,npad); 
   init_model("vh",vh,nx,nz,npad); 
   init_model("del",del,nx,nz,npad); 
  } 
  else{ 
   read("vn",vn,nnxz);
   read("vh",vh,nnxz);
   read("del",del,nnxz);
  }
 
  float vn0,vh0,del0;
  get_param("vn0",vn0,"vh0",vh0,"del0",del0);
 
  scale(vn,vn,1./vn0,nnxz);
  scale(vh,vh,1./vh0,nnxz);
  scale(del,del,1./del0,nnxz);
  
  float *gvn=new float[nnxz]();
  float *gvh=new float[nnxz]();
  float *gdel=new float[nnxz]();
  
  chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
  float val=objFuncGradientVnVhDel(gvn,gvh,gdel,data,vn,vh,del,vn0,vh0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  
  chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
  chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
  cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
  cout<<"Objfunc "<<val<<endl;

  float agv=fabs(gvn[0]),av=fabs(vn[0]);
  float age=fabs(gvh[0]),ae=fabs(vh[0]);
  float agd=fabs(gdel[0]),ad=fabs(del[0]);
  for(int i=0;i<nnxz;i++){
      if(fabs(gvn[i])>agv) agv=fabs(gvn[i]);
      if(fabs(vn[i])>av) av=fabs(vn[i]);
      if(fabs(gvh[i])>age) age=fabs(gvh[i]);
      if(fabs(vh[i])>ae) ae=fabs(vh[i]);
      if(fabs(gdel[i])>agd) agd=fabs(gdel[i]);
      if(fabs(del[i])>ad) ad=fabs(del[i]);
  }

  if(ae==0.) get_param("maxvh",ae);
  if(ad==0.) get_param("maxdel",ad);

  vn0=sqrt((agd/ad)/(agv/av));
  vh0=sqrt((agd/ad)/(age/ae));

  cout<<"vn0 should be "<<vn0<<endl;
  cout<<"vh0 should be "<<vh0<<endl;
  cout<<"del0 should be 1"<<endl;
 
  write("gvn",gvn,nnxz);
  to_header("gvn","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gvn","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gvh",gvh,nnxz);
  to_header("gvh","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gvh","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gdel",gdel,nnxz);
  to_header("gdel","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gdel","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []vn; delete []vh; delete []del;
  delete []gvn; delete []gvh; delete []gdel;
 }
 else if(parameter.compare("vvnvh")==0){
  float *v=new float[nnxz]();
  float *vn=new float[nnxz]();
  float *vh=new float[nnxz]();
  
  if(padded==0){
   init_model("v",v,nx,nz,npad); 
   init_model("vn",vn,nx,nz,npad); 
   init_model("vh",vh,nx,nz,npad); 
  } 
  else{ 
   read("v",v,nnxz);
   read("vn",vn,nnxz);
   read("vh",vh,nnxz);
  }
 
  float v0,vn0,vh0;
  get_param("v0",v0,"vn0",vn0,"vh0",vh0);
 
  scale(v,v,1./v0,nnxz);
  scale(vn,vn,1./vn0,nnxz);
  scale(vh,vh,1./vh0,nnxz);
  
  float *gv=new float[nnxz]();
  float *gvn=new float[nnxz]();
  float *gvh=new float[nnxz]();
  
  chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
  float val=objFuncGradientVVnVh(gv,gvn,gvh,data,v,vn,vh,v0,vn0,vh0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  
  chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
  chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
  cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
  cout<<"Objfunc "<<val<<endl;

  float agv=fabs(gv[0]),av=fabs(v[0]);
  float age=fabs(gvn[0]),ae=fabs(vn[0]);
  float agd=fabs(gvh[0]),ad=fabs(vh[0]);
  for(int i=0;i<nnxz;i++){
      if(fabs(gv[i])>agv) agv=fabs(gv[i]);
      if(fabs(v[i])>av) av=fabs(v[i]);
      if(fabs(gvn[i])>age) age=fabs(gvn[i]);
      if(fabs(vn[i])>ae) ae=fabs(vn[i]);
      if(fabs(gvh[i])>agd) agd=fabs(gvh[i]);
      if(fabs(vh[i])>ad) ad=fabs(vh[i]);
  }

  if(ae==0.) get_param("maxvn",ae);
  if(ad==0.) get_param("maxvh",ad);

  v0=sqrt((agd/ad)/(agv/av));
  vn0=sqrt((agd/ad)/(age/ae));

  cout<<"v0 should be "<<v0<<endl;
  cout<<"vn0 should be "<<vn0<<endl;
  cout<<"vh0 should be 1"<<endl;
 
  write("gv",gv,nnxz);
  to_header("gv","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gv","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gvn",gvn,nnxz);
  to_header("gvn","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gvn","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gvh",gvh,nnxz);
  to_header("gvh","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gvh","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []v; delete []vn; delete []vh;
  delete []gv; delete []gvn; delete []gvh;
 }
 else if(parameter.compare("cij")==0){
  float *c11=new float[nnxz]();
  float *c13=new float[nnxz]();
  float *c33=new float[nnxz]();
  
  if(padded==0){
   init_model("c11",c11,nx,nz,npad); 
   init_model("c13",c13,nx,nz,npad); 
   init_model("c33",c33,nx,nz,npad); 
  } 
  else{ 
   read("c11",c11,nnxz);
   read("c13",c13,nnxz);
   read("c33",c33,nnxz);
  }
 
  float c110,c130,c330;
  get_param("c110",c110,"c130",c130,"c330",c330);
 
  scale(c11,c11,1./c110,nnxz);
  scale(c13,c13,1./c130,nnxz);
  scale(c33,c33,1./c330,nnxz);
  
  float *gc11=new float[nnxz]();
  float *gc13=new float[nnxz]();
  float *gc33=new float[nnxz]();
  
  chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
  float val=objFuncGradientCij(gc11,gc13,gc33,data,c11,c13,c33,c110,c130,c330,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  
  chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
  chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
  cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
  cout<<"Objfunc "<<val<<endl;

  float agv=fabs(gc11[0]),av=fabs(c11[0]);
  float age=fabs(gc13[0]),ae=fabs(c13[0]);
  float agd=fabs(gc33[0]),ad=fabs(c33[0]);
  for(int i=0;i<nnxz;i++){
      if(fabs(gc11[i])>agv) agv=fabs(gc11[i]);
      if(fabs(c11[i])>av) av=fabs(c11[i]);
      if(fabs(gc13[i])>age) age=fabs(gc13[i]);
      if(fabs(c13[i])>ae) ae=fabs(c13[i]);
      if(fabs(gc33[i])>agd) agd=fabs(gc33[i]);
      if(fabs(c33[i])>ad) ad=fabs(c33[i]);
  }

  if(ae==0.) get_param("maxc13",ae);
  if(ad==0.) get_param("maxc33",ad);

  c110=sqrt((agd/ad)/(agv/av));
  c130=sqrt((agd/ad)/(age/ae));

  cout<<"c110 should be "<<c110<<endl;
  cout<<"c130 should be "<<c130<<endl;
  cout<<"c330 should be 1"<<endl;
 
  write("gc11",gc11,nnxz);
  to_header("gc11","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gc11","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gc13",gc13,nnxz);
  to_header("gc13","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gc13","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gc33",gc33,nnxz);
  to_header("gc33","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gc33","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []c11; delete []c13; delete []c33;
  delete []gc11; delete []gc13; delete []gc33;
 }

 delete []wavelet;delete []data;delete []sloc;delete []rloc;delete []taper;
 delete []m;
 delete []souloc;delete []recloc;

 myio_close();
 return 0;
}
