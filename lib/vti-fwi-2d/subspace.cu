#include <cmath>
#include <omp.h>
#include <cstdio>
#include "lapacke.h"

#include "myio.h"
#include "mylib.h"
#include "init.h"
#include "wave.h"
#include "HessianOp.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);

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
 string hesstype=get_s("hesstype");

  float *vepsdel=new float[3*nnxz]();
  float *v=vepsdel,*eps=vepsdel+nnxz,*del=vepsdel+2*nnxz;
  
  float *pvepsdel=new float[3*nnxz]();
  float *pv=pvepsdel,*peps=pvepsdel+nnxz,*pdel=pvepsdel+2*nnxz;
  
  float *gvepsdel=new float[3*nnxz]();
  float *gv=gvepsdel,*geps=gvepsdel+nnxz,*gdel=gvepsdel+2*nnxz;
  
  init_model("v",v,nx,nz,npad); 
  init_model("eps",eps,nx,nz,npad); 
  init_model("del",del,nx,nz,npad); 

  float v0,eps0,del0;
  get_param("v0",v0,"eps0",eps0,"del0",del0);
 
  scale(v,v,1./v0,nnxz);
  scale(eps,eps,1./eps0,nnxz);
  scale(del,del,1./del0,nnxz);
  
  read("gv",gv,nnxz);
  read("geps",geps,nnxz);
  read("gdel",gdel,nnxz);
  
 float *zero=new float[nnxz]();

 double A[81];
 
  if(hesstype.compare("full")==0){
      fprintf(stderr,"full hessian\n");
  hessianVEpsDel(pv,peps,pdel,data,v,eps,del,gv,zero,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv1
  A[0]=dot_product(gv,pv,nnx*nnz); //v1tHv1
  fprintf(stderr,"Done Hv1\n");
 
  hessianVEpsDel(pv,peps,pdel,data,v,eps,del,geps,zero,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv2
  A[1]=dot_product(gv,pv,nnx*nnz); A[9]=A[1]; //v1tHv2 and v2tHv1
  A[10]=dot_product(geps,pv,nnx*nnz); //v2tHv2
  fprintf(stderr,"Done Hv2\n");
 
  hessianVEpsDel(pv,peps,pdel,data,v,eps,del,gdel,zero,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv3
  A[2]=dot_product(gv,pv,nnx*nnz); A[18]=A[2]; //v1tHv3 and v3tHv1
  A[11]=dot_product(geps,pv,nnx*nnz); A[19]=A[11]; //v2tHv3 and v3tHv2
  A[20]=dot_product(gdel,pv,nnx*nnz); //v3tHv3
  fprintf(stderr,"Done Hv3\n");
 
  hessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,gv,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv4
  A[3]=dot_product(gv,pv,nnx*nnz); A[27]=A[3]; //v1tHv4 and v4tHv1
  A[12]=dot_product(geps,pv,nnx*nnz); A[28]=A[12]; //v2tHv4 and v4tHv2
  A[21]=dot_product(gdel,pv,nnx*nnz); A[29]=A[21]; //v3tHv4 and v4tHv3
  A[30]=dot_product(gv,peps,nnx*nnz); //v4tHv4 
  fprintf(stderr,"Done Hv4\n");
  
  hessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,geps,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv5
  A[4]=dot_product(gv,pv,nnx*nnz); A[36]=A[4]; //v1tHv5 and v5tHv1
  A[13]=dot_product(geps,pv,nnx*nnz); A[37]=A[13]; //v2tHv5 and v5tHv2
  A[22]=dot_product(gdel,pv,nnx*nnz); A[38]=A[22]; //v3tHv5 and v5tHv3
  A[31]=dot_product(gv,peps,nnx*nnz); A[39]=A[31]; //v4tHv5 and v5tHv4
  A[40]=dot_product(geps,peps,nnx*nnz); //v5tHv5 
  fprintf(stderr,"Done Hv5\n");
 
  hessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,gdel,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv6
  A[5]=dot_product(gv,pv,nnx*nnz); A[45]=A[5]; //v1tHv6 and v6tHv1
  A[14]=dot_product(geps,pv,nnx*nnz); A[46]=A[14]; //v2tHv6 and v6tHv2
  A[23]=dot_product(gdel,pv,nnx*nnz); A[47]=A[23]; //v3tHv6 and v6tHv3
  A[32]=dot_product(gv,peps,nnx*nnz); A[48]=A[32]; //v4tHv6 and v6tHv4
  A[41]=dot_product(geps,peps,nnx*nnz); A[49]=A[41]; //v5tHv6 and v6tHv5
  A[50]=dot_product(gdel,peps,nnx*nnz); //v6tHv6 
  fprintf(stderr,"Done Hv6\n");
  
  hessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,zero,gv,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv7
  A[6]=dot_product(gv,pv,nnx*nnz); A[54]=A[6]; //v1tHv7 and v7tHv1
  A[15]=dot_product(geps,pv,nnx*nnz); A[55]=A[15]; //v2tHv7 and v7tHv2
  A[24]=dot_product(gdel,pv,nnx*nnz); A[56]=A[24]; //v3tHv7 and v7tHv3
  A[33]=dot_product(gv,peps,nnx*nnz); A[57]=A[33]; //v4tHv7 and v7tHv4
  A[42]=dot_product(geps,peps,nnx*nnz); A[58]=A[42]; //v5tHv7 and v7tHv5
  A[51]=dot_product(gdel,peps,nnx*nnz); A[59]=A[51]; //v6tHv7 and v7tHv6
  A[60]=dot_product(gv,pdel,nnx*nnz); //v7tHv7 
  fprintf(stderr,"Done Hv7\n");
  
  hessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,zero,geps,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv8
  A[7]=dot_product(gv,pv,nnx*nnz); A[63]=A[7]; //v1tHv8 and v8tHv1
  A[16]=dot_product(geps,pv,nnx*nnz); A[64]=A[16]; //v2tHv8 and v8tHv2
  A[25]=dot_product(gdel,pv,nnx*nnz); A[65]=A[25]; //v3tHv8 and v8tHv3
  A[34]=dot_product(gv,peps,nnx*nnz); A[66]=A[34]; //v4tHv8 and v8tHv4
  A[43]=dot_product(geps,peps,nnx*nnz); A[67]=A[43]; //v5tHv8 and v8tHv5
  A[52]=dot_product(gdel,peps,nnx*nnz); A[68]=A[52]; //v6tHv8 and v8tHv6
  A[61]=dot_product(gv,pdel,nnx*nnz); A[69]=A[61]; //v7tHv8 and v8tHv7
  A[70]=dot_product(geps,pdel,nnx*nnz); //v8tHv8
  fprintf(stderr,"Done Hv8\n");
  
  hessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,zero,gdel,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv9
  A[8]=dot_product(gv,pv,nnx*nnz); A[72]=A[8]; //v1tHv9 and v9tHv1
  A[17]=dot_product(geps,pv,nnx*nnz); A[73]=A[17]; //v2tHv9 and v9tHv2
  A[26]=dot_product(gdel,pv,nnx*nnz); A[74]=A[26]; //v3tHv9 and v9tHv3
  A[35]=dot_product(gv,peps,nnx*nnz); A[75]=A[35]; //v4tHv9 and v9tHv4
  A[44]=dot_product(geps,peps,nnx*nnz); A[76]=A[44]; //v5tHv9 and v9tHv5
  A[53]=dot_product(gdel,peps,nnx*nnz); A[77]=A[53]; //v6tHv9 and v9tHv6
  A[62]=dot_product(gv,pdel,nnx*nnz); A[78]=A[62]; //v7tHv9 and v9tHv7
  A[71]=dot_product(geps,pdel,nnx*nnz); A[79]=A[71]; //v8tHv9 and v9tHv8
  A[80]=dot_product(gdel,pdel,nnx*nnz); //v9tHv9
  fprintf(stderr,"Done Hv9\n");
 }
 else{
      fprintf(stderr,"GN hessian\n");
  GNhessianVEpsDel(pv,peps,pdel,data,v,eps,del,gv,zero,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv1
  A[0]=dot_product(gv,pv,nnx*nnz); //v1tHv1
  fprintf(stderr,"Done Hv1\n");
 
  GNhessianVEpsDel(pv,peps,pdel,data,v,eps,del,geps,zero,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv2
  A[1]=dot_product(gv,pv,nnx*nnz); A[9]=A[1]; //v1tHv2 and v2tHv1
  A[10]=dot_product(geps,pv,nnx*nnz); //v2tHv2
  fprintf(stderr,"Done Hv2\n");
 
  GNhessianVEpsDel(pv,peps,pdel,data,v,eps,del,gdel,zero,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv3
  A[2]=dot_product(gv,pv,nnx*nnz); A[18]=A[2]; //v1tHv3 and v3tHv1
  A[11]=dot_product(geps,pv,nnx*nnz); A[19]=A[11]; //v2tHv3 and v3tHv2
  A[20]=dot_product(gdel,pv,nnx*nnz); //v3tHv3
  fprintf(stderr,"Done Hv3\n");
 
  GNhessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,gv,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv4
  A[3]=dot_product(gv,pv,nnx*nnz); A[27]=A[3]; //v1tHv4 and v4tHv1
  A[12]=dot_product(geps,pv,nnx*nnz); A[28]=A[12]; //v2tHv4 and v4tHv2
  A[21]=dot_product(gdel,pv,nnx*nnz); A[29]=A[21]; //v3tHv4 and v4tHv3
  A[30]=dot_product(gv,peps,nnx*nnz); //v4tHv4 
  fprintf(stderr,"Done Hv4\n");
  
  GNhessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,geps,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv5
  A[4]=dot_product(gv,pv,nnx*nnz); A[36]=A[4]; //v1tHv5 and v5tHv1
  A[13]=dot_product(geps,pv,nnx*nnz); A[37]=A[13]; //v2tHv5 and v5tHv2
  A[22]=dot_product(gdel,pv,nnx*nnz); A[38]=A[22]; //v3tHv5 and v5tHv3
  A[31]=dot_product(gv,peps,nnx*nnz); A[39]=A[31]; //v4tHv5 and v5tHv4
  A[40]=dot_product(geps,peps,nnx*nnz); //v5tHv5 
  fprintf(stderr,"Done Hv5\n");
 
  GNhessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,gdel,zero,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv6
  A[5]=dot_product(gv,pv,nnx*nnz); A[45]=A[5]; //v1tHv6 and v6tHv1
  A[14]=dot_product(geps,pv,nnx*nnz); A[46]=A[14]; //v2tHv6 and v6tHv2
  A[23]=dot_product(gdel,pv,nnx*nnz); A[47]=A[23]; //v3tHv6 and v6tHv3
  A[32]=dot_product(gv,peps,nnx*nnz); A[48]=A[32]; //v4tHv6 and v6tHv4
  A[41]=dot_product(geps,peps,nnx*nnz); A[49]=A[41]; //v5tHv6 and v6tHv5
  A[50]=dot_product(gdel,peps,nnx*nnz); //v6tHv6 
  fprintf(stderr,"Done Hv6\n");
  
  GNhessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,zero,gv,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv7
  A[6]=dot_product(gv,pv,nnx*nnz); A[54]=A[6]; //v1tHv7 and v7tHv1
  A[15]=dot_product(geps,pv,nnx*nnz); A[55]=A[15]; //v2tHv7 and v7tHv2
  A[24]=dot_product(gdel,pv,nnx*nnz); A[56]=A[24]; //v3tHv7 and v7tHv3
  A[33]=dot_product(gv,peps,nnx*nnz); A[57]=A[33]; //v4tHv7 and v7tHv4
  A[42]=dot_product(geps,peps,nnx*nnz); A[58]=A[42]; //v5tHv7 and v7tHv5
  A[51]=dot_product(gdel,peps,nnx*nnz); A[59]=A[51]; //v6tHv7 and v7tHv6
  A[60]=dot_product(gv,pdel,nnx*nnz); //v7tHv7 
  fprintf(stderr,"Done Hv7\n");
  
  GNhessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,zero,geps,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv8
  A[7]=dot_product(gv,pv,nnx*nnz); A[63]=A[7]; //v1tHv8 and v8tHv1
  A[16]=dot_product(geps,pv,nnx*nnz); A[64]=A[16]; //v2tHv8 and v8tHv2
  A[25]=dot_product(gdel,pv,nnx*nnz); A[65]=A[25]; //v3tHv8 and v8tHv3
  A[34]=dot_product(gv,peps,nnx*nnz); A[66]=A[34]; //v4tHv8 and v8tHv4
  A[43]=dot_product(geps,peps,nnx*nnz); A[67]=A[43]; //v5tHv8 and v8tHv5
  A[52]=dot_product(gdel,peps,nnx*nnz); A[68]=A[52]; //v6tHv8 and v8tHv6
  A[61]=dot_product(gv,pdel,nnx*nnz); A[69]=A[61]; //v7tHv8 and v8tHv7
  A[70]=dot_product(geps,pdel,nnx*nnz); //v8tHv8
  fprintf(stderr,"Done Hv8\n");
  
  GNhessianVEpsDel(pv,peps,pdel,data,v,eps,del,zero,zero,gdel,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m); //Hv9
  A[8]=dot_product(gv,pv,nnx*nnz); A[72]=A[8]; //v1tHv9 and v9tHv1
  A[17]=dot_product(geps,pv,nnx*nnz); A[73]=A[17]; //v2tHv9 and v9tHv2
  A[26]=dot_product(gdel,pv,nnx*nnz); A[74]=A[26]; //v3tHv9 and v9tHv3
  A[35]=dot_product(gv,peps,nnx*nnz); A[75]=A[35]; //v4tHv9 and v9tHv4
  A[44]=dot_product(geps,peps,nnx*nnz); A[76]=A[44]; //v5tHv9 and v9tHv5
  A[53]=dot_product(gdel,peps,nnx*nnz); A[77]=A[53]; //v6tHv9 and v9tHv6
  A[62]=dot_product(gv,pdel,nnx*nnz); A[78]=A[62]; //v7tHv9 and v9tHv7
  A[71]=dot_product(geps,pdel,nnx*nnz); A[79]=A[71]; //v8tHv9 and v9tHv8
  A[80]=dot_product(gdel,pdel,nnx*nnz); //v9tHv9
 }

 double b[9];
 b[0]=-dot_product(gv,gv,nnx*nnz); //v1tg
 b[1]=-dot_product(geps,gv,nnx*nnz); //v2tg
 b[2]=-dot_product(gdel,gv,nnx*nnz); //v3tg
 b[3]=-dot_product(gv,geps,nnx*nnz); //v4tg
 b[4]=-dot_product(geps,geps,nnx*nnz); //v5tg
 b[5]=-dot_product(gdel,geps,nnx*nnz); //v6tg
 b[6]=-dot_product(gv,gdel,nnx*nnz); //v7tg
 b[7]=-dot_product(geps,gdel,nnx*nnz); //v8tg
 b[8]=-dot_product(gdel,gdel,nnx*nnz); //v9tg

 int N=9,NRHS=1,LDA=9,IPIV[9],LDB=9,INFO;

 dgesv_(&N,&NRHS,A,&LDA,IPIV,b,&LDB,&INFO);

 if(INFO!=0)
  fprintf(stderr,"Something is wrong when inverting matrix VtHV\n"); 
 else{
  for(int i=0;i<9;++i) fprintf(stderr,"b[%d]=%f\n",i,b[i]);

  #pragma omp parallel for num_threads(16) 
  for(int i=0;i<nnx*nnz;++i){
   pv[i]=b[0]*gv[i]+b[1]*geps[i]+b[2]*gdel[i];
   peps[i]=b[3]*gv[i]+b[4]*geps[i]+b[5]*gdel[i];
   pdel[i]=b[6]*gv[i]+b[7]*geps[i]+b[8]*gdel[i];
  }
  write("pv",pv,nnxz);
  to_header("pv","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("pv","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("peps",peps,nnxz);
  to_header("peps","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("peps","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("pdel",pdel,nnxz);
  to_header("pdel","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("pdel","n2",nnz,"o2",-dz*npad,"d2",dz);
 }

  delete []vepsdel;
  delete []pvepsdel;
  delete []gvepsdel;

 delete []wavelet;
 delete []data;
 delete []sloc;
 delete []rloc;
 delete []taper;
 delete []souloc;delete []recloc;

 myio_close();
 return 0;
}
