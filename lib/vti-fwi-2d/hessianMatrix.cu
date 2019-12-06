#include <iostream>
#include <cmath>
#include <omp.h>
#include <cstdio>
#include <cstring>

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
int nnxz=nnx*nnz,nn=3*nnxz;
int nxz=nx*nz,n=3*nxz;

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

 if(parameter.compare("vepsdel")==0){
  fprintf(stderr,"parameter v eps del\n");
  float *vepsdel=new float[nn]();
  float *v=vepsdel,*eps=vepsdel+nnxz,*del=vepsdel+2*nnxz;
  float *gvepsdel=new float[nn]();
  float *gv=gvepsdel,*geps=gvepsdel+nnxz,*gdel=gvepsdel+2*nnxz;
  
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

  float *dvepsdel=new float[nn];
  float *hessmat=new float[n*n]();
  
  if(hesstype.compare("full")==0){
   HessianOpVEpsDel H(data,v,eps,del,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
   for(int k=0;k<3;k++){
       for(int iz=0;iz<nz;iz++){
           int j=iz+npad;
           for(int ix=0;ix<nx;ix++){
               int i=ix+npad+j*nnx+k*nnxz;
               fprintf(stderr,"i=%d\n",i);
                memset(dvepsdel,0,nn*sizeof(float));
                dvepsdel[i]=1.f; 
                H.forward(false,dvepsdel,gvepsdel);
                i=ix+iz*nx+k*nxz;
                #pragma omp parallel for num_threads(16)
                for(int l=0;l<nz;l++){
                    memcpy(hessmat+i*n+l*nx,gv+npad+(l+npad)*nnx,nx*sizeof(float));
                    memcpy(hessmat+i*n+nxz+l*nx,geps+npad+(l+npad)*nnx,nx*sizeof(float));
                    memcpy(hessmat+i*n+2*nxz+l*nx,gdel+npad+(l+npad)*nnx,nx*sizeof(float));
                }
           }
       }
   }
  }
  else if(hesstype.compare("GN")==0){
   GNHessianOpVEpsDel H(data,v,eps,del,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
   for(int k=0;k<3;k++){
       for(int iz=0;iz<nz;iz++){
           int j=iz+npad;
           for(int ix=0;ix<nx;ix++){
               int i=ix+npad+j*nnx+k*nnxz;
               fprintf(stderr,"i=%d\n",i);
                memset(dvepsdel,0,nn*sizeof(float));
                dvepsdel[i]=1.f; 
                H.forward(false,dvepsdel,gvepsdel);
                i=ix+iz*nx+k*nxz;
                #pragma omp parallel for num_threads(16)
                for(int l=0;l<nz;l++){
                    memcpy(hessmat+i*n+l*nx,gv+npad+(l+npad)*nnx,nx*sizeof(float));
                    memcpy(hessmat+i*n+nxz+l*nx,geps+npad+(l+npad)*nnx,nx*sizeof(float));
                    memcpy(hessmat+i*n+2*nxz+l*nx,gdel+npad+(l+npad)*nnx,nx*sizeof(float));
                }
           }
       }
   }
  }
  else fprintf(stderr,"please specify hessian type by hesstype=full or hesstype=GN\n");
 
  write("hessmat",hessmat,n*n);
  to_header("hessmat","n1",n,"o1",0,"d1",1);
  to_header("hessmat","n2",n,"o2",0,"d2",1);
  
  delete []vepsdel;
  delete []dvepsdel;
  delete []gvepsdel;
  delete []hessmat;
 }
 else if(parameter.compare("vhepsdel")==0){
  fprintf(stderr,"parameter vh eps del\n");
  float *vhepsdel=new float[nn]();
  float *vh=vhepsdel,*eps=vhepsdel+nnxz,*del=vhepsdel+2*nnxz;
  float *gvhepsdel=new float[nn]();
  float *gvh=gvhepsdel,*geps=gvhepsdel+nnxz,*gdel=gvhepsdel+2*nnxz;
  
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

  float *dvhepsdel=new float[nn]();
  float *hessmat=new float[n*n]();
  
//  if(hesstype.compare("full")==0) ;
//  else if(hesstype.compare("GN")==0) GNhessianVEpsDel(gv,geps,gdel,data,v,eps,del,dv,deps,ddel,v0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  if(hesstype.compare("full")==0){
   HessianOpVhEpsDel H(data,vh,eps,del,vh0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
   for(int k=0;k<3;k++){
       for(int iz=0;iz<nz;iz++){
           int j=iz+npad;
           for(int ix=0;ix<nx;ix++){
               int i=ix+npad+j*nnx+k*nnxz;
               fprintf(stderr,"i=%d\n",i);
                memset(dvhepsdel,0,nn*sizeof(float));
                dvhepsdel[i]=1.f; 
                H.forward(false,dvhepsdel,gvhepsdel);
                i=ix+iz*nx+k*nxz;
                #pragma omp parallel for num_threads(16)
                for(int l=0;l<nz;l++){
                    memcpy(hessmat+i*n+l*nx,gvh+npad+(l+npad)*nnx,nx*sizeof(float));
                    memcpy(hessmat+i*n+nxz+l*nx,geps+npad+(l+npad)*nnx,nx*sizeof(float));
                    memcpy(hessmat+i*n+2*nxz+l*nx,gdel+npad+(l+npad)*nnx,nx*sizeof(float));
                }
           }
       }
   }
  }
  else if(hesstype.compare("GN")==0){
   GNHessianOpVhEpsDel H(data,vh,eps,del,vh0,eps0,del0,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
   for(int k=0;k<3;k++){
       for(int iz=0;iz<nz;iz++){
           int j=iz+npad;
           for(int ix=0;ix<nx;ix++){
               int i=ix+npad+j*nnx+k*nnxz;
               fprintf(stderr,"i=%d\n",i);
                memset(dvhepsdel,0,nn*sizeof(float));
                dvhepsdel[i]=1.f; 
                H.forward(false,dvhepsdel,gvhepsdel);
                i=ix+iz*nx+k*nxz;
                #pragma omp parallel for num_threads(16)
                for(int l=0;l<nz;l++){
                    memcpy(hessmat+i*n+l*nx,gvh+npad+(l+npad)*nnx,nx*sizeof(float));
                    memcpy(hessmat+i*n+nxz+l*nx,geps+npad+(l+npad)*nnx,nx*sizeof(float));
                    memcpy(hessmat+i*n+2*nxz+l*nx,gdel+npad+(l+npad)*nnx,nx*sizeof(float));
                }
           }
       }
   }
  }
  else fprintf(stderr,"please specify hessian type by hesstype=full or hesstype=GN\n");
 
  write("hessmat",hessmat,n*n);
  to_header("hessmat","n1",n,"o1",0,"d1",1);
  to_header("hessmat","n2",n,"o2",0,"d2",1);
  
  delete []vhepsdel;
  delete []dvhepsdel;
  delete []gvhepsdel;
  delete []hessmat;
 }
 else if(parameter.compare("cij")==0){
  fprintf(stderr,"parameter c11 c13 c33\n");
  float *c11c13c33=new float[nn]();
  float *c11=c11c13c33,*c13=c11c13c33+nnxz,*c33=c11c13c33+2*nnxz;
  float *dc11c13c33=new float[nn]();
  float *dc11=dc11c13c33,*dc13=dc11c13c33+nnxz,*dc33=dc11c13c33+2*nnxz;
 
  if(padded==0){
   init_model("c11",c11,nx,nz,npad); 
   init_model("c13",c13,nx,nz,npad); 
   init_model("c33",c33,nx,nz,npad); 
   init_model("dc11",dc11,nx,nz,npad); 
   init_model("dc13",dc13,nx,nz,npad); 
   init_model("dc33",dc33,nx,nz,npad); 
  } 
  else{ 
   read("c11",c11,nnxz);
   read("c13",c13,nnxz);
   read("c33",c33,nnxz);
   read("dc11",dc11,nnxz);
   read("dc13",dc13,nnxz);
   read("dc33",dc33,nnxz);
  }
  
  float c110,c130,c330;
  get_param("c110",c110,"c130",c130,"c330",c330);
 
  scale(c11,c11,1./c110,nnxz);
  scale(c13,c13,1./c130,nnxz);
  scale(c33,c33,1./c330,nnxz);

  scale(dc11,dc11,1./c110,nnxz);
  scale(dc13,dc13,1./c130,nnxz);
  scale(dc33,dc33,1./c330,nnxz);
 
  float *gc11c13c33=new float[nn]();
  float *gc11=gc11c13c33,*gc13=gc11c13c33+nnxz,*gc33=gc11c13c33+2*nnxz;
 
  if(hesstype.compare("full")==0) hessianCij(gc11,gc13,gc33,data,c11,c13,c33,dc11,dc13,dc33,c110,c130,c330,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  else if(hesstype.compare("GN")==0) GNhessianCij(gc11,gc13,gc33,data,c11,c13,c33,dc11,dc13,dc33,c110,c130,c330,wavelet,sloc,ns,rloc,nr,taper,nx,nz,nt,npad,dx,dz,dt,rate,ot,wbottom,m);
  else fprintf(stderr,"please specify hessian type by hesstype=full or hesstype=GN\n");
 
  write("gc11",gc11,nnxz);
  to_header("gc11","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gc11","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gc13",gc13,nnxz);
  to_header("gc13","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gc13","n2",nnz,"o2",-dz*npad,"d2",dz);
  write("gc33",gc33,nnxz);
  to_header("gc33","n1",nnx,"o1",-dx*npad,"d1",dx);
  to_header("gc33","n2",nnz,"o2",-dz*npad,"d2",dz);
 
  delete []c11c13c33;
  delete []dc11c13c33;
  delete []gc11c13c33;
 }
 else{
     fprintf(stderr,"please specify parameterization by parameter=something in commandline where something is one of vepsdel, vhepsdel, vnetadel,vhepseta, vvhdel, vnvhdel, cij, or vvnvh\n");
 }
 delete []wavelet;delete []data;delete []sloc;delete []rloc;delete []taper;
 delete []m;
 delete []souloc;delete []recloc;

 myio_close();
 return 0;
}
