#include <cstdio>
#include <cmath>
#include <cstring>
#include "check.h"

void checkCij(float *c11,float *c13,float *c33,float c110,float c130,float c330,size_t n,float *m){
 int count=0;
 memset(m,0,n*sizeof(float));
 #pragma omp parallel for reduction(+:count) num_threads(16)
 for(size_t i=0;i<n;++i){
  float temp=sqrt(c110*c11[i]*c330*c33[i])/c130;
  if(c13[i]>temp){
   c13[i]=0.99*temp;
   count+=1;
   m[i]=1;
  }
 }
 fprintf(stderr,"Modified models at %d points\n",count);
 return;
}

void checkEpsDel(float *eps,float *del,float eps0,float del0,size_t n,float *m){
 int count=0;
 memset(m,0,n*sizeof(float));
 #pragma omp parallel for reduction(+:count) num_threads(16)
 for(size_t i=0;i<n;++i){
  if(del0*del[i]==eps0*eps[i]){
   m[i]=1;
  }
  else if(del0*del[i]>eps0*eps[i]){
   float temp=(eps0*del[i]+del0*eps[i])/(eps0*eps0+del0*del0);
   eps[i]=temp*del0;
   del[i]=temp*eps0;
   m[i]=2;
   count+=1;
  }
 }
 fprintf(stderr,"Modified models at %d points\n",count);
 return;
}

void checkEta(float *eta,size_t n,float *m){
 int count=0;
 memset(m,0,n*sizeof(float));
 #pragma omp parallel for reduction(+:count) num_threads(16)
 for(size_t i=0;i<n;++i){
  if(eta[i]<0.){
   eta[i]=0.;
   count+=1;
   m[i]=1;
  }
 }
 fprintf(stderr,"Modified models at %d points\n",count);
 return;
}

void projection(float &c11,float &c13,float &c33){
 float D=(c11-c33)*(c11-c33)+4.*c13*c13;
 float lambda1=0.5*(c11+c33+sqrt(D));
 float lambda2=0.5*(c11+c33-sqrt(D));
 float u1=1.;
 float u2=(lambda1-c11)/c13;
 float uu=sqrt(u1*u1+u2*u2);
 u1=u1/uu;
 u2=u2/uu;
 float v1=1.;
 float v2=(lambda2-c11)/c13;
 float vv=sqrt(v1*v1+v2*v2);
 v1=v1/vv;
 v2=v2/vv;
 if(lambda1<0.) lambda1=0.;
 if(lambda2<0.) lambda2=0.;
 c11=lambda1*u1*u1+lambda2*v1*v1;
 c13=lambda1*u1*u2+lambda2*v1*v2;
 c33=lambda1*u2*u2+lambda2*v2*v2;
 return;
}

void checkVVhDel(float *v,float *vh,float *del,float v0,float vh0,float del0,size_t n,float *m){
 int count=0;
 memset(m,0,n*sizeof(float));
 #pragma omp parallel for reduction(+:count) num_threads(16)
 for(size_t i=0;i<n;++i){
  float v2=v0*v[i]*v0*v[i];
  float vh2=vh0*vh[i]*vh0*vh[i];
  float tdel=1.+2*del0*del[i];
  if(vh2<v2*tdel){
   tdel=vh2/v2;
   del[i]=(tdel-1.)*0.5/del0;
   count+=1;
   m[i]=1;
  }
 }
 fprintf(stderr,"Modified models at %d points\n",count);
 return;
}

void checkVnVh(float *vn,float *vh,float vn0,float vh0,size_t n,float *m){
 int count=0;
 memset(m,0,n*sizeof(float));
 #pragma omp parallel for reduction(+:count) num_threads(16)
 for(size_t i=0;i<n;++i){
  float temp=vh0*vh[i]/vn0;
  if(temp<vn[i]){
   vn[i]=temp;
   count+=1;
   m[i]=1;
  }
 }
 fprintf(stderr,"Modified models at %d points\n",count);
 return;
}
