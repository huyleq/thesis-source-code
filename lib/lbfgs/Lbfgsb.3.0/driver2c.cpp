#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <cmath>

using namespace std;

extern "C"{
 void setulb_(int *n,int *m,float *x,float *l,float *u,int *nbd,float *f,float *g,float *factr,float *pgtol,float *wa,int *iwa,char task[],int *iprint,char csave[],int lsave[],int isave[],float dsave[],unsigned int tasklen,unsigned int csavelen);
}

int main(){
 int n=250,m=5,iprint=-1,lsave[4],isave[44];
 float factr=0.,pgtol=0.,f,dsave[29];
 char task[60],csave[60];
 int *nbd=new int[n]();
 int *iwa=new int[3*n]();
 float *x=new float[n]();
 float *l=new float[n]();
 float *u=new float[n]();
 float *g=new float[n]();
 float *wa=new float[2*m*n+5*n+11*m*m+8*m]();

 for(int i=0;i<n;i+=2){
  nbd[i]=2;
  l[i]=1.;
  u[i]=1e2;
 }
 
 for(int i=1;i<n;i+=2){
  nbd[i]=2;
  l[i]=-1e2;
  u[i]=1e2;
 }
 
 for(int i=0;i<n;++i) x[i]=3.;

 strcpy(task,"START");
 for(int i=5;i<60;++i) task[i]=' ';

 while((task[0]=='F' && task[1]=='G') || 
       (task[0]=='N' && task[1]=='E' && task[2]=='W' && task[3]=='_' && task[4]=='X') || 
       (task[0]=='S' && task[1]=='T' && task[2]=='A' && task[3]=='R' && task[4]=='T')){
  
  setulb_(&n,&m,x,l,u,nbd,&f,g,&factr,&pgtol,wa,iwa,task,&iprint,csave,lsave,isave,dsave,60,60);

  if(task[0]=='F' && task[1]=='G'){
   f=0.25*(x[0]-1.)*(x[0]-1.);
   for(int i=1;i<n;++i) f+=(x[i]-x[i-1]*x[i-1])*(x[i]-x[i-1]*x[i-1]);
   f*=4.;

   float t1=x[1]-x[0]*x[0];
   g[0]=2.*(x[0]-1.)-1.6e1*x[0]*t1;
   for(int i=1;i<n-1;++i){
    float t2=t1;
	t1=x[i+1]-x[i]*x[i];
	g[i]=8.*t2-1.6e1*x[i]*t1;
   }
   g[n-1]=8.*t1;
  }
  else{
   if(task[0]=='N' && task[1]=='E' && task[2]=='W' && task[3]=='_' && task[4]=='X'){
    fprintf(stderr,"Iterate %d nfg=%d f=%f |proj g|=%f\n",isave[29],isave[33],f,dsave[12]);
    
    if(isave[33]>=99){
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

 for(int i=0;i<n;++i){
  fprintf(stderr,"%f ",x[i]);
  if(i%5==0 || i==n-1) fprintf(stderr,"\n");
 }
 
 delete []nbd;delete []iwa;delete []x;delete []l;delete []u;delete []g;delete []wa;
 return 0;
}
