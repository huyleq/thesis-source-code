#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <iostream>

#include "mylib.h"
#include "agc.h"

#define EPS 1e-6f

using namespace std;

void runsum(int nt,int halfwidth,float *w,float *s){
    #pragma omp parallel for num_threads(16)
    for(int i=0;i<nt;i++){
        s[i]=0.f;
//        for(int j=max(0,i-halfwidth);j<=min(nt-1,i+halfwidth);j++) s[i]+=w[j];
        for(int j=i-halfwidth;j<=i+halfwidth;j++){
            if(j<0) s[i]+=w[j+nt];
            else if(j>=nt) s[i]+=w[j-nt];
            else s[i]+=w[j];
        }
    }
    return;
}

void agc(int nt,int halfwidth,const float *d,float *hd,float *sd){
    float *d2=new float[nt];
    multiply(d2,d,d,nt);
    runsum(nt,halfwidth,d2,sd);
    #pragma omp parallel for num_threads(16)
    for(int i=0;i<nt;i++) hd[i]=d[i]/sqrt(max(EPS,sd[i])); 
    delete []d2;
    return;
}

double residualAGC(int nt,int ntr,int halfwidth,const float *d,const float *d0,float *res,float *adjsou){
    float *hd=new float[nt];
    float *sd=new float[nt];
    float *hd0=new float[nt];
    float *sd0=new float[nt];

    double fi=0.;
    for(int j=0;j<ntr;j++){
        agc(nt,halfwidth,d+j*nt,hd,sd);
        agc(nt,halfwidth,d0+j*nt,hd0,sd0);
        
        subtract(res+j*nt,hd,hd0,nt);
        fi+=dot_product(res+j*nt,res+j*nt,nt);
        
        float *rs=sd,*drss=sd0;
        #pragma omp parallel for num_threads(16)
        for(int i=0;i<nt;i++){
            float temp1=max(EPS,sd[i]);
            float temp2=res[i+j*nt]/sqrt(temp1);
            drss[i]=d[i+j*nt]*temp2/temp1;
            rs[i]=temp2;
        }
    
        runsum(nt,halfwidth,drss,adjsou+j*nt);

        #pragma omp parallel for num_threads(16)
        for(int i=0;i<nt;i++) adjsou[i+j*nt]=rs[i]-d[i+j*nt]*adjsou[i+j*nt];
    }

    delete []hd;delete []sd;delete []hd0;delete []sd0;
    return fi;
}

double residualAGC(int nt,int ntr,int halfwidth,float *d,const float *d0){
    float *hd=new float[nt];
    float *sd=new float[nt];
    float *hd0=new float[nt];
    float *sd0=new float[nt];

    double fi=0.;
    for(int j=0;j<ntr;j++){
        agc(nt,halfwidth,d+j*nt,hd,sd);
        agc(nt,halfwidth,d0+j*nt,hd0,sd0);
        
        subtract(hd,hd,hd0,nt);
        fi+=dot_product(hd,hd,nt);
        
        float *rs=sd,*drss=sd0;
        #pragma omp parallel for num_threads(16)
        for(int i=0;i<nt;i++){
            float temp1=max(EPS,sd[i]);
            float temp2=hd[i]/sqrt(temp1);
            drss[i]=d[i+j*nt]*temp2/temp1;
            rs[i]=temp2;
        }
    
        runsum(nt,halfwidth,drss,hd);

        #pragma omp parallel for num_threads(16)
        for(int i=0;i<nt;i++) d[i+j*nt]=rs[i]-d[i+j*nt]*hd[i];
    }

    delete []hd;delete []sd;delete []hd0;delete []sd0;
    return fi;
}

void tpower(float *data,int nt,float ot,float dt,int nr,float p){
    #pragma omp parallel for
    for(int ir=0;ir<nr;ir++){
        #pragma omp simd
        for(int it=0;it<nt;it++){
            float t=ot+it*dt;
            t=pow(t,p);
            data[it+ir*nt]*=t;
        }
    }
    return;
}
