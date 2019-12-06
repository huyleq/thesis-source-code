// from https://www.sfu.ca/~ssurjano/optimization.html
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "functions.h"

float quad(float *x,float *g){
    //simple quadratic function
    //f=x1^2+x2^2-x1x2-4x1-7x2
    g[0]=2*x[0]-x[1]-4.f;
    g[1]=2*x[1]-x[0]-7.f;
//    fprintf(stderr,"gradient at %.10f %.10f is %.10f %.10f\n",x[0],x[1],g[0],g[1]);
    return x[0]*x[0]+x[1]*x[1]-x[0]*x[1]-4*x[0]-7.f*x[1];
}

float camel6(float *x,float *g){
    // 6 hump camel function
    float x02=x[0]*x[0];
    float x04=x02*x02;
    float x12=x[1]*x[1];
    float f=(4.-2.1*x02+x04/3.)*x02+x[0]*x[1]+4*(x12-1)*x12;
    g[0]=8*x[0]-8.2*x[0]*x02+2*x[0]*x04+x[1];
    g[1]=x[0]+16*x[1]*x12-8*x[1];
    return f;
}

float rosenbrock(int n,float *x,float *g){
    float f=0.;
    for(int i=0;i<n;i+=2){
        float t1=1.-x[i]; 
        float t2=1e1*(x[i+1]-x[i]*x[i]);
        g[i+1]=2e1*t2;
        g[i]=-2*(x[i]*g[i+1]+t1);
        f+=t1*t1+t2*t2;
    }
    return f;
}

float powell(int n,float *x,float *g){
    float f=0.;
    for(int i=0;i<n;i+=4){
        float t1=x[i]+10.*x[i+1];
        float t2=x[i+2]-x[i+3];
        float t3=x[i+1]-2*x[i+2];
        float t33=t3*t3*t3;
        float t4=x[i]-x[i+3];
        float t43=t4*t4*t4;
        f+=t1*t1+5.*t2*t2+t3*t33+10.*t4*t43;
        g[i]=2*t1+40*t43;
        g[i+1]=20.*t1+4*t33;
        g[i+2]=10.*t2-8*t33;
        g[i+3]=-10.*t2-40.*t43;
    }
    return f;
}

float trid(int n,float *x,float *g){
    float f=0.;
    for(int i=0;i<n;i++){
        float t=x[i]-1.;
        f+=t*t;
        g[i]=2*t;
    }
    for(int i=0;i<n-1;i++){
        f-=x[i]*x[i+1];
        g[i]-=x[i+1];
        g[i+1]-=x[i];
    }
    return f;
}

float rosenbrock1(int n,float *x,float *g){
    float f=0.;
    memset(g,0,n*sizeof(float));
    for(int i=0;i<n-1;i++){
        float t1=x[i]-1.;
        float t2=x[i]*x[i]-x[i+1];
        f+=t1*t1+100.*t2*t2;
        g[i]+=2*t1+400.*t2*x[i];
        g[i+1]-=200.*t2;
    }
    return f;
}
