#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "lbfgs.h"
#include "functions.h"

using namespace std;

int main(int argc,char **argv){
    /* func is type of test function
     * 1 for camel6
     * 2 for rosenbrock
     * 3 for powell
     * 4 for trid 
     * 5 for a variant of rosenbrock
     */
    int func=stoi(string(argv[1]));

    //N is number of unknowns
    //M is number of remembered gradients used to build approximated Hessian
    int N;
    if(func==0 || func==1) N=2;
    else if(func==4) N=50;
    else if(func==5) N=25;
    else N=100;

    int M=5,IPRINT[2]={1,0},DIAGCO=0,ICALL=0,IFLAG=0;
    float EPS=1e-5,XTOL=1e-16,F;
    float *X=new float[N]();
    if(func==0 || func==1){
        X[0]=-1.f;
        X[1]=-0.5f;   
    }
    else if(func==2){
        for(int i=0;i<N;i++) X[i]=2.;
    }
    else if(func==3){
        for(int i=0;i<N;i+=4){
            X[i]=3.f;
            X[i+1]=-1.f;
            X[i+3]=1.f;
        }
    }

    float *G=new float[N]();
    float *DIAG=new float[N]();
    float *W=new float[N*(2*M+1)+2*M]();
    
    while(true){
        //user supplied objective function and gradient evaluation function
        if(func==0) F=quad(X,G);
        else if(func==1) F=camel6(X,G);
        else if(func==2) F=rosenbrock(N,X,G);
        else if(func==3) F=powell(N,X,G);
        else if(func==4) F=trid(N,X,G);
        else if(func==5) F=rosenbrock1(N,X,G);
         
        //call solver
        lbfgs_(&N,&M,X,&F,G,&DIAGCO,DIAG,IPRINT,&EPS,&XTOL,W,&IFLAG);
        
        ICALL++;
        if(IFLAG<=0 || ICALL>200) break;
    }
    
    for(int i=0;i<N;i++) fprintf(stderr,"x[%d]=%.10f\n",i,X[i]);
    
    delete []X;delete []G;delete []DIAG;delete []W;
    return 0;
}
