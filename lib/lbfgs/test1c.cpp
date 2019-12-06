#include <cstdio>
#include <cstdlib>
#include <string>

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
    string solver=string(argv[2]);

    //n is number of unknowns
    //m is number of remembered gradients used to build approximated Hessian
    int n;
    if(func==0 || func==1) n=2;
    else if(func==4) n=50;
    else if(func==5) n=25;
    else n=100;

    int m=5,diagco=0,icall=0,iflag=0;
    float f;
    float *x=new float[n]();
    if(func==0 || func==1){ 
        x[0]=-1.f;
        x[1]=-0.5f;
    }
    else if(func==2){
        for(int i=0;i<n;i++) x[i]=2.;
    }
    else if(func==3){
        for(int i=0;i<n;i+=4){
            x[i]=3.f;
            x[i+1]=-1.f;
            x[i+3]=1.f;
        }
    }

    float *g=new float[n]();
    float *diag=new float[n]();
    float *w;
    if(solver.compare("STEEPEST")==0) w=new float[n]();
    else if(solver.compare("NLCG")==0) w=new float[2*n+1]();
    else w=new float[n*(2*m+1)+2*m]();
    int *isave=new int[nisave]();
    float *dsave=new float[ndsave]();
    
    while(true){
        //user supplied objective function and gradient evaluation function
        if(func==0) f=quad(x,g);
        else if(func==1) f=camel6(x,g);
        else if(func==2) f=rosenbrock(n,x,g);
        else if(func==3) f=powell(n,x,g);
        else if(func==4) f=trid(n,x,g);
        else if(func==5) f=rosenbrock1(n,x,g);
        
        //call solver
        if(solver.compare("STEEPEST")==0) steepest(n,x,f,g,diag,w,iflag,isave,dsave);
        else if(solver.compare("NLCG")==0) nlcg(n,x,f,g,diag,w,iflag,isave,dsave);
        else lbfgs(n,m,x,f,g,diagco,diag,w,iflag,isave,dsave);
        
        icall++;
        if(iflag<=0 || icall>200) break;
    }
    
    for(int i=0;i<n;i++) fprintf(stderr,"x[%d]=%.10f\n",i,x[i]);
    
    delete []x;delete []g;delete []diag;delete []w;
    delete []isave;delete []dsave;
    return 0;
}
