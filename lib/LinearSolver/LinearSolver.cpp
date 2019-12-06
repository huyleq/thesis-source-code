#include "LinearSolver.h"
#include "mylib.h"
#include "myio.h"
#include <cfloat>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace std;

void MatMul::forward(bool add,const float *model,float *data){
    if(!add) memset(data,0,_dataSize*sizeof(float));
    for(int i=0;i<_dataSize;i++){
        for(int j=0;j<_modelSize;j++){
            data[i]+=_mat[j+i*_modelSize]*model[j];
        }
    }
    return;
}

void MatMul::adjoint(bool add,float *model,const float *data){
    if(!add) memset(model,0,_modelSize*sizeof(float));
    for(int i=0;i<_dataSize;i++){
        for(int j=0;j<_modelSize;j++){
            model[j]+=_mat[j+i*_modelSize]*data[i];
        }
    }
    return;
}


void Lap2d::forward(bool add,const float *model,float *data){
    if(!add) memset(data,0,_dataSize*sizeof(float));
    for(int iy=1;iy<_ny-1;iy++){
        for(int ix=1;ix<_nx-1;ix++){
            data[ix+iy*_nx]+=model[ix+1+iy*_nx]+model[ix-1+iy*_nx]+model[ix+(iy+1)*_nx]+model[ix+(iy-1)*_nx]-4.*model[ix+iy*_nx];
        }
    }
    return;
}

void Lap2d::adjoint(bool add,float *model,const float *data){
    if(!add) memset(model,0,_modelSize*sizeof(float));
    for(int iy=1;iy<_ny-1;iy++){
        for(int ix=1;ix<_nx-1;ix++){
            model[ix+iy*_nx]+=data[ix+1+iy*_nx]+data[ix-1+iy*_nx]+data[ix+(iy+1)*_nx]+data[ix+(iy-1)*_nx]-4.*data[ix+iy*_nx];
        }
    }
    return;
}

void simpleSolver(Operator *Fop,float *model,const float *data,int niter){
    //solve Fm-d=0
    long long modelSize=Fop->_modelSize;
    long long dataSize=Fop->_dataSize;

//    write("model",model,modelSize);
//    to_header("model","n1",modelSize,"o1",0,"d1",1.);
//    to_header("model","n2",1,"o2",0,"d2",1.);

    float *s=new float[modelSize]();
    float *g=new float[modelSize]();
    float *ss=new float[dataSize]();
    float *gg=new float[dataSize]();
    float *r=new float[dataSize]();
    
    mynegate(r,data,dataSize);
    Fop->forward(true,model,r);
    double rdr=dot_product(r,r,dataSize);
    fprintf(stderr,"initial residual norm %e\n",rdr);
   
    Fop->adjoint(false,g,r);
    
//    write("grad",g,modelSize);
//    to_header("grad","n1",modelSize,"o1",0,"d1",1.);
//    to_header("grad","n2",1,"o2",0,"d2",1.);

    Fop->forward(false,g,gg);
    double gdg=dot_product(gg,gg,dataSize);
    if(gdg<FLT_MIN){
        fprintf(stderr,"gdg too small\n");
        return;
    }
    double gdr=dot_product(gg,r,dataSize);
    double alpha=-gdr/gdg;
    scale(s,alpha,g,modelSize);
    scale(ss,alpha,gg,dataSize);
    add(model,model,s,modelSize);
    add(r,r,ss,dataSize);

    for(int i=1;i<niter;i++){
//        write("model",model,modelSize,std::ios_base::app);
//        to_header("model","n2",i+1);
        rdr=dot_product(r,r,dataSize);
        fprintf(stderr,"iteration %d residual norm %e\n",i,rdr);
        Fop->adjoint(false,g,r);
//        write("grad",g,modelSize);
//        to_header("grad","n2",i+1);
        Fop->forward(false,g,gg);
        gdg=dot_product(gg,gg,dataSize);
        double sds=dot_product(ss,ss,dataSize);
        if(gdg<FLT_MIN || sds<FLT_MIN){
            fprintf(stderr,"gdg or sds too small\n");
            break;
        }
        double gds=dot_product(gg,ss,dataSize);
        double determ=gdg*sds*std::max(1.-(gds/gdg)*(gds/sds),(double)FLT_MIN);
        gdr=-dot_product(gg,r,dataSize);
        double sdr=-dot_product(ss,r,dataSize);
        alpha=(sds*gdr-gds*sdr)/determ;
        double beta=(-gds*gdr+gdg*sdr)/determ;
        lin_comb(s,alpha,g,beta,s,modelSize);
        lin_comb(ss,alpha,gg,beta,ss,dataSize);
        add(model,model,s,modelSize);
        add(r,r,ss,dataSize);
    }

    delete []s;delete []g;delete []ss;delete []gg;delete []r;
    return;
}

void simplePositiveSolver(Operator *Fop,float *model,const float *data,int niter){
    //solve Fm-d=0 for positive m
    long long modelSize=Fop->_modelSize;
    long long dataSize=Fop->_dataSize;

    float *s=new float[modelSize]();
    float *temps=new float[modelSize]();
    float *g=new float[modelSize]();
    float *ss=new float[dataSize]();
    float *gg=new float[dataSize]();
    float *r=new float[dataSize]();
    
    mynegate(r,data,dataSize);
    Fop->forward(true,model,r);
    double rdr=dot_product(r,r,dataSize);
    fprintf(stderr,"initial residual norm %e\n",rdr);
   
    Fop->adjoint(false,g,r);
    Fop->forward(false,g,gg);
    double gdg=dot_product(gg,gg,dataSize);
    if(gdg<FLT_MIN){
        fprintf(stderr,"gdg too small\n");
        return;
    }
    double gdr=dot_product(gg,r,dataSize);
    double alpha=-gdr/gdg;
    scale(temps,alpha,g,modelSize);
    vector<float> rho0;
    for(size_t i=0;i<modelSize;i++){
        if(model[i]+temps[i]<=0.) rho0.push_back(model[i]/(-temps[i]));
    }
    float scalefactor=*min_element(rho0.begin(),rho0.end());
    alpha*=scalefactor;
    scale(s,alpha,g,modelSize);
    scale(ss,alpha,gg,dataSize);
    add(model,model,s,modelSize);
    add(r,r,ss,dataSize);

    for(int i=1;i<niter;i++){
        rdr=dot_product(r,r,dataSize);
        fprintf(stderr,"iteration %d residual norm %e\n",i,rdr);
        Fop->adjoint(false,g,r);
        Fop->forward(false,g,gg);
        gdg=dot_product(gg,gg,dataSize);
        double sds=dot_product(ss,ss,dataSize);
        if(gdg<FLT_MIN || sds<FLT_MIN){
            fprintf(stderr,"gdg or sds too small\n");
            break;
        }
        double gds=dot_product(gg,ss,dataSize);
        double determ=gdg*sds*std::max(1.-(gds/gdg)*(gds/sds),(double)FLT_MIN);
        gdr=-dot_product(gg,r,dataSize);
        double sdr=-dot_product(ss,r,dataSize);
        alpha=(sds*gdr-gds*sdr)/determ;
        double beta=(-gds*gdr+gdg*sdr)/determ;
        lin_comb(temps,alpha,g,beta,s,modelSize);
        vector<float> rho;
        for(size_t j=0;j<modelSize;j++){
            if(model[j]+temps[j]<=0.) rho.push_back(model[j]/(-temps[j]));
        }
        scalefactor=*min_element(rho.begin(),rho.end());
        alpha*=scalefactor;
        beta*=scalefactor;
        lin_comb(s,alpha,g,beta,s,modelSize);
        lin_comb(ss,alpha,gg,beta,ss,dataSize);
        add(model,model,s,modelSize);
        add(r,r,ss,dataSize);
    }

    delete []s;delete []temps;delete []g;delete []ss;delete []gg;delete []r;
    return;
}

void regularizedSolver(Operator *Fop,Operator *Aop,float eps,float *model, const float *data,int niter){
    //solve Fm-d=0 and eAm=0
    long long modelSize=Fop->_modelSize;
    long long dataSizeF=Fop->_dataSize;
    long long dataSizeA=Aop->_dataSize;
    long long dataSize=dataSizeF+dataSizeA;

    float *s=new float[modelSize]();
    float *g=new float[modelSize]();
    float *ss=new float[dataSize]();
    float *gg=new float[dataSize](); 
    float *gd=gg,*gm=gg+dataSizeF;
    float *r=new float[dataSize]();
    float *rd=r,*rm=r+dataSizeF;
    
    mynegate(rd,data,dataSizeF);
    Fop->forward(true,model,rd);
    Aop->forward(false,model,rm);
    scale(rm,eps,rm,dataSizeA);
    double rd2=dot_product(rd,rd,dataSizeF);
    double rm2=dot_product(rm,rm,dataSizeA);
    fprintf(stderr,"initial data-fitting residual %e and model regularization %e\n",rd2,rm2);
   
    Aop->adjoint(false,g,rm);
    scale(g,eps,g,modelSize);
    Fop->adjoint(true,g,rd);
    Fop->forward(false,g,gd);
    Aop->forward(false,g,gm);
    scale(gm,eps,gm,dataSizeA);
    double gdg=dot_product(gg,gg,dataSize);
    if(gdg<FLT_MIN){
        fprintf(stderr,"gdg too small\n");
        return;
    }
    double gdr=dot_product(gg,r,dataSize);
    double alpha=-gdr/gdg;
    scale(s,alpha,g,modelSize);
    scale(ss,alpha,gg,dataSize);
    add(model,model,s,modelSize);
    add(r,r,ss,dataSize);

    for(int i=1;i<niter;i++){
        rd2=dot_product(rd,rd,dataSizeF);
        rm2=dot_product(rm,rm,dataSizeA);
        fprintf(stderr,"iteration %d data-fitting residual %e and model regularization %e\n",i,rd2,rm2);
        Aop->adjoint(false,g,rm);
        scale(g,eps,g,modelSize);
        Fop->adjoint(true,g,rd);
        Fop->forward(false,g,gd);
        Aop->forward(false,g,gm);
        gdg=dot_product(gg,gg,dataSize);
        double sds=dot_product(ss,ss,dataSize);
        if(gdg<FLT_MIN || sds<FLT_MIN){
            fprintf(stderr,"gdg or sds too small\n");
            break;
        }
        double gds=dot_product(gg,ss,dataSize);
        double determ=gdg*sds*std::max(1.-(gds/gdg)*(gds/sds),(double)FLT_MIN);
        gdr=-dot_product(gg,r,dataSize);
        double sdr=-dot_product(ss,r,dataSize);
        alpha=(sds*gdr-gds*sdr)/determ;
        double beta=(-gds*gdr+gdg*sdr)/determ;
        lin_comb(s,alpha,g,beta,s,modelSize);
        lin_comb(ss,alpha,gg,beta,ss,dataSize);
        add(model,model,s,modelSize);
        add(r,r,ss,dataSize);
    }

    delete []s;delete []g;delete []ss;delete []gg;delete []r;
    return;
}

float normalize(float *x,int n){
    float b=sqrt(dot_product(x,x,n));
    scale(x,1./b,n);
    return b;
}

void lsqr(Operator *Fop,float *model,const float *data,int niter){
    long long modelSize=Fop->_modelSize;
    long long dataSize=Fop->_dataSize;
    
    float *u=new float[dataSize];
    memcpy(u,data,dataSize*sizeof(float));

    memset(model,0,modelSize);

    float beta=normalize(u,dataSize);
    
    float *v=new float[modelSize];
    Fop->adjoint(false,v,u);
    float alfa=normalize(v,modelSize);

    float *w=new float[modelSize];
    memcpy(w,v,modelSize*sizeof(float));

    float rhobar=alfa;
    float phibar=beta;

    for(int i=0;i<niter;i++){
        scale(u,-alfa,dataSize);
        Fop->forward(true,v,u);
        beta=normalize(u,dataSize);
        
        scale(v,-beta,modelSize);
        Fop->adjoint(true,v,u);
        alfa=normalize(v,modelSize);

        float rho=sqrt(rhobar*rhobar+beta*beta);
        if(rho<FLT_MIN){
            fprintf(stderr,"rho too small\n");
            break;
        }
        float c=rhobar/rho;
        float s=beta/rho;
        float teta=s*alfa;
        rhobar=-c*alfa;
        float phi=c*phibar;
        phibar=s*phibar;
        float t1=phi/rho;
        float t2=-teta/rho;

        scale_add(model,t1,w,modelSize);
        lin_comb(w,1.f,v,t2,w,modelSize);
    }

    delete []u;delete []v;delete []w;
    return;
}

void CG(Operator *A,float *x,const float *b,int niter){
// solve Ax=b for spd matrix A following Algorithm 5.2 in Nocedal book
    
    long long n=A->_modelSize; //modelSize=dataSize
    float *r=new float[n];
    float *rmb=new float[n];
    float *p=new float[n];
    float *Ap=new float[n];

    A->forward(false,x,r); 
    subtract(r,r,b,n);
    mynegate(p,r,n);

    for(int k=0;k<niter;k++){
        subtract(rmb,r,b,n);
        float anorm=dot_product(x,rmb,n);
        fprintf(stderr,"iter %d A norm %.10f\n",k,anorm);
        A->forward(false,p,Ap);
        float ptAp=dot_product(p,Ap,n);
        float rtr=dot_product(r,r,n);
        float alpha=rtr/ptAp;
        scale_add(x,alpha,p,n);
        scale_add(r,alpha,Ap,n);
        float beta=dot_product(r,r,n)/rtr;
        lin_comb(p,-1.,r,beta,p,n);
    }

    delete []r;delete []rmb;delete []p;delete []Ap;
    return;
}

int indefCG(Operator *A,float *x,const float *b,int niter_max){
// solve Ax=-b for symmetric indefinite matrix A following Algorithm 7.1 in Nocedal book
// usually b is the gradient and A is the Hessian
    
    long long n=A->_modelSize; //modelSize=dataSize
    float *r=new float[n];
    float *rmb=new float[n];
    float *p=new float[n];
    float *Ap=new float[n];

    int niter=0;
    float bnorm=sqrt(dot_product(b,b,n));
    float epsilon=min(0.5f,sqrt(bnorm));

    memset(x,0,n*sizeof(float));
    memcpy(r,b,n*sizeof(float));
    mynegate(p,r,n);

//    to_header("model","n1",n,"o1",0,"d1",1.);
    fprintf(stderr,"niter_max %d",niter_max);    
    for(int k=0;k<niter_max;k++){
        add(rmb,r,b,n);
        float anorm=dot_product(x,rmb,n);
        fprintf(stderr,"iter %d A norm %.10f\n",k,anorm);
        
        niter++;
        A->forward(false,p,Ap);
        float ptAp=dot_product(p,Ap,n);
        fprintf(stderr,"curvature %.10f\n",ptAp);
        if(ptAp<=CURVE_TOL){
            fprintf(stderr,"curvature too small. terminate after %d iterations.\n",niter);
            if(k==0){
                memcpy(x,p,n*sizeof(float));
                fprintf(stderr,"use steepest descent direction\n");
            }
            return niter;
        }

        float rtr=dot_product(r,r,n);
        float alpha=rtr/ptAp;
        scale_add(x,alpha,p,n);
//        write("model",x,n,std::ios_base::app);
//        to_header("model","n2",niter);
        scale_add(r,alpha,Ap,n);
        float rtr1=dot_product(r,r,n);
        if(rtr1<epsilon){
            fprintf(stderr,"termination condition passed after %d iterations.\n",niter);
            return niter;
        }
        float beta=rtr1/rtr;
        lin_comb(p,-1.,r,beta,p,n);
    }
    
    delete []r;delete []rmb;delete []p;delete []Ap;
    return niter;
}

