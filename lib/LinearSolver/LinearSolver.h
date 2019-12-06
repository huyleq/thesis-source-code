#ifndef LINEARSOLVER_H
#define LINEARSOLVER_H

#define CURVE_TOL 1e-16

class Operator{
    public:
    Operator(long long modelSize,long long dataSize):_modelSize(modelSize),_dataSize(dataSize){};
    virtual void forward(bool add,const float *model,float *data)=0;
    virtual void adjoint(bool add,float *model,const float *data)=0;
    long long _modelSize,_dataSize;
};

class MatMul:public Operator{
    public:
    MatMul(float *mat,int nrow,int ncol):Operator(ncol,nrow),_mat(mat){};
    void forward(bool add,const float *model,float *data); 
    void adjoint(bool add,float *model,const float *data); 
    float *_mat;
};

class Lap2d:public Operator{
    public:
    Lap2d(int nx,int ny):Operator(nx*ny,nx*ny),_nx(nx),_ny(ny){};
    void forward(bool add,const float *model,float *data); 
    void adjoint(bool add,float *model,const float *data); 
    int _nx,_ny;
};

void simpleSolver(Operator *Fop,float *model,const float *data,int niter);

void simplePositiveSolver(Operator *Fop,float *model,const float *data,int niter);

void regularizedSolver(Operator *Fop,Operator *Aop,float eps,float *model, const float *data,int niter);

float normalize(float *x,int n);

void lsqr(Operator *Fop,float *model,const float *data,int niter);

void CG(Operator *A,float *x,const float *b,int niter);

int indefCG(Operator *A,float *x,const float *b,int niter_max);

#endif
