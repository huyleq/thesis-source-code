#ifndef LAPLACIAN3D_H
#define LAPLACIAN3D_H

#define cc0 -2.84722222222
#define cc1 1.6
#define cc2 -0.2
#define cc3 0.02539682539
#define cc4 -0.00178571428

#define aa1 1.1962890625
#define aa2 -0.07975260416
#define aa3 0.0095703125
#define aa4 -0.00069754464

#define ONETHIRD 0.33333333333
#define TWOTHIRD 0.66666666666

#define BLOCKSIZE 16

extern "C"{
void forward(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ);
}

extern "C"{
void gradient(float *gv,float *geps,float *gdel,int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ,float *curSigmaXa,float *curSigmaZa);
}

extern "C"{
void adjoint(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ,float *cc);
}

extern "C"{
void adjoint1(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaY,float *curSigmaY,float *prevSigmaZ,float *curSigmaZ,float *cc);
}

extern "C"{
void forwardCij(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *c11,float *c13,float *c33,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ);
}

extern "C"{
void bornCij(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *c11,float *c13,float *c33,float *dc11,float *dc13,float *dc33,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ,float *prevDSigmaX,float *curDSigmaX,float *prevDSigmaZ,float *curDSigmaZ);
}

extern "C"{
void gradientCij(float *gc11,float *gc13,float *gc33,int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *c11,float *c13,float *c33,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ,float *curSigmaXa,float *curSigmaZa);
}

extern "C"{
void adjointCij(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *c11,float *c13,float *c33,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ,float *cc);
}

extern "C"{
void forwardAdjoint(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *r11,float *r13,float *r33,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ,float *cc);
}

extern "C"{
void dxdydz(int n,int nx,int nxy,float dx,float dy,float dz,float *dSigmaXdx,float *dSigmaYdy,float *dSigmaZdz,float *sigmaX,float *sigmaY,float *sigmaZ);
}

extern "C"{
void forwardAdjoint1(int n,int nx,int nxy,float dx,float dy,float dz,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaY,float *curSigmaY,float *prevSigmaZ,float *curSigmaZ,float *dSigmaXdx,float *dSigmaYdy,float *dSigmaZdz,float *aa);
}

extern "C"{
void forward2(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ);
}

extern "C"{
void adjoint2(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ,float *cc);
}

extern "C"{
void forward3(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ);
}

extern "C"{
void forward4(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ);
}

extern "C"{
void forward5(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ);
}

extern "C"{
void forward6(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ);
}

extern "C"{
void forward7(int n,int nx,int nxy,float dx2,float dy2,float dz2,float dt2,float *v,float *eps,float *del,float *prevSigmaX,float *curSigmaX,float *prevSigmaZ,float *curSigmaZ);
}

#endif
