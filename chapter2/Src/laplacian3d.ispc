#include "laplacian3d.h"

export void forward(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[]){
    foreach(i=0...n){
        float c33=v[i]*v[i];
        float c11=c33*(1.f+2.f*eps[i]);
        float c13=c33*sqrt(1.f+2.f*del[i]);
        float d2SigmaXdx2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+1]+curSigmaX[i-1])
                                          +cc2*(curSigmaX[i+2]+curSigmaX[i-2])
                                          +cc3*(curSigmaX[i+3]+curSigmaX[i-3])
                                          +cc4*(curSigmaX[i+4]+curSigmaX[i-4]))/dx2;
        float d2SigmaXdy2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nx]+curSigmaX[i-nx])
                                          +cc2*(curSigmaX[i+2*nx]+curSigmaX[i-2*nx])
                                          +cc3*(curSigmaX[i+3*nx]+curSigmaX[i-3*nx])
                                          +cc4*(curSigmaX[i+4*nx]+curSigmaX[i-4*nx]))/dy2;
        float d2SigmaZdz2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nxy]+curSigmaZ[i-nxy])
                                          +cc2*(curSigmaZ[i+2*nxy]+curSigmaZ[i-2*nxy])
                                          +cc3*(curSigmaZ[i+3*nxy]+curSigmaZ[i-3*nxy])
                                          +cc4*(curSigmaZ[i+4*nxy]+curSigmaZ[i-4*nxy]))/dz2;
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(c11*(d2SigmaXdx2+d2SigmaXdy2)+c13*d2SigmaZdz2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*(c13*(d2SigmaXdx2+d2SigmaXdy2)+c33*d2SigmaZdz2)-prevSigmaZ[i];
    }
}

export void adjoint(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[],uniform float cc[]){
    foreach(i=0...n){
        float c33=v[i]*v[i];
        float c11=c33*(1.f+2.f*eps[i]);
        float c13=c33*sqrt(1.f+2.f*del[i]);
        
        float d2SigmaXdx2=cc0*(c11*curSigmaX[i]+c13*curSigmaZ[i]);
        float d2SigmaXdy2=d2SigmaXdx2;
        float d2SigmaZdz2=cc0*(c13*curSigmaX[i]+c33*curSigmaZ[i]);
        
        for(int j=1;j<5;j++){
            size_t ij=i+j;
            c33=v[ij]*v[ij];
            c11=c33*(1.f+2.f*eps[ij]);
            c13=c33*sqrt(1.f+2.f*del[ij]);
            float temp=c11*curSigmaX[ij]+c13*curSigmaZ[ij];
            
            ij=i-j;
            c33=v[ij]*v[ij];
            c11=c33*(1.f+2.f*eps[ij]);
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp+=c11*curSigmaX[ij]+c13*curSigmaZ[ij];
            
            d2SigmaXdx2+=cc[j]*temp;
            
            ij=i+j*nx;
            c33=v[ij]*v[ij];
            c11=c33*(1.f+2.f*eps[ij]);
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp=c11*curSigmaX[ij]+c13*curSigmaZ[ij];
            
            ij=i-j*nx;
            c33=v[ij]*v[ij];
            c11=c33*(1.f+2.f*eps[ij]);
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp+=c11*curSigmaX[ij]+c13*curSigmaZ[ij];
            
            d2SigmaXdy2+=cc[j]*temp;
            
            ij=i+j*nxy;
            c33=v[ij]*v[ij];
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp=c13*curSigmaX[ij]+c33*curSigmaZ[ij];
            
            ij=i-j*nxy;
            c33=v[ij]*v[ij];
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp+=c13*curSigmaX[ij]+c33*curSigmaZ[ij];
            
            d2SigmaZdz2+=cc[j]*temp;
        }
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(d2SigmaXdx2/dx2+d2SigmaXdy2/dy2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*d2SigmaZdz2/dz2-prevSigmaZ[i];
    }
}

export void gradient(uniform float gv[],uniform float geps[],uniform float gdel[],uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[],uniform float curSigmaXa[],uniform float curSigmaZa[]){
    foreach(i=0...n){
        float c33=v[i]*v[i];
        float c11=c33*(1.f+2.f*eps[i]);
        float sqrt12del=sqrt(1.f+2.f*del[i]);
        float c13=c33*sqrt12del;

        float d2SigmaXdx2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+1]+curSigmaX[i-1])
                                          +cc2*(curSigmaX[i+2]+curSigmaX[i-2])
                                          +cc3*(curSigmaX[i+3]+curSigmaX[i-3])
                                          +cc4*(curSigmaX[i+4]+curSigmaX[i-4]))/dx2;
        float d2SigmaXdy2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nx]+curSigmaX[i-nx])
                                          +cc2*(curSigmaX[i+2*nx]+curSigmaX[i-2*nx])
                                          +cc3*(curSigmaX[i+3*nx]+curSigmaX[i-3*nx])
                                          +cc4*(curSigmaX[i+4*nx]+curSigmaX[i-4*nx]))/dy2;
        float d2SigmaZdz2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nxy]+curSigmaZ[i-nxy])
                                          +cc2*(curSigmaZ[i+2*nxy]+curSigmaZ[i-2*nxy])
                                          +cc3*(curSigmaZ[i+3*nxy]+curSigmaZ[i-3*nxy])
                                          +cc4*(curSigmaZ[i+4*nxy]+curSigmaZ[i-4*nxy]))/dz2;
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(c11*(d2SigmaXdx2+d2SigmaXdy2)+c13*d2SigmaZdz2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*(c13*(d2SigmaXdx2+d2SigmaXdy2)+c33*d2SigmaZdz2)-prevSigmaZ[i];
        
        float gc11=curSigmaXa[i]*(d2SigmaXdx2+d2SigmaXdy2);
        float gc33=curSigmaZa[i]*d2SigmaZdz2;
        float gc13=curSigmaXa[i]*d2SigmaZdz2+curSigmaZa[i]*(d2SigmaXdx2+d2SigmaXdy2);
        
        gv[i]+=2.f*v[i]*(gc11*(1.f+2.f*eps[i])+gc13*sqrt12del+gc33);
        geps[i]+=gc11*2.f*c33;
        gdel[i]+=gc13*c33/sqrt12del;
    }
}

export void forwardCij(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float c11[],uniform float c13[],uniform float c33[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[]){
    foreach(i=0...n){
        float d2SigmaXdx2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+1]+curSigmaX[i-1])
                                          +cc2*(curSigmaX[i+2]+curSigmaX[i-2])
                                          +cc3*(curSigmaX[i+3]+curSigmaX[i-3])
                                          +cc4*(curSigmaX[i+4]+curSigmaX[i-4]))/dx2;
        float d2SigmaXdy2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nx]+curSigmaX[i-nx])
                                          +cc2*(curSigmaX[i+2*nx]+curSigmaX[i-2*nx])
                                          +cc3*(curSigmaX[i+3*nx]+curSigmaX[i-3*nx])
                                          +cc4*(curSigmaX[i+4*nx]+curSigmaX[i-4*nx]))/dy2;
        float d2SigmaZdz2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nxy]+curSigmaZ[i-nxy])
                                          +cc2*(curSigmaZ[i+2*nxy]+curSigmaZ[i-2*nxy])
                                          +cc3*(curSigmaZ[i+3*nxy]+curSigmaZ[i-3*nxy])
                                          +cc4*(curSigmaZ[i+4*nxy]+curSigmaZ[i-4*nxy]))/dz2;
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(c11[i]*(d2SigmaXdx2+d2SigmaXdy2)+c13[i]*d2SigmaZdz2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*(c13[i]*(d2SigmaXdx2+d2SigmaXdy2)+c33[i]*d2SigmaZdz2)-prevSigmaZ[i];
    }
}

export void bornCij(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float c11[],uniform float c13[],uniform float c33[],uniform float dc11[],uniform float dc13[],uniform float dc33[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[],uniform float prevDSigmaX[],uniform float curDSigmaX[],uniform float prevDSigmaZ[],uniform float curDSigmaZ[]){
    foreach(i=0...n){
        float d2SigmaXdx2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+1]+curSigmaX[i-1])
                                          +cc2*(curSigmaX[i+2]+curSigmaX[i-2])
                                          +cc3*(curSigmaX[i+3]+curSigmaX[i-3])
                                          +cc4*(curSigmaX[i+4]+curSigmaX[i-4]))/dx2;
        float d2SigmaXdy2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nx]+curSigmaX[i-nx])
                                          +cc2*(curSigmaX[i+2*nx]+curSigmaX[i-2*nx])
                                          +cc3*(curSigmaX[i+3*nx]+curSigmaX[i-3*nx])
                                          +cc4*(curSigmaX[i+4*nx]+curSigmaX[i-4*nx]))/dy2;
        float d2SigmaZdz2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nxy]+curSigmaZ[i-nxy])
                                          +cc2*(curSigmaZ[i+2*nxy]+curSigmaZ[i-2*nxy])
                                          +cc3*(curSigmaZ[i+3*nxy]+curSigmaZ[i-3*nxy])
                                          +cc4*(curSigmaZ[i+4*nxy]+curSigmaZ[i-4*nxy]))/dz2;
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(c11[i]*(d2SigmaXdx2+d2SigmaXdy2)+c13[i]*d2SigmaZdz2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*(c13[i]*(d2SigmaXdx2+d2SigmaXdy2)+c33[i]*d2SigmaZdz2)-prevSigmaZ[i];
        
        float d2DSigmaXdx2=(cc0*curDSigmaX[i]+cc1*(curDSigmaX[i+1]+curDSigmaX[i-1])
                                            +cc2*(curDSigmaX[i+2]+curDSigmaX[i-2])
                                            +cc3*(curDSigmaX[i+3]+curDSigmaX[i-3])
                                            +cc4*(curDSigmaX[i+4]+curDSigmaX[i-4]))/dx2;
        float d2DSigmaXdy2=(cc0*curDSigmaX[i]+cc1*(curDSigmaX[i+nx]+curDSigmaX[i-nx])
                                            +cc2*(curDSigmaX[i+2*nx]+curDSigmaX[i-2*nx])
                                            +cc3*(curDSigmaX[i+3*nx]+curDSigmaX[i-3*nx])
                                            +cc4*(curDSigmaX[i+4*nx]+curDSigmaX[i-4*nx]))/dy2;
        float d2DSigmaZdz2=(cc0*curDSigmaZ[i]+cc1*(curDSigmaZ[i+nxy]+curDSigmaZ[i-nxy])
                                            +cc2*(curDSigmaZ[i+2*nxy]+curDSigmaZ[i-2*nxy])
                                            +cc3*(curDSigmaZ[i+3*nxy]+curDSigmaZ[i-3*nxy])
                                            +cc4*(curDSigmaZ[i+4*nxy]+curDSigmaZ[i-4*nxy]))/dz2;
        prevDSigmaX[i]=2.f*curDSigmaX[i]+dt2*(c11[i]*(d2DSigmaXdx2+d2DSigmaXdy2)+c13[i]*d2DSigmaZdz2+dc11[i]*(d2SigmaXdx2+d2SigmaXdy2)+dc13[i]*d2SigmaZdz2)-prevDSigmaX[i];
        prevDSigmaZ[i]=2.f*curDSigmaZ[i]+dt2*(c13[i]*(d2DSigmaXdx2+d2DSigmaXdy2)+c33[i]*d2DSigmaZdz2+dc13[i]*(d2SigmaXdx2+d2SigmaXdy2)+dc33[i]*d2SigmaZdz2)-prevDSigmaZ[i];
    }
}

export void adjointCij(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float c11[],uniform float c13[],uniform float c33[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[],uniform float cc[]){
    foreach(i=0...n){
        float d2SigmaXdx2=cc0*(c11[i]*curSigmaX[i]+c13[i]*curSigmaZ[i]);
        float d2SigmaXdy2=d2SigmaXdx2;
        float d2SigmaZdz2=cc0*(c13[i]*curSigmaX[i]+c33[i]*curSigmaZ[i]);
        
        for(int j=1;j<5;j++){
            size_t ij=i+j;
            float temp=c11[ij]*curSigmaX[ij]+c13[ij]*curSigmaZ[ij];
            
            ij=i-j;
            temp+=c11[ij]*curSigmaX[ij]+c13[ij]*curSigmaZ[ij];
            
            d2SigmaXdx2+=cc[j]*temp;
            
            ij=i+j*nx;
            temp=c11[ij]*curSigmaX[ij]+c13[ij]*curSigmaZ[ij];
            
            ij=i-j*nx;
            temp+=c11[ij]*curSigmaX[ij]+c13[ij]*curSigmaZ[ij];
            
            d2SigmaXdy2+=cc[j]*temp;
            
            ij=i+j*nxy;
            temp=c13[ij]*curSigmaX[ij]+c33[ij]*curSigmaZ[ij];
            
            ij=i-j*nxy;
            temp+=c13[ij]*curSigmaX[ij]+c33[ij]*curSigmaZ[ij];
            
            d2SigmaZdz2+=cc[j]*temp;
        }
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(d2SigmaXdx2/dx2+d2SigmaXdy2/dy2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*d2SigmaZdz2/dz2-prevSigmaZ[i];
    }
}

export void gradientCij(uniform float gc11[],uniform float gc13[],uniform float gc33[],uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[],uniform float curSigmaXa[],uniform float curSigmaZa[]){
    foreach(i=0...n){
        float c33=v[i]*v[i];
        float c11=c33*(1.f+2.f*eps[i]);
        float sqrt12del=sqrt(1.f+2.f*del[i]);
        float c13=c33*sqrt12del;

        float d2SigmaXdx2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+1]+curSigmaX[i-1])
                                          +cc2*(curSigmaX[i+2]+curSigmaX[i-2])
                                          +cc3*(curSigmaX[i+3]+curSigmaX[i-3])
                                          +cc4*(curSigmaX[i+4]+curSigmaX[i-4]))/dx2;
        float d2SigmaXdy2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nx]+curSigmaX[i-nx])
                                          +cc2*(curSigmaX[i+2*nx]+curSigmaX[i-2*nx])
                                          +cc3*(curSigmaX[i+3*nx]+curSigmaX[i-3*nx])
                                          +cc4*(curSigmaX[i+4*nx]+curSigmaX[i-4*nx]))/dy2;
        float d2SigmaZdz2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nxy]+curSigmaZ[i-nxy])
                                          +cc2*(curSigmaZ[i+2*nxy]+curSigmaZ[i-2*nxy])
                                          +cc3*(curSigmaZ[i+3*nxy]+curSigmaZ[i-3*nxy])
                                          +cc4*(curSigmaZ[i+4*nxy]+curSigmaZ[i-4*nxy]))/dz2;
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(c11*(d2SigmaXdx2+d2SigmaXdy2)+c13*d2SigmaZdz2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*(c13*(d2SigmaXdx2+d2SigmaXdy2)+c33*d2SigmaZdz2)-prevSigmaZ[i];
        
        gc11[i]+=curSigmaXa[i]*(d2SigmaXdx2+d2SigmaXdy2);
        gc33[i]+=curSigmaZa[i]*d2SigmaZdz2;
        gc13[i]+=curSigmaXa[i]*d2SigmaZdz2+curSigmaZa[i]*(d2SigmaXdx2+d2SigmaXdy2);
    }
}

export void forwardAdjoint(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float r11[],uniform float r13[],uniform float r33[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[],uniform float cc[]){
    foreach(i=0...n){
        float d2SigmaXdx2=cc0*(r11[i]*curSigmaX[i]+r13[i]*curSigmaZ[i]);
        float d2SigmaXdy2=d2SigmaXdx2;
        float d2SigmaZdz2=cc0*(r13[i]*curSigmaX[i]+r33[i]*curSigmaZ[i]);
        
        for(int j=1;j<5;j++){
            size_t ij=i+j;
            float temp=r11[ij]*curSigmaX[ij]+r13[ij]*curSigmaZ[ij];
            
            ij=i-j;
            temp+=r11[ij]*curSigmaX[ij]+r13[ij]*curSigmaZ[ij];
            
            d2SigmaXdx2+=cc[j]*temp;
            
            ij=i+j*nx;
            temp=r11[ij]*curSigmaX[ij]+r13[ij]*curSigmaZ[ij];
            
            ij=i-j*nx;
            temp+=r11[ij]*curSigmaX[ij]+r13[ij]*curSigmaZ[ij];
            
            d2SigmaXdy2+=cc[j]*temp;
            
            ij=i+j*nxy;
            temp=r13[ij]*curSigmaX[ij]+r33[ij]*curSigmaZ[ij];
            
            ij=i-j*nxy;
            temp+=r13[ij]*curSigmaX[ij]+r33[ij]*curSigmaZ[ij];
            
            d2SigmaZdz2+=cc[j]*temp;
        }
        d2SigmaXdx2=d2SigmaXdx2/dx2;
        d2SigmaXdy2=d2SigmaXdy2/dy2;
        d2SigmaZdz2=d2SigmaZdz2/dz2;

        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(r11[i]*(d2SigmaXdx2+d2SigmaXdy2)+r13[i]*d2SigmaZdz2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*(r13[i]*(d2SigmaXdx2+d2SigmaXdy2)+r33[i]*d2SigmaZdz2)-prevSigmaZ[i];
    }
}

export void adjoint1(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaY[],uniform float curSigmaY[],uniform float prevSigmaZ[],uniform float curSigmaZ[],uniform float cc[]){
    foreach(i=0...n){
        float c33=v[i]*v[i];
        float c11=c33*(1.f+2.f*eps[i]);
        float c13=c33*sqrt(1.f+2.f*del[i]);
        
        float d2SigmaXdx2=cc0*(c11*curSigmaX[i]+c11*curSigmaY[i]+c13*curSigmaZ[i]);
        float d2SigmaYdy2=d2SigmaXdx2;
        float d2SigmaZdz2=cc0*(c13*curSigmaX[i]+c13*curSigmaY[i]+c33*curSigmaZ[i]);
        
        for(int j=1;j<5;j++){
            size_t ij=i+j;
            c33=v[ij]*v[ij];
            c11=c33*(1.f+2.f*eps[ij]);
            c13=c33*sqrt(1.f+2.f*del[ij]);
            float temp=c11*curSigmaX[ij]+c11*curSigmaY[ij]+c13*curSigmaZ[ij];
            
            ij=i-j;
            c33=v[ij]*v[ij];
            c11=c33*(1.f+2.f*eps[ij]);
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp+=c11*curSigmaX[ij]+c11*curSigmaY[ij]+c13*curSigmaZ[ij];
            
            d2SigmaXdx2+=cc[j]*temp;
            
            ij=i+j*nx;
            c33=v[ij]*v[ij];
            c11=c33*(1.f+2.f*eps[ij]);
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp=c11*curSigmaX[ij]+c11*curSigmaY[ij]+c13*curSigmaZ[ij];
            
            ij=i-j*nx;
            c33=v[ij]*v[ij];
            c11=c33*(1.f+2.f*eps[ij]);
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp+=c11*curSigmaX[ij]+c11*curSigmaY[ij]+c13*curSigmaZ[ij];
            
            d2SigmaYdy2+=cc[j]*temp;
            
            ij=i+j*nxy;
            c33=v[ij]*v[ij];
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp=c13*curSigmaX[ij]+c13*curSigmaY[ij]+c33*curSigmaZ[ij];
            
            ij=i-j*nxy;
            c33=v[ij]*v[ij];
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp+=c13*curSigmaX[ij]+c13*curSigmaY[ij]+c33*curSigmaZ[ij];
            
            d2SigmaZdz2+=cc[j]*temp;
        }
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*d2SigmaXdx2/dx2-prevSigmaX[i];
        prevSigmaY[i]=2.f*curSigmaY[i]+dt2*d2SigmaYdy2/dy2-prevSigmaY[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*d2SigmaZdz2/dz2-prevSigmaZ[i];
    }
}

export void dxdydz(uniform int n,uniform int nx,uniform int nxy,uniform float dx,uniform float dy,uniform float dz,uniform float dSigmaXdx[],uniform float dSigmaYdy[],uniform float dSigmaZdz[],uniform float sigmaX[],uniform float sigmaY[],uniform float sigmaZ[]){
    foreach(i=0...n){
        float dSigmaX=aa1*(sigmaX[i+1]-sigmaX[i])+
                      aa2*(sigmaX[i+2]-sigmaX[i-1])+
                      aa3*(sigmaX[i+3]-sigmaX[i-2])+
                      aa4*(sigmaX[i+4]-sigmaX[i-3]);
        float dSigmaY=aa1*(sigmaY[i+nx]-sigmaY[i])+
                      aa2*(sigmaY[i+2*nx]-sigmaY[i-nx])+
                      aa3*(sigmaY[i+3*nx]-sigmaY[i-2*nx])+
                      aa4*(sigmaY[i+4*nx]-sigmaY[i-3*nx]);
        float dSigmaZ=aa1*(sigmaZ[i+nxy]-sigmaZ[i])+
                      aa2*(sigmaZ[i+2*nxy]-sigmaZ[i-nxy])+
                      aa3*(sigmaZ[i+3*nxy]-sigmaZ[i-2*nxy])+
                      aa4*(sigmaZ[i+4*nxy]-sigmaZ[i-3*nxy]);
        dSigmaXdx[i]=dSigmaX/dx;
        dSigmaYdy[i]=dSigmaY/dy;
        dSigmaZdz[i]=dSigmaZ/dz;
    }
}

export void forwardAdjoint1(uniform int n,uniform int nx,uniform int nxy,uniform float dx,uniform float dy,uniform float dz,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaY[],uniform float curSigmaY[],uniform float prevSigmaZ[],uniform float curSigmaZ[],uniform float dSigmaXdx[],uniform float dSigmaYdy[],uniform float dSigmaZdz[],uniform float aa[]){
    foreach(i=0...n){
        float dxx=0.f,dyy=0.f,dzz=0.f;
        for(int j=1;j<5;j++){
            size_t ij=i+j-1;
            float c33=v[ij]*v[ij];
            float c11=c33*(1.f+2.f*eps[ij]);
            float c13=c33*sqrt(1.f+2.f*del[ij]);
            float temp=c11*dSigmaXdx[ij]+c11*dSigmaYdy[ij]+c13*dSigmaZdz[ij];
            
            ij=i-j;
            c33=v[ij]*v[ij];
            c11=c33*(1.f+2.f*eps[ij]);
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp-=c11*dSigmaXdx[ij]+c11*dSigmaYdy[ij]+c13*dSigmaZdz[ij];
            
            dxx+=aa[j]*temp;
            
            ij=i+(j-1)*nx;
            c33=v[ij]*v[ij];
            c11=c33*(1.f+2.f*eps[ij]);
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp=c11*dSigmaXdx[ij]+c11*dSigmaYdy[ij]+c13*dSigmaZdz[ij];
            
            ij=i-j*nx;
            c33=v[ij]*v[ij];
            c11=c33*(1.f+2.f*eps[ij]);
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp-=c11*dSigmaXdx[ij]+c11*dSigmaYdy[ij]+c13*dSigmaZdz[ij];
            
            dyy+=aa[j]*temp;
            
            ij=i+(j-1)*nxy;
            c33=v[ij]*v[ij];
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp=c13*dSigmaXdx[ij]+c13*dSigmaYdy[ij]+c33*dSigmaZdz[ij];
            
            ij=i-j*nxy;
            c33=v[ij]*v[ij];
            c13=c33*sqrt(1.f+2.f*del[ij]);
            temp-=c13*dSigmaXdx[ij]+c13*dSigmaYdy[ij]+c33*dSigmaZdz[ij];
            
            dzz+=aa[j]*temp;
        }
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*dxx/dx-prevSigmaX[i];
        prevSigmaY[i]=2.f*curSigmaY[i]+dt2*dyy/dy-prevSigmaY[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*dzz/dz-prevSigmaZ[i];
    }
}

export void forward2(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[]){
    foreach(i=0...n){
        float dSigmaXdx2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+1]+curSigmaX[i-1])
                                         +cc2*(curSigmaX[i+2]+curSigmaX[i-2])
                                         +cc3*(curSigmaX[i+3]+curSigmaX[i-3])
                                         +cc4*(curSigmaX[i+4]+curSigmaX[i-4]))/dx2;
        float dSigmaXdy2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nx]+curSigmaX[i-nx])
                                         +cc2*(curSigmaX[i+2*nx]+curSigmaX[i-2*nx])
                                         +cc3*(curSigmaX[i+3*nx]+curSigmaX[i-3*nx])
                                         +cc4*(curSigmaX[i+4*nx]+curSigmaX[i-4*nx]))/dy2;
        float dSigmaXdz2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nxy]+curSigmaX[i-nxy])
                                         +cc2*(curSigmaX[i+2*nxy]+curSigmaX[i-2*nxy])
                                         +cc3*(curSigmaX[i+3*nxy]+curSigmaX[i-3*nxy])
                                         +cc4*(curSigmaX[i+4*nxy]+curSigmaX[i-4*nxy]))/dz2;
        
        float dSigmaZdx2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+1]+curSigmaZ[i-1])
                                         +cc2*(curSigmaZ[i+2]+curSigmaZ[i-2])
                                         +cc3*(curSigmaZ[i+3]+curSigmaZ[i-3])
                                         +cc4*(curSigmaZ[i+4]+curSigmaZ[i-4]))/dx2;
        float dSigmaZdy2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nx]+curSigmaZ[i-nx])
                                         +cc2*(curSigmaZ[i+2*nx]+curSigmaZ[i-2*nx])
                                         +cc3*(curSigmaZ[i+3*nx]+curSigmaZ[i-3*nx])
                                         +cc4*(curSigmaZ[i+4*nx]+curSigmaZ[i-4*nx]))/dy2;
        float dSigmaZdz2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nxy]+curSigmaZ[i-nxy])
                                         +cc2*(curSigmaZ[i+2*nxy]+curSigmaZ[i-2*nxy])
                                         +cc3*(curSigmaZ[i+3*nxy]+curSigmaZ[i-3*nxy])
                                         +cc4*(curSigmaZ[i+4*nxy]+curSigmaZ[i-4*nxy]))/dz2;
        
        float a=sqrt(1+2.f*eps[i]),b=sqrt(1+2.f*del[i]);
        float y=b/a;
        float y2=y*y;
        float z=sqrt(1-y2);
        float vpx=v[i]*a;
        float vpx2=vpx*vpx;
        float vpn=v[i]*b;
        float v2=v[i]*v[i];
        float b1=0.5*vpn*vpx,b2=0.5*v2*y;
        float a1=0.5*vpx2*(1+z);
        float d1=0.5*vpx2*(1-z);
        float a2=0.5*v2*(1-z);
        float d2=0.5*v2*(1+z);
        
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(a1*(dSigmaXdx2+dSigmaXdy2)+a2*dSigmaXdz2+b1*(dSigmaZdx2+dSigmaZdy2)+b2*dSigmaZdz2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*(b1*(dSigmaXdx2+dSigmaXdy2)+b2*dSigmaXdz2+d1*(dSigmaZdx2+dSigmaZdy2)+d2*dSigmaZdz2)-prevSigmaZ[i];
    }
}

export void adjoint2(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[],uniform float cc[]){
    foreach(i=0...n){
        float a=sqrt(1+2.f*eps[i]),b=sqrt(1+2.f*del[i]);
        float y=b/a;
        float y2=y*y;
        float z=sqrt(1-y2);
        float vpx=v[i]*a;
        float vpx2=vpx*vpx;
        float vpn=v[i]*b;
        float v2=v[i]*v[i];
        float b1=0.5*vpn*vpx,b2=0.5*v2*y;
        float a1=0.5*vpx2*(1+z);
        float d1=0.5*vpx2*(1-z);
        float a2=0.5*v2*(1-z);
        float d2=0.5*v2*(1+z);
        
        float dxa1b1=cc0*(a1*curSigmaX[i]+b1*curSigmaZ[i]);
        float dya1b1=dxa1b1;
        float dza2b2=cc0*(a2*curSigmaX[i]+b2*curSigmaZ[i]);
        float dxb1d1=cc0*(b1*curSigmaX[i]+d1*curSigmaZ[i]);
        float dyb1d1=dxb1d1;
        float dzb2d2=cc0*(b2*curSigmaX[i]+d2*curSigmaZ[i]);
        
        for(int j=1;j<5;j++){
            size_t ij=i+j;
            a=sqrt(1+2.f*eps[ij]);
            b=sqrt(1+2.f*del[ij]);
            y=b/a;
            y2=y*y;
            z=sqrt(1-y2);
            vpx=v[ij]*a;
            vpx2=vpx*vpx;
            vpn=v[ij]*b;
            b1=0.5*vpn*vpx;
            a1=0.5*vpx2*(1+z);
            d1=0.5*vpx2*(1-z);
            
            float temp1=a1*curSigmaX[ij]+b1*curSigmaZ[ij];
            float temp2=b1*curSigmaX[ij]+d1*curSigmaZ[ij];
            
            ij=i-j;
            a=sqrt(1+2.f*eps[ij]);
            b=sqrt(1+2.f*del[ij]);
            y=b/a;
            y2=y*y;
            z=sqrt(1-y2);
            vpx=v[ij]*a;
            vpx2=vpx*vpx;
            vpn=v[ij]*b;
            b1=0.5*vpn*vpx;
            a1=0.5*vpx2*(1+z);
            d1=0.5*vpx2*(1-z);
            
            temp1+=a1*curSigmaX[ij]+b1*curSigmaZ[ij];
            temp2+=b1*curSigmaX[ij]+d1*curSigmaZ[ij];

            dxa1b1+=cc[j]*temp1;
            dxb1d1+=cc[j]*temp2;
            
            ij=i+j*nx;
            a=sqrt(1+2.f*eps[ij]);
            b=sqrt(1+2.f*del[ij]);
            y=b/a;
            y2=y*y;
            z=sqrt(1-y2);
            vpx=v[ij]*a;
            vpx2=vpx*vpx;
            vpn=v[ij]*b;
            b1=0.5*vpn*vpx;
            a1=0.5*vpx2*(1+z);
            d1=0.5*vpx2*(1-z);
            
            temp1=a1*curSigmaX[ij]+b1*curSigmaZ[ij];
            temp2=b1*curSigmaX[ij]+d1*curSigmaZ[ij];
            
            ij=i-j*nx;
            a=sqrt(1+2.f*eps[ij]);
            b=sqrt(1+2.f*del[ij]);
            y=b/a;
            y2=y*y;
            z=sqrt(1-y2);
            vpx=v[ij]*a;
            vpx2=vpx*vpx;
            vpn=v[ij]*b;
            b1=0.5*vpn*vpx;
            a1=0.5*vpx2*(1+z);
            d1=0.5*vpx2*(1-z);
            
            temp1+=a1*curSigmaX[ij]+b1*curSigmaZ[ij];
            temp2+=b1*curSigmaX[ij]+d1*curSigmaZ[ij];

            dya1b1+=cc[j]*temp1;
            dyb1d1+=cc[j]*temp2;
            
            ij=i+j*nxy;
            a=sqrt(1+2.f*eps[ij]);
            b=sqrt(1+2.f*del[ij]);
            y=b/a;
            y2=y*y;
            z=sqrt(1-y2);
            v2=v[ij]*v[ij];
            b2=0.5*v2*y;
            a2=0.5*v2*(1-z);
            d2=0.5*v2*(1+z);
            
            temp1=a2*curSigmaX[ij]+b2*curSigmaZ[ij];
            temp2=b2*curSigmaX[ij]+d2*curSigmaZ[ij];
            
            ij=i-j*nxy;
            a=sqrt(1+2.f*eps[ij]);
            b=sqrt(1+2.f*del[ij]);
            y=b/a;
            y2=y*y;
            z=sqrt(1-y2);
            v2=v[ij]*v[ij];
            b2=0.5*v2*y;
            a2=0.5*v2*(1-z);
            d2=0.5*v2*(1+z);
            
            temp1+=a2*curSigmaX[ij]+b2*curSigmaZ[ij];
            temp2+=b2*curSigmaX[ij]+d2*curSigmaZ[ij];

            dza2b2+=cc[j]*temp1;
            dzb2d2+=cc[j]*temp2;
        }
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(dxa1b1/dx2+dya1b1/dy2+dza2b2/dz2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*(dxb1d1/dx2+dyb1d1/dy2+dzb2d2/dz2)-prevSigmaZ[i];
    }
}

export void forward3(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[]){
    foreach(i=0...n){
        float dSigmaXdx2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+1]+curSigmaX[i-1])
                                         +cc2*(curSigmaX[i+2]+curSigmaX[i-2])
                                         +cc3*(curSigmaX[i+3]+curSigmaX[i-3])
                                         +cc4*(curSigmaX[i+4]+curSigmaX[i-4]))/dx2;
        float dSigmaXdy2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nx]+curSigmaX[i-nx])
                                         +cc2*(curSigmaX[i+2*nx]+curSigmaX[i-2*nx])
                                         +cc3*(curSigmaX[i+3*nx]+curSigmaX[i-3*nx])
                                         +cc4*(curSigmaX[i+4*nx]+curSigmaX[i-4*nx]))/dy2;
        float dSigmaXdz2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nxy]+curSigmaX[i-nxy])
                                         +cc2*(curSigmaX[i+2*nxy]+curSigmaX[i-2*nxy])
                                         +cc3*(curSigmaX[i+3*nxy]+curSigmaX[i-3*nxy])
                                         +cc4*(curSigmaX[i+4*nxy]+curSigmaX[i-4*nxy]))/dz2;
        
        float dSigmaZdx2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+1]+curSigmaZ[i-1])
                                         +cc2*(curSigmaZ[i+2]+curSigmaZ[i-2])
                                         +cc3*(curSigmaZ[i+3]+curSigmaZ[i-3])
                                         +cc4*(curSigmaZ[i+4]+curSigmaZ[i-4]))/dx2;
        float dSigmaZdy2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nx]+curSigmaZ[i-nx])
                                         +cc2*(curSigmaZ[i+2*nx]+curSigmaZ[i-2*nx])
                                         +cc3*(curSigmaZ[i+3*nx]+curSigmaZ[i-3*nx])
                                         +cc4*(curSigmaZ[i+4*nx]+curSigmaZ[i-4*nx]))/dy2;
        float dSigmaZdz2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nxy]+curSigmaZ[i-nxy])
                                         +cc2*(curSigmaZ[i+2*nxy]+curSigmaZ[i-2*nxy])
                                         +cc3*(curSigmaZ[i+3*nxy]+curSigmaZ[i-3*nxy])
                                         +cc4*(curSigmaZ[i+4*nxy]+curSigmaZ[i-4*nxy]))/dz2;
        
        float c33=v[i]*v[i];
        float c11=c33*(1.f+2.f*eps[i]);
        float c13=c33*sqrt(1.f+2.f*del[i]);
        
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(c33*dSigmaXdz2+c13*dSigmaZdz2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*(c13*(dSigmaXdx2+dSigmaXdy2)+c11*(dSigmaZdx2+dSigmaZdy2))-prevSigmaZ[i];
    }
}

export void forward4(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[]){
    foreach(i=0...n){
        float dSigmaXdx2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+1]+curSigmaX[i-1])
                                         +cc2*(curSigmaX[i+2]+curSigmaX[i-2])
                                         +cc3*(curSigmaX[i+3]+curSigmaX[i-3])
                                         +cc4*(curSigmaX[i+4]+curSigmaX[i-4]))/dx2;
        float dSigmaXdy2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nx]+curSigmaX[i-nx])
                                         +cc2*(curSigmaX[i+2*nx]+curSigmaX[i-2*nx])
                                         +cc3*(curSigmaX[i+3*nx]+curSigmaX[i-3*nx])
                                         +cc4*(curSigmaX[i+4*nx]+curSigmaX[i-4*nx]))/dy2;
        
        float dSigmaZdx2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+1]+curSigmaZ[i-1])
                                         +cc2*(curSigmaZ[i+2]+curSigmaZ[i-2])
                                         +cc3*(curSigmaZ[i+3]+curSigmaZ[i-3])
                                         +cc4*(curSigmaZ[i+4]+curSigmaZ[i-4]))/dx2;
        float dSigmaZdy2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nx]+curSigmaZ[i-nx])
                                         +cc2*(curSigmaZ[i+2*nx]+curSigmaZ[i-2*nx])
                                         +cc3*(curSigmaZ[i+3*nx]+curSigmaZ[i-3*nx])
                                         +cc4*(curSigmaZ[i+4*nx]+curSigmaZ[i-4*nx]))/dy2;
        float dSigmaZdz2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nxy]+curSigmaZ[i-nxy])
                                         +cc2*(curSigmaZ[i+2*nxy]+curSigmaZ[i-2*nxy])
                                         +cc3*(curSigmaZ[i+3*nxy]+curSigmaZ[i-3*nxy])
                                         +cc4*(curSigmaZ[i+4*nxy]+curSigmaZ[i-4*nxy]))/dz2;
        
        float v2=v[i]*v[i];
        
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*v2*(del[i]-eps[i])*dSigmaZdz2-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*(v2*(dSigmaXdx2+dSigmaXdy2)+v2*(1.+2.*eps[i])*(dSigmaZdx2+dSigmaZdy2)+v2*dSigmaZdz2)-prevSigmaZ[i];
    }
}

export void forward5(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[]){
    foreach(i=0...n){
        float dSigmaXdx2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+1]+curSigmaX[i-1])
                                         +cc2*(curSigmaX[i+2]+curSigmaX[i-2])
                                         +cc3*(curSigmaX[i+3]+curSigmaX[i-3])
                                         +cc4*(curSigmaX[i+4]+curSigmaX[i-4]))/dx2;
        float dSigmaXdy2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nx]+curSigmaX[i-nx])
                                         +cc2*(curSigmaX[i+2*nx]+curSigmaX[i-2*nx])
                                         +cc3*(curSigmaX[i+3*nx]+curSigmaX[i-3*nx])
                                         +cc4*(curSigmaX[i+4*nx]+curSigmaX[i-4*nx]))/dy2;
        
        float dSigmaZdx2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+1]+curSigmaZ[i-1])
                                         +cc2*(curSigmaZ[i+2]+curSigmaZ[i-2])
                                         +cc3*(curSigmaZ[i+3]+curSigmaZ[i-3])
                                         +cc4*(curSigmaZ[i+4]+curSigmaZ[i-4]))/dx2;
        float dSigmaZdy2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nx]+curSigmaZ[i-nx])
                                         +cc2*(curSigmaZ[i+2*nx]+curSigmaZ[i-2*nx])
                                         +cc3*(curSigmaZ[i+3*nx]+curSigmaZ[i-3*nx])
                                         +cc4*(curSigmaZ[i+4*nx]+curSigmaZ[i-4*nx]))/dy2;
        float dSigmaZdz2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nxy]+curSigmaZ[i-nxy])
                                         +cc2*(curSigmaZ[i+2*nxy]+curSigmaZ[i-2*nxy])
                                         +cc3*(curSigmaZ[i+3*nxy]+curSigmaZ[i-3*nxy])
                                         +cc4*(curSigmaZ[i+4*nxy]+curSigmaZ[i-4*nxy]))/dz2;
        
        float v2=v[i]*v[i];
        
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*v2*dSigmaZdz2-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*(v2*(del[i]-eps[i])*(dSigmaXdx2+dSigmaXdy2)+v2*(1.+2.*eps[i])*(dSigmaZdx2+dSigmaZdy2)+v2*dSigmaZdz2)-prevSigmaZ[i];
    }
}

export void forward6(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[]){
    foreach(i=0...n){
        float dSigmaXdx2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+1]+curSigmaX[i-1])
                                         +cc2*(curSigmaX[i+2]+curSigmaX[i-2])
                                         +cc3*(curSigmaX[i+3]+curSigmaX[i-3])
                                         +cc4*(curSigmaX[i+4]+curSigmaX[i-4]))/dx2;
        float dSigmaXdy2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nx]+curSigmaX[i-nx])
                                         +cc2*(curSigmaX[i+2*nx]+curSigmaX[i-2*nx])
                                         +cc3*(curSigmaX[i+3*nx]+curSigmaX[i-3*nx])
                                         +cc4*(curSigmaX[i+4*nx]+curSigmaX[i-4*nx]))/dy2;
        float dSigmaXdz2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nxy]+curSigmaX[i-nxy])
                                         +cc2*(curSigmaX[i+2*nxy]+curSigmaX[i-2*nxy])
                                         +cc3*(curSigmaX[i+3*nxy]+curSigmaX[i-3*nxy])
                                         +cc4*(curSigmaX[i+4*nxy]+curSigmaX[i-4*nxy]))/dz2;
        
        float dSigmaZdz2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nxy]+curSigmaZ[i-nxy])
                                         +cc2*(curSigmaZ[i+2*nxy]+curSigmaZ[i-2*nxy])
                                         +cc3*(curSigmaZ[i+3*nxy]+curSigmaZ[i-3*nxy])
                                         +cc4*(curSigmaZ[i+4*nxy]+curSigmaZ[i-4*nxy]))/dz2;
        
        float v2=v[i]*v[i];
        
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(v2*(1.+2.*eps[i])*(dSigmaXdx2+dSigmaXdy2)+v2*dSigmaXdz2+v2*dSigmaZdz2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*v2*(del[i]-eps[i])*(dSigmaXdx2+dSigmaXdy2)-prevSigmaZ[i];
    }
}

export void forward7(uniform int n,uniform int nx,uniform int nxy,uniform float dx2,uniform float dy2,uniform float dz2,uniform float dt2,uniform float v[],uniform float eps[],uniform float del[],uniform float prevSigmaX[],uniform float curSigmaX[],uniform float prevSigmaZ[],uniform float curSigmaZ[]){
    foreach(i=0...n){
        float dSigmaXdx2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+1]+curSigmaX[i-1])
                                         +cc2*(curSigmaX[i+2]+curSigmaX[i-2])
                                         +cc3*(curSigmaX[i+3]+curSigmaX[i-3])
                                         +cc4*(curSigmaX[i+4]+curSigmaX[i-4]))/dx2;
        float dSigmaXdy2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nx]+curSigmaX[i-nx])
                                         +cc2*(curSigmaX[i+2*nx]+curSigmaX[i-2*nx])
                                         +cc3*(curSigmaX[i+3*nx]+curSigmaX[i-3*nx])
                                         +cc4*(curSigmaX[i+4*nx]+curSigmaX[i-4*nx]))/dy2;
        float dSigmaXdz2=(cc0*curSigmaX[i]+cc1*(curSigmaX[i+nxy]+curSigmaX[i-nxy])
                                         +cc2*(curSigmaX[i+2*nxy]+curSigmaX[i-2*nxy])
                                         +cc3*(curSigmaX[i+3*nxy]+curSigmaX[i-3*nxy])
                                         +cc4*(curSigmaX[i+4*nxy]+curSigmaX[i-4*nxy]))/dz2;
        
        float dSigmaZdz2=(cc0*curSigmaZ[i]+cc1*(curSigmaZ[i+nxy]+curSigmaZ[i-nxy])
                                         +cc2*(curSigmaZ[i+2*nxy]+curSigmaZ[i-2*nxy])
                                         +cc3*(curSigmaZ[i+3*nxy]+curSigmaZ[i-3*nxy])
                                         +cc4*(curSigmaZ[i+4*nxy]+curSigmaZ[i-4*nxy]))/dz2;
        
        float v2=v[i]*v[i];
        
        prevSigmaX[i]=2.f*curSigmaX[i]+dt2*(v2*(1.+2.*eps[i])*(dSigmaXdx2+dSigmaXdy2)+v2*dSigmaXdz2+v2*(del[i]-eps[i])*dSigmaZdz2)-prevSigmaX[i];
        prevSigmaZ[i]=2.f*curSigmaZ[i]+dt2*v2*(dSigmaXdx2+dSigmaXdy2)-prevSigmaZ[i];
    }
}



