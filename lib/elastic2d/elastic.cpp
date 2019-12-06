#include <cmath>

#include "elastic.h"

void vepsdel2cij(float *c11,float *c13,float *c33,float *c44,const float *vp,const float *vs,const float *eps,const float *del,const float *rho,int nxz){
    //no interpolation
    #pragma omp parallel for
    for(int i=0;i<nxz;i++){
        float t33=rho[i]*vp[i]*vp[i];
        float t44=rho[i]*vs[i]*vs[i];
        c33[i]=t33;
        c44[i]=t44;
        c11[i]=t33*(1.f+2.f*eps[i]);
        c13[i]=sqrt((t33-t44)*((1.f+2.f*del[i])*t33-t44))-t44;
    }
    return;
}

void inject_source(float *sx,float *sz,float dtsou,int slocxz){
    sx[slocxz]+=dtsou;
    sz[slocxz]+=dtsou;
    return;
}

void inject_dipole_source(float *sx,float *sz,float dtsou,int slocxz,int nx){
    sx[slocxz]+=dtsou;
    sz[slocxz]+=dtsou;
    sx[slocxz+nx]-=dtsou;
    sz[slocxz+nx]-=dtsou;
    return;
}

void update_vel(float *vx,float *vz,const float *sx,const float *sz,const float *sxz,const float *buoy,int nx,int nz,const float *coeff){
    //coeff=c*dt/dx
    #pragma omp parallel for 
    for(int iz=4;iz<nz-4;iz++){
        #pragma omp simd
        for(int ix=4;ix<nx-4;ix++){
            float dtdsxdx=coeff[0]*(sx[ix+iz*nx]-sx[ix-1+iz*nx])
                         +coeff[1]*(sx[ix+1+iz*nx]-sx[ix-2+iz*nx])
                         +coeff[2]*(sx[ix+2+iz*nx]-sx[ix-3+iz*nx])
                         +coeff[3]*(sx[ix+3+iz*nx]-sx[ix-4+iz*nx]);
            float dtdsxzdz=coeff[0]*(sxz[ix+iz*nx]-sxz[ix+(iz-1)*nx])
                          +coeff[1]*(sxz[ix+(iz+1)*nx]-sxz[ix+(iz-2)*nx])
                          +coeff[2]*(sxz[ix+(iz+2)*nx]-sxz[ix+(iz-3)*nx])
                          +coeff[3]*(sxz[ix+(iz+3)*nx]-sxz[ix+(iz-4)*nx]);
            vx[ix+iz*nx]+=buoy[ix+iz*nx]*(dtdsxdx+dtdsxzdz);
            float dtdsxzdx=coeff[0]*(sxz[ix+1+iz*nx]-sxz[ix+iz*nx])
                          +coeff[1]*(sxz[ix+2+iz*nx]-sxz[ix-1+iz*nx])
                          +coeff[2]*(sxz[ix+3+iz*nx]-sxz[ix-2+iz*nx])
                          +coeff[3]*(sxz[ix+4+iz*nx]-sxz[ix-3+iz*nx]);
            float dtdszdz=coeff[0]*(sz[ix+(iz+1)*nx]-sz[ix+iz*nx])
                         +coeff[1]*(sz[ix+(iz+2)*nx]-sz[ix+(iz-1)*nx])
                         +coeff[2]*(sz[ix+(iz+3)*nx]-sz[ix+(iz-2)*nx])
                         +coeff[3]*(sz[ix+(iz+4)*nx]-sz[ix+(iz-3)*nx]);
            vz[ix+iz*nx]+=buoy[ix+iz*nx]*(dtdsxzdx+dtdszdz);
        }
    }
    return;
}

void update_stress(float *sx,float *sz,float *sxz,const float *vx,const float *vz,const float *c11,const float *c13,const float *c33,const float *c44,int nx,int nz,const float *coeff){
    #pragma omp parallel for 
    for(int iz=4;iz<nz-4;iz++){
        #pragma omp simd
        for(int ix=4;ix<nx-4;ix++){
            float dtdvxdx=coeff[0]*(vx[ix+1+iz*nx]-vx[ix+iz*nx])
                         +coeff[1]*(vx[ix+2+iz*nx]-vx[ix-1+iz*nx])
                         +coeff[2]*(vx[ix+3+iz*nx]-vx[ix-2+iz*nx])
                         +coeff[3]*(vx[ix+4+iz*nx]-vx[ix-3+iz*nx]);
            float dtdvzdz=coeff[0]*(vz[ix+iz*nx]-vz[ix+(iz-1)*nx])
                         +coeff[1]*(vz[ix+(iz+1)*nx]-vz[ix+(iz-2)*nx])
                         +coeff[2]*(vz[ix+(iz+2)*nx]-vz[ix+(iz-3)*nx])
                         +coeff[3]*(vz[ix+(iz+3)*nx]-vz[ix+(iz-4)*nx]);
            sx[ix+iz*nx]+=c11[ix+iz*nx]*dtdvxdx+c13[ix+iz*nx]*dtdvzdz;
            sz[ix+iz*nx]+=c13[ix+iz*nx]*dtdvxdx+c33[ix+iz*nx]*dtdvzdz;
            float dtdvxdz=coeff[0]*(vx[ix+(iz+1)*nx]-vx[ix+iz*nx])
                         +coeff[1]*(vx[ix+(iz+2)*nx]-vx[ix+(iz-1)*nx])
                         +coeff[2]*(vx[ix+(iz+3)*nx]-vx[ix+(iz-2)*nx])
                         +coeff[3]*(vx[ix+(iz+4)*nx]-vx[ix+(iz-3)*nx]);
            float dtdvzdx=coeff[0]*(vz[ix+iz*nx]-vz[ix-1+iz*nx])
                         +coeff[1]*(vz[ix+1+iz*nx]-vz[ix-2+iz*nx])
                         +coeff[2]*(vz[ix+2+iz*nx]-vz[ix-3+iz*nx])
                         +coeff[3]*(vz[ix+3+iz*nx]-vz[ix-4+iz*nx]);
            sxz[ix+iz*nx]+=c44[ix+iz*nx]*(dtdvxdz+dtdvzdx);
        }
    }
    return;
}

void update_stress_memory(float *sx,float *sz,float *sxz,float **xix,float **xiz,float **xixz,const float *vx,const float *vz,const float *c11,const float *c13,const float *c33,const float *c44,const float *dtgbarQinv,const float *g,const float *w,int nx,int nz,const float *coeff,float dt){
    #pragma omp parallel for
    for(int iz=4;iz<nz-4;iz++){
        #pragma omp simd
        for(int ix=4;ix<nx-4;ix++){
            float dtdvxdx=coeff[0]*(vx[ix+1+iz*nx]-vx[ix+iz*nx])
                         +coeff[1]*(vx[ix+2+iz*nx]-vx[ix-1+iz*nx])
                         +coeff[2]*(vx[ix+3+iz*nx]-vx[ix-2+iz*nx])
                         +coeff[3]*(vx[ix+4+iz*nx]-vx[ix-3+iz*nx]);
            float dtdvzdz=coeff[0]*(vz[ix+iz*nx]-vz[ix+(iz-1)*nx])
                         +coeff[1]*(vz[ix+(iz+1)*nx]-vz[ix+(iz-2)*nx])
                         +coeff[2]*(vz[ix+(iz+2)*nx]-vz[ix+(iz-3)*nx])
                         +coeff[3]*(vz[ix+(iz+3)*nx]-vz[ix+(iz-4)*nx]);
            float dtdvxdz=coeff[0]*(vx[ix+(iz+1)*nx]-vx[ix+iz*nx])
                         +coeff[1]*(vx[ix+(iz+2)*nx]-vx[ix+(iz-1)*nx])
                         +coeff[2]*(vx[ix+(iz+3)*nx]-vx[ix+(iz-2)*nx])
                         +coeff[3]*(vx[ix+(iz+4)*nx]-vx[ix+(iz-3)*nx]);
            float dtdvzdx=coeff[0]*(vz[ix+iz*nx]-vz[ix-1+iz*nx])
                         +coeff[1]*(vz[ix+1+iz*nx]-vz[ix-2+iz*nx])
                         +coeff[2]*(vz[ix+2+iz*nx]-vz[ix-3+iz*nx])
                         +coeff[3]*(vz[ix+3+iz*nx]-vz[ix-4+iz*nx]);
            float tmpx=0.f,tmpz=0.f,tmpxz=0.f;
            for(int i=0;i<L;i++){
                tmpx+=g[i]*xix[i][ix+iz*nx];
                tmpz+=g[i]*xiz[i][ix+iz*nx];
                tmpxz+=g[i]*xixz[i][ix+iz*nx];

            }
            sx[ix+iz*nx]+=c11[ix+iz*nx]*dtdvxdx+c13[ix+iz*nx]*dtdvzdz-dtgbarQinv[ix+iz*nx]*(c11[ix+iz*nx]*tmpx+c13[ix+iz*nx]*tmpz);
            sz[ix+iz*nx]+=c13[ix+iz*nx]*dtdvxdx+c33[ix+iz*nx]*dtdvzdz-dtgbarQinv[ix+iz*nx]*(c13[ix+iz*nx]*tmpx+c33[ix+iz*nx]*tmpz);
            sxz[ix+iz*nx]+=c44[ix+iz*nx]*(dtdvxdz+dtdvzdx)-dtgbarQinv[ix+iz*nx]*c44[ix+iz*nx]*tmpxz;
            for(int i=0;i<L;i++){
                xix[i][ix+iz*nx]=xix[i][ix+iz*nx]+w[i]*(dtdvxdx-dt*xix[i][ix+iz*nx]);
                xiz[i][ix+iz*nx]=xiz[i][ix+iz*nx]+w[i]*(dtdvzdz-dt*xiz[i][ix+iz*nx]);
                xixz[i][ix+iz*nx]=xixz[i][ix+iz*nx]+w[i]*(dtdvxdz+dtdvzdx-dt*xixz[i][ix+iz*nx]);
            }
        }
    }
    return;
}

void init_abc(float *taper,int npad){
    #pragma omp parallel for 
    for(int i=0;i<npad;++i){
     taper[i]=DAMPER+(1.-DAMPER)*cos(pi*(float)(npad-1-i)/npad);
    }
    return;
}

void abc(float *u,const float *taper,int nx,int nz,int npad){
    #pragma omp parallel for 
    for(int iz=0;iz<nz;iz++){
        #pragma omp simd
        for(int ix=0;ix<npad;ix++){
            u[ix+iz*nx]*=taper[ix]; //left boundary
            u[nx-1-ix+iz*nx]*=taper[ix]; //right boundary
        }
    }
    #pragma omp parallel for 
    for(int iz=0;iz<npad;iz++){
        #pragma omp simd
        for(int ix=0;ix<nx;ix++){
            u[ix+iz*nx]*=taper[iz]; //left boundary
            u[ix+(nz-1-iz)*nx]*=taper[iz]; //right boundary
        }
    }
    return;
}

void record_pressure(float *pdata,const int *rloc,int nr,const float *sx,const float *sz){
    #pragma omp parallel for
    for(int ir=0;ir<nr;ir++) pdata[ir]=0.5*(sx[rloc[ir]]+sz[rloc[ir]]);
    return;
}

void record_vel(float *vxdata,float *vzdata,const int *rloc,int nr,const float *vx,const float *vz){
    //no interpolation
    #pragma omp parallel for
    for(int ir=0;ir<nr;ir++){
        vxdata[ir]=vx[rloc[ir]];
        vzdata[ir]=vz[rloc[ir]];
    }
    return;
}
