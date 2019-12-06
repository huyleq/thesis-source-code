#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "myio.h"

#include "elastic.h"

void viscoelastic_synthetic_f(float *pdata,float *vxdata,float *vzdata,const float *wavelet,const int *sloc,int ns,const int *rloc,int nr,const float *c11,const float *c13,const float *c33,const float *c44,const float *buoy,const float *dtgbarQinv,const float *g,const float *w,int nx,int nz,int nt,int npad, float dxz,float dt,float sampling_rate){

    float *taper=new float[npad];
    init_abc(taper,npad);

    int nxz=nx*nz,nbytexz=nxz*sizeof(float);
    int ratio=round(sampling_rate/dt);
    
    float dtdxz=dt/dxz;
    float coeff[4];
    coeff[0]=dtdxz*c0;
    coeff[1]=dtdxz*c1;
    coeff[2]=dtdxz*c2;
    coeff[3]=dtdxz*c3;

    float *sx=new float[nxz];
    float *sz=new float[nxz];
    float *sxz=new float[nxz];
    float *xix[L],*xiz[L],*xixz[L];
    for(int i=0;i<L;i++){
        xix[i]=new float[nxz];
        xiz[i]=new float[nxz];
        xixz[i]=new float[nxz];
    }
    float *vx=new float[nxz];
    float *vz=new float[nxz];

    for(int is=0;is<ns;is++){
        fprintf(stderr,"shot # %d at %d\n",is,sloc[is*3]);
        
        memset(sx,0,nbytexz);
        memset(sz,0,nbytexz);
        memset(sxz,0,nbytexz);
        for(int i=0;i<L;i++){
            memset(xix[i],0,nbytexz);
            memset(xiz[i],0,nbytexz);
            memset(xixz[i],0,nbytexz);
        }
        memset(vx,0,nbytexz);
        memset(vz,0,nbytexz);

        for(int it=1;it<nt;it++){
            //update stress at it. need source at it-1/2
            update_stress_memory(sx,sz,sxz,xix,xiz,xixz,vx,vz,c11,c13,c33,c44,dtgbarQinv,g,w,nx,nz,coeff,dt);
            
            float dtsou=dt*0.5*(wavelet[it]+wavelet[it-1]);
            inject_source(sx,sz,dtsou,sloc[is*3]);
            
            update_vel(vx,vz,sx,sz,sxz,buoy,nx,nz,coeff);
            
            abc(sx,taper,nx,nz,npad);
            abc(sz,taper,nx,nz,npad);
            abc(sxz,taper,nx,nz,npad);
            for(int i=0;i<L;i++){
                abc(xix[i],taper,nx,nz,npad);
                abc(xiz[i],taper,nx,nz,npad);
                abc(xixz[i],taper,nx,nz,npad);
            }
            abc(vx,taper,nx,nz,npad);
            abc(vz,taper,nx,nz,npad);
            
            if(it%ratio==0){
                size_t tmp=(it/ratio)*nr+sloc[is*3+2];
                if(pdata) record_pressure(pdata+tmp,rloc+sloc[is*3+2],sloc[is*3+1],sx,sz);
                if(vxdata) record_vel(vxdata+tmp,vzdata+tmp,rloc+sloc[is*3+2],sloc[is*3+1],vx,vz);
            }
        }
    }

    delete[]sx;delete []sz;delete []sxz;
    for(int i=0;i<L;i++){
        delete []xix[i];delete []xiz[i];delete []xixz[i];
    }
    delete []vx;delete []vz;
    delete []taper;
    return;
}
