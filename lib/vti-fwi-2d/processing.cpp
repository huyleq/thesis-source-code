#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cfloat>
#include "myio.h"
#include "mylib.h"

using namespace std;

void aniso_nmo(float *dataout,const float *datain,const float *vnmo,const float *eta,int nt,float ot,float dt,float *offset,int noffset){
    float *vnmo2=new float[nt];
    float *tau2=new float[nt];

    for(int it=0;it<nt;it++){
        vnmo2[it]=vnmo[it]*vnmo[it];
        float tau=ot+it*dt;
        tau2[it]=tau*tau;
        dataout[it]=datain[it];
    }

    for(int ioffset=1;ioffset<noffset;ioffset++){
        double offset2=offset[ioffset]*offset[ioffset];
        double offset4=offset2*offset2;
        for(int itau=0;itau<nt;itau++){
            double a4=-2.*eta[itau]/(vnmo2[itau]*(tau2[itau]*vnmo2[itau]+(1.+2.*eta[itau])*offset2));
            double temp=tau2[itau]+offset2/vnmo2[itau]+a4*offset4;
            double t=sqrt(temp)+FLT_EPSILON;
            double tau=itau*dt+ot;
            double wt=tau/t/sqrt(t);
            double f=(t-ot)/dt;
            int it=f;
            if(it>=0  && it<nt-1){
                double fx=f-it;
                dataout[itau+ioffset*nt]=wt*((1.-fx)*datain[it+ioffset*nt]+fx*datain[it+1+ioffset*nt]);
            }
        }
    }
    delete []vnmo2;delete []tau2;
    return;
}

void aniso_nmo_stack(float *dataout,const float *datain,const float *vnmo,const float *eta,int nt,float ot,float dt,float *offset,int noffset){
    float *vnmo2=new float[nt];

    for(int it=0;it<nt;it++) vnmo2[it]=vnmo[it]*vnmo[it];
    
    for(int ioffset=0;ioffset<noffset;ioffset++){
        float offset2=offset[ioffset]*offset[ioffset];
        float offset4=offset2*offset2;
        for(int itau=0;itau<nt;itau++){
            float tau=ot+itau*dt;
            float tau2=tau*tau;
            float a4=-2.*eta[itau]/(vnmo2[itau]*(tau2*vnmo2[itau]+(1.+2.*eta[itau])*offset2));
            float t=sqrt(tau2+offset2/vnmo2[itau]+a4*offset4);
            int it=(t-ot)/dt;
            if(it>=0  && it<nt-1){
                float f=t-(ot+it*dt);
                dataout[itau]+=(1.-f)*datain[it+ioffset*nt]+f*datain[it+1+ioffset*nt];
            }
        }
    }
    delete []vnmo2;
    return;
}

void vint2vrms(float *&vrms,int &nt,float ot,float dt,float *vint,int nz,float dz){
    float *vrms0=new float[nz];
    vrms0[0]=0.;
    float *tau=new float[nz];
    tau[0]=0.;
    float top=0.,bot=0.;

    for(int iz=0;iz<nz;iz++){
        float dtau=2.*dz/vint[iz];
        top+=vint[iz]*vint[iz]*dtau;
        bot+=dtau;
        vrms0[iz]=sqrt(top/bot);
        tau[iz]=bot;
    }
    
    nt=(bot-ot)/dt+1;
    vrms=new float[nt]();

    for(int iz=0;iz<nz;iz++){
        int it=(tau[iz]-ot)/dt;
        if(it>=0 && it<nt-1){
            float f=tau[iz]-(ot+it*dt);
            vrms[it]+=(1.-f)*vrms0[iz];
            vrms[it+1]+=f*vrms0[iz];
        }
    }

    delete []vrms0;delete []tau;
    return;
}
