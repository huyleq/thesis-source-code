#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "myio.h"
#include "mylib.h"
#include "processing.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);

    int nx,nz;
    float ox,oz,dx,dz;
    get_param("nx",nx,"ox",ox,"dx",dx);
    get_param("nz",nz,"oz",oz,"dz",dz);

    float *v=new float[nx*nz];
    read("v",v,nx*nz);
    float *v1=new float[nz];
    for(int iz=0;iz<nz;iz++) v1[iz]=v[iz*nx];

//    float *vrms1;
//    int nt;
//    float ot=0.,dt=0.004;
//
//    vint2vrms(vrms1,nt,ot,dt,v1,nz,dz);
//    write("vrms",vrms1,nt);
//    to_header("vrms","n1",nt,"o1",ot,"d1",dt);

    int ns;
    from_header("souloc","n2",ns);
    float *souloc=new float[ns*4]();
    read("souloc",souloc,ns*4);
    
    int nr;
    from_header("recloc","n2",nr);
    float *recloc=new float[nr*2]();
    read("recloc",recloc,nr*2);
    
    float rate;
    int nnt;
    from_header("data","n2",nnt,"d2",rate);
    float *d0=new float[nr*nnt]();
    read("data",d0,nr*nnt);

    float minv,maxv,dv;
    get_param("minv",minv,"maxv",maxv,"dv",dv);
    int nv=(maxv-minv)/dv+1;

    float mineta,maxeta,deta;
    get_param("mineta",mineta,"maxeta",maxeta,"deta",deta);
    int neta=(maxeta-mineta)/deta+1;

    float *dataout=new float[nnt*nr*nv*neta]();
    float *vrms=new float[nnt];
    float *eta=new float[nnt]();
   
    for(int is=0;is<ns;is++){
        int nr1=souloc[is*4+2];
        int ir1=souloc[is*4+3];
        float *offset=new float[nr1];
        float *datain=new float[nnt*nr1]();
        for(int ir=0;ir<nr1;ir++){
            offset[ir]=souloc[is*4]-recloc[(ir1+ir)*2];
            for(int it=0;it<nnt;it++){
                datain[it+ir*nnt]=d0[ir1+ir+it*nr];
            }
        }
        for(int iv=0;iv<nv;iv++){
            set(vrms,minv+iv*dv,nnt);
            for(int ieta=0;ieta<neta;ieta++){
                set(eta,mineta+ieta*deta,nnt);
                float *dataout0=new float[nnt*nr1]();
//                aniso_nmo_stack(dataout+iv*nnt+ieta*nnt*nv,datain,vrms,eta,nnt,0.,rate,offset,nr1);
                aniso_nmo(dataout0,datain,vrms,eta,nnt,0.,rate,offset,nr1);
                memcpy(dataout+iv*nnt*nr+ieta*nnt*nr*nv,dataout0,nnt*nr1*sizeof(float));
                delete []dataout0;
            }
        }
        delete []offset; delete []datain;
    }
    
    int halfwindow;
    get_param("halfwindow",halfwindow);

    float *sem=new float[nnt*nv*neta]();

    for(int ieta=0;ieta<neta;ieta++){
        for(int iv=0;iv<nv;iv++){
            for(int it=halfwindow;it<nnt-halfwindow;it++){
                double a=0.,b=0.;
                for(int i=it-halfwindow;i<=it+halfwindow;i++){
                    double temp=0.;
                    for(int ir=0;ir<nr;ir++){
                        temp+=dataout[i+ir*nnt+iv*nnt*nr+ieta*nnt*nr*nv];
                        b+=dataout[i+ir*nnt+iv*nnt*nr+ieta*nnt*nr*nv]*dataout[i+ir*nnt+iv*nnt*nr+ieta*nnt*nr*nv];
                    }
                    a+=temp*temp;
                }
                sem[it+iv*nnt+ieta*nnt*nv]=a/b/nr;
            }
        }
    }
    
    write("dataout",dataout,nnt*nr*nv*neta);
    to_header("dataout","n1",nnt,"o1",0.,"d1",rate);
    to_header("dataout","n2",nr,"o2",0.,"d2",1.);
    to_header("dataout","n3",nv,"o3",minv,"d3",dv);
    to_header("dataout","n4",neta,"o4",mineta,"d4",deta);

    write("semblance",sem,nnt*nv*neta);
    to_header("semblance","n1",nnt,"o1",0.,"d1",rate);
    to_header("semblance","n2",nv,"o2",minv,"d2",dv);
    to_header("semblance","n3",neta,"o3",mineta,"d3",deta);

    delete []v;delete []v1;delete []dataout;//delete []vrms1;
    delete []sem;

    myio_close();
    return 0;
}
