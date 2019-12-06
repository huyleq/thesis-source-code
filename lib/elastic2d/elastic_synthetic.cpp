#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <chrono>

#include "myio.h"
#include "mylib.h"

#include "elastic.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);

    int nx,nz,nt,npad,nxz;
    float ox,oz,dxz,dt,sampling_rate;

    float *c11,*c13,*c33,*c44,*buoy;

    string parameter=get_s("parameter");
    if(parameter.compare("vepsdel")==0){
        cout<<"input are vp vs eps del rho"<<endl;
        from_header("vp","n1",nx,"o1",ox,"d1",dxz);
        from_header("vp","n2",nz,"o2",oz);
        nxz=nx*nz;
        float *vp=new float[nxz]; read("vp",vp,nxz);
        float *vs=new float[nxz]; read("vs",vs,nxz);
        float *eps=new float[nxz]; read("eps",eps,nxz);
        float *del=new float[nxz]; read("del",del,nxz);
        buoy=new float[nxz]; read("rho",buoy,nxz);
        c11=new float[nxz];
        c13=new float[nxz];
        c33=new float[nxz];
        c44=new float[nxz];
        vepsdel2cij(c11,c13,c33,c44,vp,vs,eps,del,buoy,nxz);
        delete []vp;delete []vs;delete []eps;delete []del;
    }
    else{
        cout<<"input are cij"<<endl;
        from_header("c11","n1",nx,"o1",ox,"d1",dxz);
        from_header("c11","n2",nz,"o2",oz);
        nxz=nx*nz;
        c11=new float[nxz]; read("c11",c11,nxz);
        c13=new float[nxz]; read("c13",c13,nxz);
        c33=new float[nxz]; read("c33",c33,nxz);
        c44=new float[nxz]; read("c44",c44,nxz);
        buoy=new float[nxz]; read("rho",buoy,nxz);
    }
    
    reciprocal(buoy,buoy,nxz);
    
    get_param("nt",nt,"npad",npad);
    get_param("dt",dt,"sampling_rate",sampling_rate);
    int ratio=round(sampling_rate/dt);
    int nnt=(nt-1)/ratio+1;

    float *wavelet=new float[nt]; read("wavelet",wavelet,nt);

    int ns;
    from_header("souloc","n2",ns);
    fprintf(stderr,"#shots %d\n",ns);
    float *souloc=new float[ns*4];
    read("souloc",souloc,ns*4);
    int *sloc=new int[ns*3];
    #pragma omp parallel for num_threads(16)
    for(int is=0;is<ns;is++){
        int ix=(souloc[is*4+0]-ox)/dxz+0.5;
        int iz=(souloc[is*4+1]-oz)/dxz+0.5;
        sloc[is*3]=ix+iz*nx;
        sloc[is*3+1]=souloc[is*4+2];
        sloc[is*3+2]=souloc[is*4+3];
    }
    delete []souloc;

    int nr;
    from_header("recloc","n2",nr);
    fprintf(stderr,"#recs %d\n",nr);
    float *recloc=new float[nr*2];
    read("recloc",recloc,nr*2);
    int *rloc=new int[nr];
    #pragma omp parallel for num_threads(16)
    for(int ir=0;ir<nr;ir++){
        int ix=(recloc[ir*2+0]-ox)/dxz+0.5;
        int iz=(recloc[ir*2+1]-oz)/dxz+0.5;
        rloc[ir]=ix+iz*nx;
    }
    delete []recloc;

    float *pdata,*vxdata,*vzdata;
    
    if(get_s("pdata")==" ") pdata=nullptr;
    else pdata=new float[nr*nnt];
    
    if(get_s("vxdata")==" "){
        vxdata=nullptr;
        vzdata=nullptr;
    }
    else{
        vxdata=new float[nr*nnt];
        vzdata=new float[nr*nnt];
    }

    chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
    elastic_synthetic_f(pdata,vxdata,vzdata,wavelet,sloc,ns,rloc,nr,c11,c13,c33,c44,buoy,nx,nz,nt,npad,dxz,dt,sampling_rate);

    chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
    chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
    cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
    if(pdata){
        write("pdata",pdata,nr*nnt);
        to_header("pdata","n1",nr,"o1",0,"d1",1.);
        to_header("pdata","n2",nnt,"o2",0,"d2",sampling_rate);
        delete []pdata;
    }

    if(vxdata){
        write("vxdata",vxdata,nr*nnt);
        to_header("vxdata","n1",nr,"o1",0,"d1",1.);
        to_header("vxdata","n2",nnt,"o2",0,"d2",sampling_rate);
        write("vzdata",vzdata,nr*nnt);
        to_header("vzdata","n1",nr,"o1",0,"d1",1.);
        to_header("vzdata","n2",nnt,"o2",0,"d2",sampling_rate);
        delete []vxdata;delete []vzdata;
    }

    delete []c11;delete []c13;delete []c33;delete []c44;delete []buoy;
    delete []wavelet;
    delete []sloc;delete []rloc;

    myio_close();
    return 0;
}
