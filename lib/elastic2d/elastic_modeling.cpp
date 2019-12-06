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
    size_t nxzt=nxz*nnt;

    float *wavelet=new float[nt]; read("wavelet",wavelet,nt);

    float *sx_wfld=new float[nxzt];
    float *sz_wfld=new float[nxzt];
    float *sxz_wfld=new float[nxzt];
    float *vx_wfld=new float[nxzt];
    float *vz_wfld=new float[nxzt];

    float slocx,slocz;
    get_param("slocx",slocx,"slocz",slocz);
    int islocxz=(slocx-ox)/dxz+(slocz-oz)/dxz*nx;

    chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
    elastic_modeling_f(sx_wfld,sz_wfld,sxz_wfld,vx_wfld,vz_wfld,wavelet,c11,c13,c33,c44,buoy,nx,nz,nt,npad,dxz,dt,islocxz,sampling_rate);

    chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
    chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
    cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
    write("sx_wfld",sx_wfld,nxzt);
    to_header("sx_wfld","n1",nx,"o1",ox,"d1",dxz);
    to_header("sx_wfld","n2",nz,"o2",oz,"d2",dxz);
    to_header("sx_wfld","n3",nnt,"o3",0.,"d3",sampling_rate);

    write("sz_wfld",sz_wfld,nxzt);
    to_header("sz_wfld","n1",nx,"o1",ox,"d1",dxz);
    to_header("sz_wfld","n2",nz,"o2",oz,"d2",dxz);
    to_header("sz_wfld","n3",nnt,"o3",0.,"d3",sampling_rate);

    write("sxz_wfld",sxz_wfld,nxzt);
    to_header("sxz_wfld","n1",nx,"o1",ox-0.5*dxz,"d1",dxz);
    to_header("sxz_wfld","n2",nz,"o2",oz+0.5*dxz,"d2",dxz);
    to_header("sxz_wfld","n3",nnt,"o3",0.,"d3",sampling_rate);

    write("vx_wfld",vx_wfld,nxzt);
    to_header("vx_wfld","n1",nx,"o1",ox-0.5*dxz,"d1",dxz);
    to_header("vx_wfld","n2",nz,"o2",oz,"d2",dxz);
    to_header("vx_wfld","n3",nnt,"o3",0.,"d3",sampling_rate);

    write("vz_wfld",vz_wfld,nxzt);
    to_header("vz_wfld","n1",nx,"o1",ox,"d1",dxz);
    to_header("vz_wfld","n2",nz,"o2",oz+0.5*dxz,"d2",dxz);
    to_header("vz_wfld","n3",nnt,"o3",0.,"d3",sampling_rate);

    delete []c11;delete []c13;delete []c33;delete []c44;delete []buoy;
    delete []wavelet;
    delete []sx_wfld;delete []sz_wfld;delete []sxz_wfld;
    delete []vx_wfld;delete []vz_wfld;

    myio_close();
    return 0;
}
