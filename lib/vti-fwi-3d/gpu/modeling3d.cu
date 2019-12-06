#include <iostream>
#include <cstring>
#include <chrono>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"
#include "conversions.h"

using namespace std;

int main(int argc,char **argv){
 myio_init(argc,argv);
 
 int nx,ny,nz,npad,nt;
 float ox,oy,oz,ot,dx,dy,dz,dt;
 
 from_header("v","n1",nx,"o1",ox,"d1",dx);
 from_header("v","n2",ny,"o2",oy,"d2",dy);
 from_header("v","n3",nz,"o3",oz,"d3",dz);
 get_param("npad",npad);
 get_param("nt",nt,"ot",ot,"dt",dt);
 
 long long nxy=nx*ny;
 long long nxyz=nxy*nz;
 
 float *wavelet=new float[nt]();
 read("wavelet",wavelet,nt);

 float samplingRate;
 get_param("samplingRate",samplingRate);
 int samplingTimeStep=std::round(samplingRate/dt);
// int nnt=(nt-1)/samplingTimeStep+1;

 float *v=new float[nxyz];
 read("v",v,nxyz);
 float *eps=new float[nxyz]();
 read("eps",eps,nxyz);
 float *del=new float[nxyz]();
 read("del",del,nxyz);

 float *c11=new float[nxyz];
 float *c13=new float[nxyz];
 float *c33=new float[nxyz];
 VEpsDel2Cij(c11,c13,c33,v,eps,del,1.,1.,1.,nxyz);

 float soulocX,soulocY,soulocZ;
 get_param("soulocX",soulocX,"soulocY",soulocY,"soulocZ",soulocZ);

 fprintf(stderr,"modeling\n");

 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 modeling3d_f(soulocX,soulocY,soulocZ,wavelet,c11,c13,c33,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt);

 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 delete []wavelet;
 delete []v;delete []eps;delete []del;
 delete []c11;delete []c13;delete []c33;

 myio_close();
 return 0;
}

