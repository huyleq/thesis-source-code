#include <iostream>
#include <cstring>
#include <chrono>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"

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
 
 float *v=new float[nxyz];
 read("v",v,nxyz);
 float *eps=new float[nxyz]();
 read("eps",eps,nxyz);
 float *del=new float[nxyz]();
 read("del",del,nxyz);

 int ns,nr;
 from_header("souloc","n2",ns);
 float *souloc=new float[5*ns];
 read("souloc",souloc,5*ns);

 from_header("recloc","n2",nr);
 float *recloc=new float[3*nr];
 read("recloc",recloc,3*nr);
 
 float *image=new float[nxyz*(2*NLAG+1)]();
 
 chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
 
 odcig3d_f(image,souloc,ns,recloc,wavelet,v,eps,del,nx,ny,nz,nt,npad,ox,oy,oz,ot,dx,dy,dz,dt,samplingRate);

 chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
 chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
 cout<<"total time "<<time.count()/60.<<" minutes"<<endl;
 
 write("image",image,nxyz*(2*NLAG+1));
 to_header("image","n1",nx,"o1",ox,"d1",dx);
 to_header("image","n2",ny,"o2",oy,"d2",dy);
 to_header("image","n3",nz,"o3",oz,"d3",dz);
 to_header("image","n4",2*NLAG+1,"o4",-NLAG*dx,"d4",dx);

 delete []wavelet;
 delete []v;delete []eps;delete []del;
 delete []image;

 delete []souloc;
 delete []recloc;
 
 myio_close();
 return 0;
}

