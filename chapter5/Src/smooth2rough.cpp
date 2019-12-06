#include "myio.h"
#include "mylib.h"
#include <random>
#include <chrono>
#include <cstring>

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
    int nx,ny,nz;
    float ox,oy,oz,dx,dy,dz;
    from_header("bgv","n1",nx,"o1",ox,"d1",dx);
    from_header("bgv","n2",ny,"o2",oy,"d2",dy);
    from_header("bgv","n3",nz,"o3",oz,"d3",dz);
    int nxyz=nx*ny*nz;

    float *bgv=new float[nxyz](); read("bgv",bgv,nxyz);
    float *bgeps=new float[nxyz]();
    float *bgdel=new float[nxyz]();
    float *v=new float[nxyz]();
    float *eps=new float[nxyz]();
    float *del=new float[nxyz]();

    float maxv=max(bgv,nxyz); 
    float minv=min(bgv,nxyz);
    float range=maxv-minv,saltv=4000.,maxeps=0.2,maxdel=0.1,dv=0.1*saltv,depsdel=0.1,dvlayer=100.;

    #pragma omp parallel for num_threads(16)
    for(int i=0;i<nxyz;i++){
        if(bgv[i]<saltv){
            float v0=(bgv[i]-minv)/range;
            bgeps[i]=v0*maxeps;
            bgdel[i]=v0*maxdel;
        }
    }
    
    memcpy(v,bgv,nxyz*sizeof(float));
    memcpy(eps,bgeps,nxyz*sizeof(float));
    memcpy(del,bgdel,nxyz*sizeof(float));
    
    write("bgeps",bgeps,nxyz);
    to_header("bgeps","n1",nx,"o1",ox,"d1",dx);
    to_header("bgeps","n2",ny,"o2",oy,"d2",dy);
    to_header("bgeps","n3",nz,"o3",oz,"d3",dz);
    
    write("bgdel",bgdel,nxyz);
    to_header("bgdel","n1",nx,"o1",ox,"d1",dx);
    to_header("bgdel","n2",ny,"o2",oy,"d2",dy);
    to_header("bgdel","n3",nz,"o3",oz,"d3",dz);
    
    unsigned seed=chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_real_distribution<float> distribution(0,1);

    float vlayer=minv+dvlayer;
    while(vlayer<saltv){
        float r=(2*distribution(generator)-1)*dv;
        float r1=distribution(generator)*depsdel;
        float r2=distribution(generator)*depsdel;
        float deps=max(r1,r2),ddel=min(r1,r2);
        #pragma omp parallel for num_threads(16)
        for(int i=0;i<nxyz;i++){
            if(bgv[i]>=vlayer && bgv[i]<vlayer+dvlayer){
                v[i]=bgv[i]+r;
                eps[i]=bgeps[i]+deps;
                del[i]=bgdel[i]+ddel;
            }
        }
        vlayer+=dvlayer;
    }
    
    write("v",v,nxyz);
    to_header("v","n1",nx,"o1",ox,"d1",dx);
    to_header("v","n2",ny,"o2",oy,"d2",dy);
    to_header("v","n3",nz,"o3",oz,"d3",dz);
    
    write("eps",eps,nxyz);
    to_header("eps","n1",nx,"o1",ox,"d1",dx);
    to_header("eps","n2",ny,"o2",oy,"d2",dy);
    to_header("eps","n3",nz,"o3",oz,"d3",dz);
    
    write("del",del,nxyz);
    to_header("del","n1",nx,"o1",ox,"d1",dx);
    to_header("del","n2",ny,"o2",oy,"d2",dy);
    to_header("del","n3",nz,"o3",oz,"d3",dz);
    
    delete []v;delete []eps;delete []del;
    delete []bgv;delete []bgeps;delete []bgdel;

    myio_close();
    return 0;
}
