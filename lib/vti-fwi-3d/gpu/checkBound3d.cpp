#include <iostream>
#include "myio.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
    int nx,ny,nz;
    float ox,oy,oz,dx,dy,dz;
    from_header("v","n1",nx,"o1",ox,"d1",dx);
    from_header("v","n2",ny,"o2",oy,"d2",dy);
    from_header("v","n3",nz,"o3",oz,"d3",dz);

    long long nxy=nx*ny,nxyz=nxy*nz;
    float *v=new float[nxyz]; read("v",v,nxyz);
    float *upper=new float[nxyz]; read("upper",upper,nxyz);
    float *lower=new float[nxyz]; read("lower",lower,nxyz);
    
    long long count_lower=0,count_upper=0;
    float violate_lower=0.f,violate_upper=0.f;

//    #pragma omp parallel for
    for(size_t i=0;i<nxyz;i++){
        if(v[i]>upper[i]){
//            v[i]-=upper[i];
            v[i]=1;
            violate_upper+=v[i];
            count_upper++;
        }
        else if(v[i]<lower[i]){
//            v[i]-=lower[i];
            v[i]=-1;
            violate_lower+=v[i];
            count_lower++;
        }
        else v[i]=0.f;
    }
    cout<<"Violate upper bound at "<<count_upper<<" locations, "<<(float)count_upper/(float)nxyz*100<<". Average violation "<<violate_upper/count_upper<<endl;
    cout<<"Violate lower bound at "<<count_lower<<" locations, "<<(float)count_lower/(float)nxyz*100<<". Average violation "<<violate_lower/count_lower<<endl;

    write("violate",v,nxyz);
    to_header("violate","n1",nx,"o1",ox,"d1",dx);
    to_header("violate","n2",ny,"o2",oy,"d2",dy);
    to_header("violate","n3",nz,"o3",oz,"d3",dz);

    delete []v;delete []upper;delete []lower;

    myio_close();
    return 0;
}
