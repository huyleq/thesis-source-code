#include <cmath>
#include <iostream>
#include <cstring>

#include "mylib.h"

#include "gaussian_filter.h"

using namespace std;

void init_filter(float *filter,int sigma){
    int width=4*sigma+1;
    int width2=width*width;
    int width3=width2*width;
    float center=2*sigma;
    float sigma2=2*(float)sigma*(float)sigma;
    #pragma omp parellel for
    for(int iz=0;iz<width;iz++){
        for(int iy=0;iy<width;iy++){
            for(int ix=0;ix<width;ix++){
                float d=(iz-center)*(iz-center)+(iy-center)*(iy-center)+(ix-center)*(ix-center);
                int i=ix+iy*width+iz*width2;
                filter[i]=exp(-d/sigma2);
            }
        }
    }
    float s=sum(filter,width3);
    scale(filter,1./s,width3);
    return;
}

void init_filter(float *filter,float sigmaf,float dx){
    int width=std::round(4*sigmaf/dx);
    if(width%2==0) width++;
    int width2=width*width;
    int width3=width2*width;
    float center=(width-1.f)/2;
    float sigma2=2*sigmaf*sigmaf;
    #pragma omp parellel for
    for(int iz=0;iz<width;iz++){
        for(int iy=0;iy<width;iy++){
            for(int ix=0;ix<width;ix++){
                float d=(iz-center)*(iz-center)+(iy-center)*(iy-center)+(ix-center)*(ix-center);
                d*=dx*dx;
                int i=ix+iy*width+iz*width2;
                filter[i]=exp(-d/sigma2);
            }
        }
    }
    float s=sum(filter,width3);
    scale(filter,1./s,width3);
    return;
}

void apply_filter(float *fx,float *x,int nx,int ny,int nz,float *filter,int sigma){
    int width=4*sigma+1,width2=width*width,width3=width2*width;
    int center=2*sigma;
    long nxy=nx*ny;
    #pragma omp parellel for num_threads(16)
    for(int iz=center;iz<nz-center;iz++){
        for(int iy=center;iy<ny-center;iy++){
            for(int ix=center;ix<nx-center;ix++){
                size_t ixyz=ix+iy*nx+iz*nxy;
                float temp=0.f;
                #pragma omp simd reduction(+:temp)
                for(int izf=-center;izf<=center;izf++){
                    for(int iyf=-center;iyf<=center;iyf++){
                        for(int ixf=-center;ixf<=center;ixf++){
                            int ixyzf=(ixf+center)+(iyf+center)*width+(izf+center)*width2;
                            size_t i=(ix-ixf)+(iy-iyf)*nx+(iz-izf)*nxy;
                            temp+=x[i]*filter[ixyzf];
                        }
                    }
                }
                fx[ixyz]=temp;
            }
        }
    }
    return;
}

void smooth_gradient(float *g,int nx,int ny,int nz,int npad,int nwbottom,int max_iz,float max_sigma,float dx){
    long nxy=nx*ny,nxyz=nxy*nz;
    float *tg=new float[nxyz];
    memcpy(tg,g,nxyz*sizeof(float));
    for(int iz=npad+nwbottom;iz<max_iz;iz++){
        float sigmaf=(float)(iz-max_iz)*max_sigma/(float)(npad+nwbottom-max_iz);
        int width=std::round(4*sigmaf/dx);
        if(width%2==0) width++;
        int width2=width*width,width3=width2*width;
        int center=(width-1)/2;
        float *filter=new float[width3];
        init_filter(filter,sigmaf,dx);
        #pragma omp parellel for num_threads(16)
        for(int iy=npad;iy<ny-npad;iy++){
            for(int ix=npad;ix<nx-npad;ix++){
                size_t ixyz=ix+iy*nx+iz*nxy;
                float temp=0.f;
                #pragma omp simd reduction(+:temp)
                for(int izf=-center;izf<=center;izf++){
                    for(int iyf=-center;iyf<=center;iyf++){
                        for(int ixf=-center;ixf<=center;ixf++){
                            int ixyzf=(ixf+center)+(iyf+center)*width+(izf+center)*width2;
                            size_t i=(ix-ixf)+(iy-iyf)*nx+(iz-izf)*nxy;
                            temp+=tg[i]*filter[ixyzf];
                        }
                    }
                }
                g[ixyz]=temp;
            }
        }
        delete []filter;
    }
    delete []tg;
    return;
}
