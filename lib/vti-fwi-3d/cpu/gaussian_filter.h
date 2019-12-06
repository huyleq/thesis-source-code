#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

void init_filter(float *filter,int sigma);

void init_filter(float *filter,float sigmaf,float dx);

void apply_filter(float *fx,float *x,int nx,int ny,int nz,float *filter,int sigma);

void smooth_gradient(float *g,int nx,int ny,int nz,int npad,int nwbottom,int max_iz,float max_sigma,float dx);

#endif
