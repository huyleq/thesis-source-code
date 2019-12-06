#ifndef INIT_H
#define INIT_H

#include <string>
#include <cmath>

using namespace std;

const double pi=4.f*atan(1.f);

const float DAMPER=0.95f;

void init2d(int &nnx,int &nnz,int &nx,int &nz,int &nt,float &dx,float &dz,float &dt,float &ox,float &oz,float &ot,int &npad);

void init_rec(int &n_rec,float &d_rec,float &o_rec,float &z_rec);

void init_rec_loc(int *rec_loc,int n_rec,float d_rec,float o_rec,float z_rec,float dx,float dz,float ox,float oz,int npad);

void init_shot(int &n_shot,float &d_shot,float &o_shot,float &z_shot);

void init_shot_loc(int *shot_loc,int n_shot,float d_shot,float o_shot,float z_shot,float dx,float dz,float ox,float oz,int npad);

void pad(float *m,int nx,int nz,int npad);

void init_model(const string &s,float *m,int nx,int nz,int npad);

void init_abc(float *taper,int nx,int nz,int npad);

#endif
