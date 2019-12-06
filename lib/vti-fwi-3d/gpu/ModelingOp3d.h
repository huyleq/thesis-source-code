#ifndef MODELINGOP3D_H
#define MODELINGOP3D_H

#include <cmath>
#include "LinearSolver.h"

class ModelingOp3d:public Operator{
    public:
    ModelingOp3d(float *souloc,int ns,vector<int> shotid,float *recloc,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,vector<int> GPUs,int ngpugroup):Operator(nt,(long long)((nt-1)/std::round(samplingRate/dt)+1)*(long long)(souloc[5*ns-1]+souloc[5*ns-2])),_souloc(souloc),_ns(ns),_shotid(shotid),_recloc(recloc),_c11(c11),_c13(c13),_c33(c33),_nx(nx),_ny(ny),_nz(nz),_nt(nt),_npad(npad),_ox(ox),_oy(oy),_oz(oz),_ot(ot),_dx(dx),_dy(dy),_dz(dz),_dt(dt),_samplingRate(samplingRate),_GPUs(GPUs),_ngpugroup(ngpugroup){};
    void forward(bool add,const float *model,float *data);
    void adjoint(bool add,float *model,const float *data);
    float *_souloc,*_recloc,*_c11,*_c13,*_c33;
    int _ns,_nx,_ny,_nz,_nt,_npad,_ngpugroup;
    float _ox,_oy,_oz,_ot,_dx,_dy,_dz,_dt,_samplingRate;
    vector<int> _shotid,_GPUs;
};

#endif
