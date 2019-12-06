#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>

#include "myio.h"
#include "mylib.h"
#include "wave3d.h"
#include "ModelingOp3d.h"

void ModelingOp3d::forward(bool wantadd,const float *model,float *data){
    if(!wantadd) modelData3d_f(data,_souloc,_ns,_shotid,_recloc,model,_c11,_c13,_c33,_nx,_ny,_nz,_nt,_npad,_ox,_oy,_oz,_ot,_dx,_dy,_dz,_dt,_samplingRate,_GPUs,_ngpugroup);
    else{
        float *tdata=new float[_dataSize];
        modelData3d_f(tdata,_souloc,_ns,_shotid,_recloc,model,_c11,_c13,_c33,_nx,_ny,_nz,_nt,_npad,_ox,_oy,_oz,_ot,_dx,_dy,_dz,_dt,_samplingRate,_GPUs,_ngpugroup);
        add(data,data,tdata,_dataSize);
        delete []tdata;
    }
    return;
}

void ModelingOp3d::adjoint(bool wantadd,float *model,const float *data){
    if(!wantadd) waveletGradient3d_f(model,data,_souloc,_ns,_shotid,_recloc,_c11,_c13,_c33,_nx,_ny,_nz,_nt,_npad,_ox,_oy,_oz,_ot,_dx,_dy,_dz,_dt,_samplingRate,_GPUs,_ngpugroup); 
    else{
        float *gw=new float[_nt];
        waveletGradient3d_f(gw,data,_souloc,_ns,_shotid,_recloc,_c11,_c13,_c33,_nx,_ny,_nz,_nt,_npad,_ox,_oy,_oz,_ot,_dx,_dy,_dz,_dt,_samplingRate,_GPUs,_ngpugroup);
        add(model,model,gw,_nt);
        delete []gw;
    } 
    return;
}

