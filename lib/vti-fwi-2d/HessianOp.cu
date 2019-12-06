#include "mylib.h"
#include "wave.h"
#include "HessianOp.h"

void HessianOpCij::forward(bool wantadd,const float *model,float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) hessianCij(data,data+nnxz,data+2*nnxz,_d0,_c11,_c13,_c33,model,model+nnxz,model+2*nnxz,_c110,_c130,_c330,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tdata=new float[_dataSize]();
        hessianCij(tdata,tdata+nnxz,tdata+2*nnxz,_d0,_c11,_c13,_c33,model,model+nnxz,model+2*nnxz,_c110,_c130,_c330,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(data,data,tdata,_dataSize);
        delete []tdata;
    }
    return;
}

void HessianOpCij::adjoint(bool wantadd,float *model,const float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) hessianCij(model,model+nnxz,model+2*nnxz,_d0,_c11,_c13,_c33,data,data+nnxz,data+2*nnxz,_c110,_c130,_c330,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tmodel=new float[_modelSize]();
        hessianCij(tmodel,tmodel+nnxz,tmodel+2*nnxz,_d0,_c11,_c13,_c33,data,data+nnxz,data+2*nnxz,_c110,_c130,_c330,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(model,model,tmodel,_modelSize);
        delete []tmodel;
    }
    return;
}

void GNHessianOpCij::forward(bool wantadd,const float *model,float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) GNhessianCij(data,data+nnxz,data+2*nnxz,_d0,_c11,_c13,_c33,model,model+nnxz,model+2*nnxz,_c110,_c130,_c330,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tdata=new float[_dataSize]();
        GNhessianCij(tdata,tdata+nnxz,tdata+2*nnxz,_d0,_c11,_c13,_c33,model,model+nnxz,model+2*nnxz,_c110,_c130,_c330,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(data,data,tdata,_dataSize);
        delete []tdata;
    }
    return;
}

void GNHessianOpCij::adjoint(bool wantadd,float *model,const float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) GNhessianCij(model,model+nnxz,model+2*nnxz,_d0,_c11,_c13,_c33,data,data+nnxz,data+2*nnxz,_c110,_c130,_c330,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tmodel=new float[_modelSize]();
        GNhessianCij(tmodel,tmodel+nnxz,tmodel+2*nnxz,_d0,_c11,_c13,_c33,data,data+nnxz,data+2*nnxz,_c110,_c130,_c330,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(model,model,tmodel,_modelSize);
        delete []tmodel;
    }
    return;
}

void HessianOpVEpsDel::forward(bool wantadd,const float *model,float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) hessianVEpsDel(data,data+nnxz,data+2*nnxz,_d0,_v,_eps,_del,model,model+nnxz,model+2*nnxz,_v0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tdata=new float[_dataSize]();
        hessianVEpsDel(tdata,tdata+nnxz,tdata+2*nnxz,_d0,_v,_eps,_del,model,model+nnxz,model+2*nnxz,_v0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(data,data,tdata,_dataSize);
        delete []tdata;
    }
    return;
}

void HessianOpVEpsDel::adjoint(bool wantadd,float *model,const float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) hessianVEpsDel(model,model+nnxz,model+2*nnxz,_d0,_v,_eps,_del,data,data+nnxz,data+2*nnxz,_v0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tmodel=new float[_modelSize]();
        hessianVEpsDel(tmodel,tmodel+nnxz,tmodel+2*nnxz,_d0,_v,_eps,_del,data,data+nnxz,data+2*nnxz,_v0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(model,model,tmodel,_modelSize);
        delete []tmodel;
    }
    return;
}

void GNHessianOpVEpsDel::forward(bool wantadd,const float *model,float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) GNhessianVEpsDel(data,data+nnxz,data+2*nnxz,_d0,_v,_eps,_del,model,model+nnxz,model+2*nnxz,_v0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tdata=new float[_dataSize]();
        GNhessianVEpsDel(tdata,tdata+nnxz,tdata+2*nnxz,_d0,_v,_eps,_del,model,model+nnxz,model+2*nnxz,_v0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(data,data,tdata,_dataSize);
        delete []tdata;
    }
    return;
}

void GNHessianOpVEpsDel::adjoint(bool wantadd,float *model,const float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) GNhessianVEpsDel(model,model+nnxz,model+2*nnxz,_d0,_v,_eps,_del,data,data+nnxz,data+2*nnxz,_v0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tmodel=new float[_modelSize]();
        GNhessianVEpsDel(tmodel,tmodel+nnxz,tmodel+2*nnxz,_d0,_v,_eps,_del,data,data+nnxz,data+2*nnxz,_v0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(model,model,tmodel,_modelSize);
        delete []tmodel;
    }
    return;
}

void HessianOpVhEpsDel::forward(bool wantadd,const float *model,float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) hessianVhEpsDel(data,data+nnxz,data+2*nnxz,_d0,_vh,_eps,_del,model,model+nnxz,model+2*nnxz,_vh0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tdata=new float[_dataSize]();
        hessianVhEpsDel(tdata,tdata+nnxz,tdata+2*nnxz,_d0,_vh,_eps,_del,model,model+nnxz,model+2*nnxz,_vh0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(data,data,tdata,_dataSize);
        delete []tdata;
    }
    return;
}

void HessianOpVhEpsDel::adjoint(bool wantadd,float *model,const float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) hessianVhEpsDel(model,model+nnxz,model+2*nnxz,_d0,_vh,_eps,_del,data,data+nnxz,data+2*nnxz,_vh0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tmodel=new float[_modelSize]();
        hessianVhEpsDel(tmodel,tmodel+nnxz,tmodel+2*nnxz,_d0,_vh,_eps,_del,data,data+nnxz,data+2*nnxz,_vh0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(model,model,tmodel,_modelSize);
        delete []tmodel;
    }
    return;
}

void GNHessianOpVhEpsDel::forward(bool wantadd,const float *model,float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) GNhessianVhEpsDel(data,data+nnxz,data+2*nnxz,_d0,_vh,_eps,_del,model,model+nnxz,model+2*nnxz,_vh0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tdata=new float[_dataSize]();
        GNhessianVhEpsDel(tdata,tdata+nnxz,tdata+2*nnxz,_d0,_vh,_eps,_del,model,model+nnxz,model+2*nnxz,_vh0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(data,data,tdata,_dataSize);
        delete []tdata;
    }
    return;
}

void GNHessianOpVhEpsDel::adjoint(bool wantadd,float *model,const float *data){
    long long nnxz=(_nx+2*_npad)*(_nz+2*_npad);
    if(!wantadd) GNhessianVhEpsDel(model,model+nnxz,model+2*nnxz,_d0,_vh,_eps,_del,data,data+nnxz,data+2*nnxz,_vh0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
    else{
        float *tmodel=new float[_modelSize]();
        GNhessianVhEpsDel(tmodel,tmodel+nnxz,tmodel+2*nnxz,_d0,_vh,_eps,_del,data,data+nnxz,data+2*nnxz,_vh0,_eps0,_del0,_wavelet,_sloc,_ns,_rloc,_nr,_taper,_nx,_nz,_nt,_npad,_dx,_dz,_dt,_rate,_ot,_wbottom,_m);
        add(model,model,tmodel,_modelSize);
        delete []tmodel;
    }
    return;
}

