#ifndef HESSIANOP_H
#define HESSIANOP_H

#include "LinearSolver.h"

class GNHessianOpCij:public Operator{
    public:
    GNHessianOpCij(float *d0,float *c11,float *c13,float *c33,float c110,float c130,float c330,float *wavelet,int *sloc,int ns,int *rloc,int nr,float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m):Operator(3*(nx+2*npad)*(nz+2*npad),3*(nx+2*npad)*(nz+2*npad)),_d0(d0),_c11(c11),_c13(c13),_c33(c33),_c110(c110),_c130(c130),_c330(c330),_wavelet(wavelet),_sloc(sloc),_rloc(rloc),_taper(taper),_ns(ns),_nr(nr),_nx(nx),_nz(nz),_nt(nt),_npad(npad),_dx(dx),_dz(dz),_dt(dt),_rate(rate),_ot(ot),_wbottom(wbottom),_m(m){};
    void forward(bool add,const float *model,float *data);
    void adjoint(bool add,float *model,const float *data);
    int *_sloc,*_rloc;
    float *_d0,*_c11,*_c13,*_c33,*_wavelet,*_taper,*_m;
    int _ns,_nr,_nx,_nz,_nt,_npad;
    float _c110,_c130,_c330,_dx,_dz,_dt,_rate,_ot,_wbottom;
};

class HessianOpCij:public Operator{
    public:
    HessianOpCij(float *d0,float *c11,float *c13,float *c33,float c110,float c130,float c330,float *wavelet,int *sloc,int ns,int *rloc,int nr,float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m):Operator(3*(nx+2*npad)*(nz+2*npad),3*(nx+2*npad)*(nz+2*npad)),_d0(d0),_c11(c11),_c13(c13),_c33(c33),_c110(c110),_c130(c130),_c330(c330),_wavelet(wavelet),_sloc(sloc),_rloc(rloc),_taper(taper),_ns(ns),_nr(nr),_nx(nx),_nz(nz),_nt(nt),_npad(npad),_dx(dx),_dz(dz),_dt(dt),_rate(rate),_ot(ot),_wbottom(wbottom),_m(m){};
    void forward(bool add,const float *model,float *data);
    void adjoint(bool add,float *model,const float *data);
    int *_sloc,*_rloc;
    float *_d0,*_c11,*_c13,*_c33,*_wavelet,*_taper,*_m;
    int _ns,_nr,_nx,_nz,_nt,_npad;
    float _c110,_c130,_c330,_dx,_dz,_dt,_rate,_ot,_wbottom;
};

class HessianOpVEpsDel:public Operator{
    public:
    HessianOpVEpsDel(float *d0,float *v,float *eps,float *del,float v0,float eps0,float del0,float *wavelet,int *sloc,int ns,int *rloc,int nr,float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m):Operator(3*(nx+2*npad)*(nz+2*npad),3*(nx+2*npad)*(nz+2*npad)),_d0(d0),_v(v),_eps(eps),_del(del),_v0(v0),_eps0(eps0),_del0(del0),_wavelet(wavelet),_sloc(sloc),_rloc(rloc),_taper(taper),_ns(ns),_nr(nr),_nx(nx),_nz(nz),_nt(nt),_npad(npad),_dx(dx),_dz(dz),_dt(dt),_rate(rate),_ot(ot),_wbottom(wbottom),_m(m){};
    void forward(bool add,const float *model,float *data);
    void adjoint(bool add,float *model,const float *data);
    int *_sloc,*_rloc;
    float *_d0,*_v,*_eps,*_del,*_wavelet,*_taper,*_m;
    int _ns,_nr,_nx,_nz,_nt,_npad;
    float _v0,_eps0,_del0,_dx,_dz,_dt,_rate,_ot,_wbottom;
};

class GNHessianOpVEpsDel:public Operator{
    public:
    GNHessianOpVEpsDel(float *d0,float *v,float *eps,float *del,float v0,float eps0,float del0,float *wavelet,int *sloc,int ns,int *rloc,int nr,float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m):Operator(3*(nx+2*npad)*(nz+2*npad),3*(nx+2*npad)*(nz+2*npad)),_d0(d0),_v(v),_eps(eps),_del(del),_v0(v0),_eps0(eps0),_del0(del0),_wavelet(wavelet),_sloc(sloc),_rloc(rloc),_taper(taper),_ns(ns),_nr(nr),_nx(nx),_nz(nz),_nt(nt),_npad(npad),_dx(dx),_dz(dz),_dt(dt),_rate(rate),_ot(ot),_wbottom(wbottom),_m(m){};
    void forward(bool add,const float *model,float *data);
    void adjoint(bool add,float *model,const float *data);
    int *_sloc,*_rloc;
    float *_d0,*_v,*_eps,*_del,*_wavelet,*_taper,*_m;
    int _ns,_nr,_nx,_nz,_nt,_npad;
    float _v0,_eps0,_del0,_dx,_dz,_dt,_rate,_ot,_wbottom;
};

class HessianOpVhEpsDel:public Operator{
    public:
    HessianOpVhEpsDel(float *d0,float *vh,float *eps,float *del,float vh0,float eps0,float del0,float *wavelet,int *sloc,int ns,int *rloc,int nr,float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m):Operator(3*(nx+2*npad)*(nz+2*npad),3*(nx+2*npad)*(nz+2*npad)),_d0(d0),_vh(vh),_eps(eps),_del(del),_vh0(vh0),_eps0(eps0),_del0(del0),_wavelet(wavelet),_sloc(sloc),_rloc(rloc),_taper(taper),_ns(ns),_nr(nr),_nx(nx),_nz(nz),_nt(nt),_npad(npad),_dx(dx),_dz(dz),_dt(dt),_rate(rate),_ot(ot),_wbottom(wbottom),_m(m){};
    void forward(bool add,const float *model,float *data);
    void adjoint(bool add,float *model,const float *data);
    int *_sloc,*_rloc;
    float *_d0,*_vh,*_eps,*_del,*_wavelet,*_taper,*_m;
    int _ns,_nr,_nx,_nz,_nt,_npad;
    float _vh0,_eps0,_del0,_dx,_dz,_dt,_rate,_ot,_wbottom;
};

class GNHessianOpVhEpsDel:public Operator{
    public:
    GNHessianOpVhEpsDel(float *d0,float *vh,float *eps,float *del,float vh0,float eps0,float del0,float *wavelet,int *sloc,int ns,int *rloc,int nr,float *taper,int nx,int nz,int nt,int npad,float dx,float dz,float dt,float rate,float ot,float wbottom,float *m):Operator(3*(nx+2*npad)*(nz+2*npad),3*(nx+2*npad)*(nz+2*npad)),_d0(d0),_vh(vh),_eps(eps),_del(del),_vh0(vh0),_eps0(eps0),_del0(del0),_wavelet(wavelet),_sloc(sloc),_rloc(rloc),_taper(taper),_ns(ns),_nr(nr),_nx(nx),_nz(nz),_nt(nt),_npad(npad),_dx(dx),_dz(dz),_dt(dt),_rate(rate),_ot(ot),_wbottom(wbottom),_m(m){};
    void forward(bool add,const float *model,float *data);
    void adjoint(bool add,float *model,const float *data);
    int *_sloc,*_rloc;
    float *_d0,*_vh,*_eps,*_del,*_wavelet,*_taper,*_m;
    int _ns,_nr,_nx,_nz,_nt,_npad;
    float _vh0,_eps0,_del0,_dx,_dz,_dt,_rate,_ot,_wbottom;
};

#endif
