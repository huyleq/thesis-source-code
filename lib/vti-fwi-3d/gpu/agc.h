#ifndef AGC_H
#define AGC_H

void runsum(int nt,int halfwidth,float *w,float *s);

void agc(int nt,int halfwidth,const float *d,float *hd,float *sd);

double residualAGC(int nt,int ntr,int halfwidth,const float *d,const float *d0,float *res,float *adjsou);

double residualAGC(int nt,int ntr,int halfwidth,float *d,const float *d0);

void tpower(float *data,int nt,float ot,float dt,int nr,float p);

#endif
