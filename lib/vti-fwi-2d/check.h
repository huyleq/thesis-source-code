#ifndef CHECK_H
#define CHECK_H

void checkCij(float *c11,float *c13,float *c33,float c110,float c130,float c330,size_t n,float *m);

void checkEpsDel(float *eps,float *del,float eps0,float del0,size_t n,float *m);

void checkEta(float *eta,size_t n,float *m);

void checkVVhDel(float *v,float *vh,float *del,float v0,float vh0,float del0,size_t n,float *m);

void checkVnVh(float *vn,float *vh,float vn0,float vh0,size_t n,float *m);

#endif
