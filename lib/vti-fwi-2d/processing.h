#ifndef PROCESSING_H
#define PROCESSING_H

void aniso_nmo(float *dataout,const float *datain,const float *vnmo,const float *eta,int nt,float ot,float dt,float *offset,int noffset);

void aniso_nmo_stack(float *dataout,const float *datain,const float *vnmo,const float *eta,int nt,float ot,float dt,float *offset,int noffset);

void vint2vrms(float *&vrms,int &nt,float ot,float dt,float *vint,int nz,float dz);

#endif
