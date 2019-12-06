#include "wave3d.h"

void memcpyCpuToCpu2(float *dest1,float *sou1,float *dest2,float *sou2,size_t nbytes){
 memcpy(dest1,sou1,nbytes);
 memcpy(dest2,sou2,nbytes);
}

void memcpyCpuToCpu3(float *dest1,float *sou1,float *dest2,float *sou2,float *dest3,float *sou3,size_t nbytes){
 memcpy(dest1,sou1,nbytes);
 memcpy(dest2,sou2,nbytes);
 memcpy(dest3,sou3,nbytes);
}

void memcpyCpuToGpu2(float *dest1,float *sou1,float *dest2,float *sou2,size_t nbytes,cudaStream_t *stream){
 cudaMemcpyAsync(dest1,sou1,nbytes,cudaMemcpyHostToDevice,*stream);
 cudaMemcpyAsync(dest2,sou2,nbytes,cudaMemcpyHostToDevice,*stream);
}

void memcpyCpuToGpu3(float *dest1,float *sou1,float *dest2,float *sou2,float *dest3,float *sou3,size_t nbytes,cudaStream_t *stream){
 cudaMemcpyAsync(dest1,sou1,nbytes,cudaMemcpyHostToDevice,*stream);
 cudaMemcpyAsync(dest2,sou2,nbytes,cudaMemcpyHostToDevice,*stream);
 cudaMemcpyAsync(dest3,sou3,nbytes,cudaMemcpyHostToDevice,*stream);
}

void memcpyGpuToCpu2(float *dest1,float *sou1,float *dest2,float *sou2,size_t nbytes,cudaStream_t *stream){
 cudaMemcpyAsync(dest1,sou1,nbytes,cudaMemcpyDeviceToHost,*stream);
 cudaMemcpyAsync(dest2,sou2,nbytes,cudaMemcpyDeviceToHost,*stream);
}

void memcpyGpuToCpu3(float *dest1,float *sou1,float *dest2,float *sou2,float *dest3,float *sou3,size_t nbytes,cudaStream_t *stream){
 cudaMemcpyAsync(dest1,sou1,nbytes,cudaMemcpyDeviceToHost,*stream);
 cudaMemcpyAsync(dest2,sou2,nbytes,cudaMemcpyDeviceToHost,*stream);
 cudaMemcpyAsync(dest3,sou3,nbytes,cudaMemcpyDeviceToHost,*stream);
}

void memcpyGpuToGpu2(float *dest1,float *sou1,float *dest2,float *sou2,size_t nbytes,cudaStream_t *stream){
 cudaMemcpyAsync(dest1,sou1,nbytes,cudaMemcpyDefault,*stream);
 cudaMemcpyAsync(dest2,sou2,nbytes,cudaMemcpyDefault,*stream);
}

void memcpyGpuToGpu3(float *dest1,float *sou1,float *dest2,float *sou2,float *dest3,float *sou3,size_t nbytes,cudaStream_t *stream){
 cudaMemcpyAsync(dest1,sou1,nbytes,cudaMemcpyDefault,*stream);
 cudaMemcpyAsync(dest2,sou2,nbytes,cudaMemcpyDefault,*stream);
 cudaMemcpyAsync(dest3,sou3,nbytes,cudaMemcpyDefault,*stream);
}

