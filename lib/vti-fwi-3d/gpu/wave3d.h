#ifndef WAVE3D_H
#define WAVE3D_H

#define PI 3.14159265359
#define ONETHIRD 0.3333333333
#define TWOTHIRD 0.6666666666

#define C0 -2.84722222222
#define C1 1.6
#define C2 -0.2
#define C3 0.02539682539
#define C4 -0.00178571428

#define DAMPER 0.95

#define BLOCK_DIM 16
#define NUPDATE 64
#define HALF_STENCIL 4
#define NLAG 32

#include <vector>
#include <string>

// memcpy3d.cu
void memcpyCpuToCpu2(float *dest1,float *sou1,float *dest2,float *sou2,size_t nbytes);

void memcpyCpuToCpu3(float *dest1,float *sou1,float *dest2,float *sou2,float *dest3,float *sou3,size_t nbytes);

void memcpyCpuToGpu2(float *dest1,float *sou1,float *dest2,float *sou2,size_t nbytes,cudaStream_t *stream);

void memcpyCpuToGpu3(float *dest1,float *sou1,float *dest2,float *sou2,float *dest3,float *sou3,size_t nbytes,cudaStream_t *stream);

void memcpyGpuToCpu2(float *dest1,float *sou1,float *dest2,float *sou2,size_t nbytes,cudaStream_t *stream);

void memcpyGpuToCpu3(float *dest1,float *sou1,float *dest2,float *sou2,float *dest3,float *sou3,size_t nbytes,cudaStream_t *stream);

void memcpyGpuToGpu2(float *dest1,float *sou1,float *dest2,float *sou2,size_t nbytes,cudaStream_t *stream);

void memcpyGpuToGpu3(float *dest1,float *sou1,float *dest2,float *sou2,float *dest3,float *sou3,size_t nbytes,cudaStream_t *stream);


// injectRecord3d.cu
__global__ void injectSource(float *SigmaX,float *SigmaZ,float source,int souIndex);

__global__ void injectResidual(float *residual,float *SigmaX,float *SigmaZ,int nr,const int *recIndex,float dt2);

void interpolateResidual(float *fineResidual,float *coarseResidual,int timeIndex,int nnt,int nr,int samplingTimeStep);

__global__ void extractAdjWfldAtSouLoc(float *gwavelet,float *SigmaXa,float *SigmaZa,int souIndexBlock,int it);

__global__ void recordData(float *data,float *SigmaX,float *SigmaZ,int nr,const int *recIndex);


// abc3d.cu
void init_abc(float *damping,int nx,int ny,int npad);

void init_abc(float *damping,int nx,int ny,int nz,int npad);

__global__ void abc(int iblock,int nx,int ny,int nz,int npad,float *nextSigmaX,float *curSigmaX1,float *nextSigmaZ,float *curSigmaZ1,const float *damping);

__global__ void abcXYZ(int ib,int nx,int ny,int nz,int npad,float *nextSigmaX,float *curSigmaX1,float *nextSigmaZ,float *curSigmaZ1,const float *damping);

__global__ void abcXY(int ib,int nx,int ny,int nz,int npad,float *nextSigmaX,float *curSigmaX1,float *nextSigmaZ,float *curSigmaZ1,const float *damping);


// kernel3d.cu
__global__ void forwardKernel(float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *c11,float *c13,float *c33,int nx,int ny,float dt2dx2,float dt2dy2,float dt2dz2);

__global__ void forwardKernelTopBlock(float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *c11,float *c13,float *c33,int nx,int ny,float dt2dx2,float dt2dy2,float dt2dz2);

__global__ void forwardKernelBottomBlock(float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *c11,float *c13,float *c33,int nx,int ny,float dt2dx2,float dt2dy2,float dt2dz2);

__global__ void gradientKernel(float *gc11,float *gc13,float *gc33,float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *curLambdaX,float *curLambdaZ,float *c11,float *c13,float *c33,int nx,int ny,float dx2,float dy2,float dz2,float dt2);

__global__ void gradientKernelTopBlock(float *gc11,float *gc13,float *gc33,float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *curLambdaX,float *curLambdaZ,float *c11,float *c13,float *c33,int nx,int ny,float dx2,float dy2,float dz2,float dt2);

__global__ void gradientKernelBottomBlock(float *gc11,float *gc13,float *gc33,float *nextSigmaX,float *curSigmaX1,float *prevSigmaX,float *nextSigmaZ,float *curSigmaZ0,float *curSigmaZ1,float *curSigmaZ2,float *prevSigmaZ,float *curLambdaX,float *curLambdaZ,float *c11,float *c13,float *c33,int nx,int ny,float dx2,float dy2,float dz2,float dt2);

__global__ void imagingKernel(float *image,float *nextSigmaX,float *nextSigmaZ,float *nextSigmaXa,float *nextSigmaZa,int nx,int ny);

__global__ void extendedImagingKernel(float *image,float *nextSigmaX,float *nextSigmaZ,float *nextSigmaXa,float *nextSigmaZa,int nx,int ny);

// functions
void modelData3d_f(float *souloc,int ns,float *recloc,float *wavelet,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate);

void modeling3d_f(float soulocX,float soulocY,float soulocZ,float *wavelet,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt);

void rtmCij3d_f(float *image,float *souloc,int ns,std::vector<int> &shotid,float *recloc,float *wavelet,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate);

void rtmVEpsDel(float *image,float *souloc,int ns,std::vector<int> &shotid,float *recloc,float *wavelet,float *v,float *eps,float *del,float *randboundaryV,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,float wbottom);

void rtm3d(float *image,int nx,int ny,int nz,int npad,float oz,float dz,float wbottom,std::vector<int> &shotid,int max_shot_per_job,int icall,const std::string &command);

void objFuncGradientCij3d(float *fgcij,float *souloc,int ns,std::vector<int> &shotid,float *recloc,float *wavelet,float *cij,float *padboundaryCij,float *randboundaryCij,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,std::vector<int> &GPUs,int ngpugroup);

void objFuncGradientCij3d_f(float *fgcij,float *souloc,int ns,std::vector<int> &shotid,float *recloc,float *wavelet,float *cij,float *padboundaryCij,float *randboundaryCij,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,std::vector<int> &GPUs);

void objFuncCij3d(float *f,float *souloc,int ns,std::vector<int> &shotid,float *recloc,float *wavelet,float *cij,float *padboundaryCij,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,std::vector<int> &GPUs,int ngpugroup);

void objFuncCij3d_f(float *f,float *souloc,int ns,std::vector<int> &shotid,float *recloc,float *wavelet,float *cij,float *padboundaryCij,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,std::vector<int> &GPUs);

double objFuncGradientVEpsDel(float *gv,float *geps,float *gdel,float *souloc,int ns,std::vector<int> &shotid,float *recloc,float *wavelet,float *v,float *eps,float *del,float *padboundaryV,float *randboundaryV,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,float v0,float eps0,float wbottom,std::vector<int> &GPUs,int ngpugroup);

double objFuncGradientVEpsDel_cluster(float *gvepsdel,float *vepsdel,int nx,int ny,int nz,int npad,float oz,float dz,float wbottom,float v0,float eps0,std::vector<int> &shotid,float pct,int max_shot_per_job,int icall,const std::string &command);

void objFuncGradientCij_cluster(float *fgcij,int nx,int ny,int nz,std::vector<int> &shotid,float pct,int max_shot_per_job,int icall,const std::string &command,float &time_in_minute);

void modelData3d_f(float *ddata,float *souloc,int ns,std::vector<int> &shotid,float *recloc,const float *wavelet,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,std::vector<int> &GPUs,int ngpugroup);

void modelDataCij3d_f(float *ddata,float *souloc,int ns,std::vector<int> &shotid,float *recloc,const float *wavelet,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,std::vector<int> &GPUs);

void waveletGradient3d_f(float *gwavelet,const float *ddata,float *souloc,int ns,std::vector<int> &shotid,float *recloc,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,std::vector<int> &GPUs,int ngpugroup);

void waveletGradientCij3d_f(float *gwavelet,const float *ddata,float *souloc,int ns,std::vector<int> &shotid,float *recloc,float *c11,float *c13,float *c33,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate,std::vector<int> &GPUs);

void odcig3d_f(float *image,float *souloc,int ns,float *recloc,float *wavelet,float *v,float *eps,float *del,int nx,int ny,int nz,int nt,int npad,float ox,float oy,float oz,float ot,float dx,float dy,float dz,float dt,float samplingRate);

void sumImageTime(float *image,float *h_image,size_t nElemBlock);

void extendedImaging(float **image,int ndg,int b,int e,int k,float ***d_SigmaX,float ***d_SigmaZ,float ***d_SigmaXa,float ***d_SigmaZa,const int *nbuffSigma,const float *d_damping,float **d_v,float **d_eps,float **d_del,const int nbuffVEpsDel,float *wavelet,int souIndexBlock,int souBlock,float *data,int nr,const int *recIndex,int recBlock,int nx,int ny,int nz,int npad,int nt,float dx2,float dy2,float dz2,float dt2,cudaStream_t *stream,int gpu,int NGPU);

void forwardRandom(int b,int e,int k,float ***d_SigmaX,float ***d_SigmaZ,const int *nbuffSigma,float **d_v,float **d_eps,float **d_del,const int nbuffVEpsDel,float *wavelet,int souIndexBlock,int souBlock,int nx,int ny,int nz,int nt,int npad,float dx2,float dy2,float dz2,float dt2,cudaStream_t *stream,int gpu,int NGPU);

#endif
