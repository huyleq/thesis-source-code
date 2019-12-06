#include "wave3d.h"

__global__ void injectSource(float *SigmaX,float *SigmaZ,float source,int souIndexBlock){
 SigmaX[souIndexBlock]+=source;
 SigmaZ[souIndexBlock]+=source;
 return;
}

__global__ void extractAdjWfldAtSouLoc(float *gwavelet,float *SigmaXa,float *SigmaZa,int souIndexBlock,int it){
 gwavelet[it]+=SigmaXa[souIndexBlock]+SigmaZa[souIndexBlock];
 return;
}

__global__ void injectDipoleSource(float *SigmaX,float *SigmaZ,float source,int souIndexBlock,int nxy){
 SigmaX[souIndexBlock]+=source;
 SigmaZ[souIndexBlock]+=source;
 SigmaX[souIndexBlock+nxy]-=source;
 SigmaZ[souIndexBlock+nxy]-=source;
 return;
}

__global__ void recordData(float *data,float *SigmaX,float *SigmaZ,int nr,const int *recIndex){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  data[ir]=TWOTHIRD*SigmaX[recIndex[ir]]+ONETHIRD*SigmaZ[recIndex[ir]];
 }
 return;
}

__global__ void recordDipoleData(float *data,float *SigmaX,float *SigmaZ,int nr,const int *recIndex,int nxy){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  data[ir]=TWOTHIRD*(SigmaX[recIndex[ir]]-SigmaX[recIndex[ir]+nxy])+ONETHIRD*(SigmaZ[recIndex[ir]]-SigmaZ[recIndex[ir]+nxy]);
 }
 return;
}

__global__ void injectResidual(float *residual,float *SigmaX,float *SigmaZ,int nr,const int *recIndex,float dt2){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  float s=dt2*residual[ir];
  SigmaX[recIndex[ir]]+=TWOTHIRD*s;
  SigmaZ[recIndex[ir]]+=ONETHIRD*s;
 }
 return;
}

__global__ void injectDipoleResidual(float *residual,float *SigmaX,float *SigmaZ,int nr,const int *recIndex,float dt2,int nxy){
 int ir=threadIdx.x+blockIdx.x*blockDim.x;
 if(ir<nr){
  float s=dt2*residual[ir];
  SigmaX[recIndex[ir]]+=TWOTHIRD*s;
  SigmaZ[recIndex[ir]]+=ONETHIRD*s;
  SigmaX[recIndex[ir]+nxy]-=TWOTHIRD*s;
  SigmaZ[recIndex[ir]+nxy]-=ONETHIRD*s;
 }
 return;
}

void interpolateResidual(float *fineResidual,float *coarseResidual,int timeIndex,int nnt,int nr,int samplingTimeStep){
//	fprintf(stderr,"timeIndex %d\n",timeIndex);
	float f=float(timeIndex)/float(samplingTimeStep);
	int i=f;
	if(i>=nnt-1){
	 #pragma omp parallel for num_threads(16)
     for(int ir=0;ir<nr;ir++){
      fineResidual[ir]=coarseResidual[(nnt-1)+ir*nnt];
     }
	}
	else{
     f=f-i;
	 #pragma omp parallel for num_threads(16)
     for(int ir=0;ir<nr;ir++){
      fineResidual[ir]=(1.-f)*coarseResidual[i+ir*nnt]+f*coarseResidual[(i+1)+ir*nnt];
     }
	}
	return;
}

