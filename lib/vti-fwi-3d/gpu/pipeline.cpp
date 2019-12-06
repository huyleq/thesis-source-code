#include <cstdio>
#include <cmath>

#include "myio.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);

    int ngpu,nupdate,nb,nt;
    get_param("ngpu",ngpu,"nupdate",nupdate);
    get_param("nb",nb,"nt",nt);

    int pipelen=ngpu*(nupdate+3)+3;
    int nround=(nt-2)/(ngpu*nupdate);
    int roundlen=max(pipelen,nb);
    int nroundlen=nround*roundlen;
    int nk=(nround-1)*roundlen+pipelen+nb-1;
    fprintf(stderr,"pipelen=%d nround=%d roundlen=%d nk=%d\n",pipelen,nround,roundlen,nk);

    int recBlock=7;
    float dt,samplingRate;
    get_param("dt",dt,"samplingRate",samplingRate);
    int samplingTimeStep=round(samplingRate/dt),nnt=(nt-1)/samplingTimeStep+1;
    fprintf(stderr,"samplingTimeStep %d nnt %d\n",samplingTimeStep,nnt);
    int nr=10;
    float *data=new float[nnt*nr]();
    float *res=new float[nnt*nr]();
    int itdata=1,krecord=-2,gpurecord=-2,ktrans=-2;

    int nElemBlock=5;

    float *prev=new float[nb*nElemBlock]();
    float *cur=new float[nb*nElemBlock]();
    float *next1=new float[nb*nElemBlock]();
    float *next2=new float[nb*nElemBlock]();
    float *v=new float[nb*nElemBlock]();
    float *gv=new float[nb*nElemBlock]();
    
    float *h_v[2],*h_prev[2],*h_cur[2],*h_next1[2],*h_next2[2],*h_res[2],*h_gv0[2],*h_gv1[2];
    for(int i=0;i<2;i++){
        h_v[i]=new float[nElemBlock]();
        h_prev[i]=new float[nElemBlock]();
        h_cur[i]=new float[nElemBlock]();
        h_next1[i]=new float[nElemBlock]();
        h_next2[i]=new float[nElemBlock]();
        h_res[i]=new float[nr]();
        h_gv0[i]=new float[nElemBlock]();
        h_gv1[i]=new float[nElemBlock]();
    }
    float *h_data=new float[nr]();

    int nbuffSigma=nupdate+2,nbuffV=nupdate+4,nbuffG=nbuffV;
    float ****d_Sigma=new float ***[ngpu]();
    float ***d_v=new float**[ngpu]();
    float **d_data=new float*[ngpu]();
    float ***d_res=new float**[ngpu]();
    float ***d_gv=new float**[ngpu]();
    for(int gpu=0;gpu<ngpu;gpu++){
        d_Sigma[gpu]=new float**[nbuffSigma]();
        for(int i=0;i<nbuffSigma;i++){
            d_Sigma[gpu][i]=new float*[4]();
            for(int j=0;j<4;j++) d_Sigma[gpu][i][j]=new float[nElemBlock];
        }
        d_v[gpu]=new float*[nbuffV]();
        for(int i=0;i<nbuffV;i++) d_v[gpu][i]=new float[nElemBlock]();
        d_data[gpu]=new float[nr]();
        d_res[gpu]=new float*[2]();
        for(int i=0;i<2;i++) d_res[gpu][i]=new float[nr]();
        d_gv[gpu]=new float*[nbuffG]();
        for(int i=0;i<nbuffG;i++) d_gv[gpu][i]=new float[nElemBlock]();
    }

    for(int k=0;k<nk;k++){
        fprintf(stderr,"iteration k=%d\n",k);
        if(k<nroundlen){
            int ib=k%roundlen;
            if(ib<nb){
                fprintf(stderr,"transfer in block ib=%d from pageable to locked: prev %p to %p cur %p to %p v %p to %p\n",ib,prev+ib*nElemBlock,h_prev[k%2],cur+ib*nElemBlock,h_cur[k%2],v+ib*nElemBlock,h_v[k%2]);
                fprintf(stderr," transfer in grad block ib=%d from pageable at %p to locked at %p\n",ib,gv+ib*nElemBlock,h_gv0[k%2]);
            }
        }
        
        if(k>0 && k<=nroundlen){
            int ib=(k-1)%roundlen;
            if(ib<nb){
                fprintf(stderr,"transfer in block ib=%d from locked to gpu 0: prev %p to %p cur %p to %p v %p to %p\n",ib,h_prev[(k-1)%2],d_Sigma[0][0][k%4],h_cur[(k-1)%2],d_Sigma[0][1][k%4],h_v[(k-1)%2],d_v[0][k%nbuffV]);
                fprintf(stderr," transfer in grad block ib=%d from locked at %p to gpu 0 at %p\n",ib,h_gv0[(k-1)%2],d_gv[0][k%nbuffG]);
            }
        }

        for(int gpu=0;gpu<ngpu;gpu++){
            int kgpu=k+2-gpu*(nupdate+3);
            if(kgpu>2 && kgpu<=nupdate+1+nroundlen){
                for(int i=0;i<nupdate;i++){
                    int ib=(kgpu-3-i)%roundlen;
                    int iround=(kgpu-3-i)/roundlen;
                    if(ib==recBlock && iround>=0 && iround<nround){
                        int it=iround*ngpu*nupdate+gpu*nupdate+2+i;
                        it=nt-1-it;
                        float f=float(it+1)/float(samplingTimeStep);
                        int j=f;
                        if(j>=nnt-1) f=0.;
                        else f=f-j;
                        fprintf(stderr,"rec block %d will be updated by gpu=%d at k=%d kgpu=%d to time it=%d.\n interpolate residual at time %d at %p using %f residual at index %d at %p and %f at index %d at %p\n",ib,gpu,k+2,kgpu,it,it+1,h_res[k%2],1.-f,j,res+j,f,j+1,res+j+1);
                    }
                }
            }
            
            kgpu=k+1-gpu*(nupdate+3);
            if(kgpu>2 && kgpu<=nupdate+1+nroundlen){
                for(int i=0;i<nupdate;i++){
                    int ib=(kgpu-3-i)%roundlen;
                    int iround=(kgpu-3-i)/roundlen;
                    if(ib==recBlock && iround>=0 && iround<nround){
                        int it=iround*ngpu*nupdate+gpu*nupdate+2+i;
                        it=nt-1-it;
                        fprintf(stderr,"rec block %d will be updated by gpu=%d at k=%d kgpu=%d to time it=%d.\n transfer res at time %d from %p to %p\n",ib,gpu,k+1,kgpu,it,it+1,h_res[(k-1)%2],d_res[gpu][k%2]);
                    }
                }
            }
        }

        for(int gpu=0;gpu<ngpu;gpu++){
            int kgpu=k-gpu*(nupdate+3);
            if(kgpu>2 && kgpu<=nupdate+1+nroundlen){
                for(int i=0;i<nupdate;i++){
                    int ib=(kgpu-3-i)%roundlen;
                    int iround=(kgpu-3-i)/roundlen;
                    if(ib>=0 && ib<nb && iround>=0 && iround<nround){
                        int it=iround*ngpu*nupdate+gpu*nupdate+2+i;
                        it=nt-1-it;
                        fprintf(stderr,"gpu=%d kgpu=%d iround=%d i=%d updates block ib=%d at %p to time it=%d using block %d at %p block %d at %p block %d at %p at time %d block %d at %p at time %d v block at %p\n",gpu,kgpu,iround,i,ib,d_Sigma[gpu][i+2][(kgpu-i-2)%4],it,ib-1,d_Sigma[gpu][i+1][(kgpu-i-3)%4],ib,d_Sigma[gpu][i+1][(kgpu-i-2)%4],ib+1,d_Sigma[gpu][i+1][(kgpu-i-1)%4],it-1,ib,d_Sigma[gpu][i][(kgpu-i-2)%4],it-2,d_v[gpu][(kgpu-i-2)%nbuffV]);

                        fprintf(stderr," sum grad block ib=%d at time it=%d at %p\n",ib,it,d_gv[gpu][(kgpu-i-2)%nbuffG]);

//                        if(ib==recBlock && it==samplingTimeStep*itdata && itdata<nnt){
//                            krecord=k;
//                            gpurecord=gpu;
//                            fprintf(stderr,"gpu=%d k=%d it=%d itdata=%d record data at %p\n",gpu,k,it,itdata,d_data[gpu]);
//                        }
                        
                        if(ib==recBlock){
                            fprintf(stderr," update rec block %d to time %d using residual at time %d at %p\n",ib,it,it+1,d_res[gpu][(k-1)%2]);
                        }
                    }
                }
            }

            if(kgpu>nupdate+3 && kgpu<=nupdate+3+nroundlen){
                int ib=(kgpu-nupdate-4)%roundlen;
                if(ib<nb){
                    if(ngpu>1 && gpu<ngpu-1){
                        fprintf(stderr,"gpu=%d kgpu=%d transfers block ib=%d from gpu %d to gpu %d: prev %p to %p cur %p to %p v %p to %p\n",gpu,kgpu,ib,gpu,gpu+1,d_Sigma[gpu][nbuffSigma-2][(kgpu-nupdate-3)%4],d_Sigma[gpu+1][0][(kgpu-nupdate-3)%4],d_Sigma[gpu][nbuffSigma-1][(kgpu-nupdate-3)%4],d_Sigma[gpu+1][1][(kgpu-nupdate-3)%4],d_v[gpu][(kgpu-nupdate-3)%nbuffV],d_v[gpu+1][(kgpu-nupdate-3)%nbuffV]);
                        fprintf(stderr," transfer grad block ib=%d from gpu %d at %p to gpu %d at %p\n",ib,gpu,d_gv[gpu][(kgpu-nupdate-3)%nbuffG],gpu+1,d_gv[gpu+1][(kgpu-nupdate-3)%nbuffG]);
                        fprintf(stderr," zero out grad block ib=%d at %p\n",ib,d_gv[gpu][(kgpu-nupdate-3)%nbuffG]);
                    }
                    else{
                        fprintf(stderr,"gpu=%d kgpu=%d transfers block ib=%d from gpu %d to cpu: prev %p to %p cur %p to %p\n",gpu,kgpu,ib,gpu,d_Sigma[gpu][nbuffSigma-2][(kgpu-nupdate-3)%4],h_next1[k%2],d_Sigma[gpu][nbuffSigma-1][(kgpu-nupdate-3)%4],h_next2[k%2]);
                        fprintf(stderr," transfer grad block ib=%d from gpu %d at %p to cpu at %p\n",ib,gpu,d_gv[gpu][(kgpu-nupdate-3)%nbuffG],h_gv1[k%2]);
                    }
                }
            }
        }

//        if(k-1==krecord && itdata<nnt){
//            fprintf(stderr,"gpu=%d k=%d itdata=%d transfer data from %p to locked at %p\n",gpurecord,k,itdata,d_data[gpurecord],h_data);
//            krecord=-2;
//            gpurecord=-2;
//            ktrans=k;
//        }
        
//        if(k-1==ktrans && itdata<nnt){
//            fprintf(stderr,"k=%d itdata=%d transfer data from locked at %p to pageable at %p\n",k,itdata,h_data,data+itdata);
//            ktrans=-2;
//            itdata++;
//        }

        
        if(k>pipelen-2 && k<=pipelen-2+nroundlen){
            int ib=(k-pipelen+1)%roundlen;
            if(ib<nb){
                fprintf(stderr,"transfer out block ib=%d from locked to pageable: prev %p to %p cur %p to %p\n",ib,h_next1[(k-1)%2],prev+ib*nElemBlock,h_next2[(k-1)%2],cur+ib*nElemBlock);
                fprintf(stderr,"transfer out grad block ib=%d from locked at %p to pageable at %p\n",ib,h_gv1[(k-1)%2],gv+ib*nElemBlock);
            }
        }

        fprintf(stderr,"\n");
    }

    for(int i=0;i<2;i++){
        delete []h_v[i];
        delete []h_prev[i];
        delete []h_cur[i];
        delete []h_next1[i];
        delete []h_next2[i];
        delete []h_res[i];
        delete []h_gv0[i];
        delete []h_gv1[i];
    }
    delete []h_data;

    for(int gpu=0;gpu<ngpu;gpu++){
        for(int i=0;i<nbuffSigma;i++){
            for(int j=0;j<4;j++) delete []d_Sigma[gpu][i][j];
            delete []d_Sigma[gpu][i];
        }
        delete []d_Sigma[gpu];
        for(int i=0;i<nbuffV;i++) delete []d_v[gpu][i];
        delete []d_v[gpu];
        delete []d_data[gpu];
        for(int i=0;i<2;i++) delete []d_res[gpu][i];
        delete []d_res[gpu];
        for(int i=0;i<2;i++) delete []d_gv[gpu][i];
        delete []d_gv[gpu];
    }
    delete []d_Sigma;delete []d_v;delete []d_data;delete []d_res;delete []d_gv;

    delete []prev;delete []cur;delete []next1;delete []next2;delete []v;delete []data;
    delete []res;
    delete []gv;

    myio_close();
    return 0;
}
