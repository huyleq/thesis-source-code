#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

#include "myio.h"
#include "mylib.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);

    int ntrace;
    from_header("sorted_index","n2",ntrace);
    fprintf(stderr,"ntrace %d\n",ntrace);
    float *ind=new float[5*ntrace];
    read("sorted_index",ind,5*ntrace);

    int irec=0,i=0,nsou=0;
    vector<float> gx,gy;
    vector<int> traceIndex,ntrace1;
    while(i<ntrace){
        float gy1=ind[1+i*5];
        gy.push_back(gy1);
        float gx1=ind[2+i*5];
        gx.push_back(gx1);
        int nsou1=1;
        fprintf(stderr,"receiver %d gx1=%.10f gy1=%.10f",irec,gx1,gy1);
        while(i<ntrace-1 && ind[1+(i+1)*5]==gy1 && ind[2+(i+1)*5]==gx1){
            nsou1++;
            i++;
        }
        traceIndex.push_back(nsou);
        ntrace1.push_back(nsou1);
        fprintf(stderr," has %d traces/shots starting from %d shots\n",ntrace1.back(),traceIndex.back());
        i++;
        irec++;
        nsou+=nsou1;
    }

    fprintf(stderr,"total number of shots %d should be equal to number of traces %d\n",nsou,ntrace);

    vector<int> recLineIndex;
    recLineIndex.push_back(0);
    for(int i=1;i<gy.size();i++){
        if(gy[i]-gy[i-1]>300.f) recLineIndex.push_back(i);
    }
    recLineIndex.push_back(gy.size());

    int jgx;
    get_param("jgx",jgx);
    
    vector<float> gxout,gyout;
    vector<int> ntraceout,traceIndexout;

    int nRecLine=recLineIndex.size()-1;
    fprintf(stderr,"there are %d receiver lines\n",nRecLine);
    
    for(int i=0;i<nRecLine;i++){
//    for(int i=11;i<=13;i++){
        int r=0;
        if(i%2==0) r=jgx/2;
        int b=recLineIndex[i],e=recLineIndex[i+1];
        fprintf(stderr,"receiver line %d starts at receiver %d ends at receiver %d\n",i,b,e);
        vector<int> ig(e-b);
        iota(ig.begin(),ig.end(),b);
        sort(ig.begin(),ig.end(),[&gx](size_t i1, size_t i2) {return gx[i1] < gx[i2];});
        for(int j=0;j<e-b;j++){
            if(j%jgx==r){
                gxout.push_back(gx[ig[j]]);
                gyout.push_back(gy[ig[j]]);
                ntraceout.push_back(ntrace1[ig[j]]);
                traceIndexout.push_back(traceIndex[ig[j]]);
            }
        }
    }

    int nt;
    float ot,dt;
    from_header("datain","n1",nt,"o1",ot,"d1",dt);

    int start=0;
    vector<float> souloc,recloc;
    for(int i=0;i<gxout.size();i++){
        fprintf(stderr,"receiver %d gx=%.10f gy=%.10f has %d traces/shots starting from %d in original from %d in new\n",i,gxout[i],gyout[i],ntraceout[i],traceIndexout[i],start);
        recloc.push_back(gxout[i]);
        recloc.push_back(gyout[i]);
        recloc.push_back(25.f);
        recloc.push_back(ntraceout[i]);
        recloc.push_back(start);
        
        vector<int> souLineIndex;
        souLineIndex.push_back(traceIndexout[i]);
        for(int j=traceIndexout[i];j<traceIndexout[i]+ntraceout[i];j++){
            if(ind[3+j*5]-ind[3+(j-1)*5]>200.f) souLineIndex.push_back(j);
        }
        souLineIndex.push_back(traceIndexout[i]+ntraceout[i]);
        int nSouLine=souLineIndex.size()-1;
        fprintf(stderr,"this receiver has %d shot lines\n",nSouLine);

        for(int j=traceIndexout[i];j<traceIndexout[i]+ntraceout[i];j++){
            souloc.push_back(ind[3+j*5]);
            souloc.push_back(ind[4+j*5]);
            souloc.push_back(0.f);
        }
        float *data1=new float[nt*ntraceout[i]];
        #pragma omp parallel for num_threads(16)
        for(int j=traceIndexout[i];j<traceIndexout[i]+ntraceout[i];j++){
            size_t pos=(long long)ind[j*5]*(long long)nt;
            read("datain",data1+(j-traceIndexout[i])*nt,nt,pos);
        }
        write("dataout",data1,nt*ntraceout[i],std::ios_base::app);
        delete []data1;
        start+=ntraceout[i];
    }

    write("souloc",&souloc[0],souloc.size());
    to_header("souloc","n1",3,"o1",0,"d1",1);
    to_header("souloc","n2",souloc.size()/3,"o2",0,"d2",1);

    write("recloc",&recloc[0],recloc.size());
    to_header("recloc","n1",5,"o1",0,"d1",1);
    to_header("recloc","n2",gxout.size(),"o2",0,"d2",1);

    int total=*(recloc.end()-1)+*(recloc.end()-2);
    fprintf(stderr,"total number of traces %d\n",total);
    to_header("dataout","n1",nt,"o1",ot,"d1",dt);
    to_header("dataout","n2",total,"o2",0,"d2",1);

    delete []ind;
    
    myio_close();
    return 0;
}
