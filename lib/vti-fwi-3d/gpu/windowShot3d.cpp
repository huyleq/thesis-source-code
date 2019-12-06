#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

#include "myio.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
    int ns,nr;
    from_header("souloc","n2",ns);
    float *souloc=new float[5*ns];
    read("souloc",souloc,5*ns);
    
    vector<int> shotid;
    float minx,maxx,miny,maxy;

    if(!get_array("shotid",shotid)){
        int nshot;
        if(from_header("shotidFile","n1",nshot)){
            float *shot=new float[nshot];
            read("shotidFile",shot,nshot);
            for(int i=0;i<nshot;i++) shotid.push_back(shot[i]);
            delete []shot;
        }
        else if(get_param("minx",minx,"maxx",maxx)){
            get_param("miny",miny,"maxy",maxy);
            for(int i=0;i<ns;i++){
                float sx=souloc[5*i],sy=souloc[5*i+1];
                if(sx>=minx && sx<=maxx && sy>=miny && sy<=maxy){
                    fprintf(stderr,"shot %d in range\n",i);
                    shotid.push_back(i);
                }
            }
        }
        else{
            vector<int> shotrange;
            if(!get_array("shotrange",shotrange)){
                shotrange.push_back(0);
                shotrange.push_back(ns);
            }
            vector<int> badshot;
            get_array("badshot",badshot);
            for(int i=shotrange[0];i<shotrange[1];i++){
                if(find(badshot.begin(),badshot.end(),i)==badshot.end()) shotid.push_back(i);
            }
        }
    }
    
    int n1_recloc;
    from_header("recloc","n1",n1_recloc,"n2",nr);
    float *recloc=new float[n1_recloc*nr];
    read("recloc",recloc,n1_recloc*nr);

    int nt;
    float ot,dt;
    from_header("data","n1",nt,"o1",ot,"d1",dt);

    int ns1=shotid.size();
    float *souloc1=new float[5*ns1];
    vector<float> recloc1;
    int nrtotal=0;
    float *data=new float[nt];
    for(int is=0;is<ns1;is++){
        int id=shotid[is];
        int nr1=souloc[5*id+3];
        int start=souloc[5*id+4];

        fprintf(stderr,"shot %d has %d traces\n",is,nr1);
        memcpy(souloc1+5*is,souloc+5*id,3*sizeof(float));
        int nr1i=0;
        vector<int> index;
        for(int j=0;j<nr1;j++){
            int ir=start+j;
            float recx=recloc[n1_recloc*ir];
            float recy=recloc[n1_recloc*ir+1];
            if(recx>=minx && recx<=maxx && recy>=miny && recy<=maxy){
                nr1i++;
                index.push_back(j);
                for(int k=0;k<n1_recloc;k++) recloc1.push_back(recloc[n1_recloc*ir+k]);
            }
        }
        fprintf(stderr,"%d rec in range should be equal to size of index %d\n",nr1i,index.size());
        souloc1[5*is+3]=nr1i;
        souloc1[5*is+4]=nrtotal;
        nrtotal+=nr1i;

        float *data=new float[nt*nr1];
        size_t pos=(long long)nt*(long long)start;
        read("data",data,nt*nr1,pos);

        float *dataout=new float[nt*nr1i];
        for(int k=0;k<index.size();k++){
            int j=index[k];
            memcpy(dataout+nt*k,data+nt*j,nt*sizeof(float));
        }
        write("dataout",dataout,nt*nr1i,ios_base::app);

        delete []data;delete []dataout;
    }

    fprintf(stderr,"total number of rec is %d should be %d\n",nrtotal,recloc1.size()/n1_recloc);

    to_header("dataout","n1",nt,"o1",ot,"d1",dt);
    to_header("dataout","n2",nrtotal,"o2",0,"d2",1);
    
    write("soulocout",souloc1,5*ns1);
    to_header("soulocout","n1",5,"o1",0,"d1",1);
    to_header("soulocout","n2",ns1,"o2",0,"d2",1);

    write("reclocout",&recloc1[0],n1_recloc*nrtotal);
    to_header("reclocout","n1",n1_recloc,"o1",0,"d1",1);
    to_header("reclocout","n2",nrtotal,"o2",0,"d2",1);

    delete []souloc;delete[]recloc;
    delete []souloc1;

    myio_close();
    return 0;
}
