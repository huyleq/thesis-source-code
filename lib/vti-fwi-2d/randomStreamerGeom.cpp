#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include "myio.h"

using namespace std;

int main(int argc,char ** argv){
    myio_init(argc,argv);
    
    int ns,nr;
    float os,ds,zs,minoffset,dr,zr,randomness;
    get_param("nsou",ns,"osou",os,"dsou",ds);
    get_param("nrec",nr,"minoffset",minoffset,"drec",dr);
    get_param("zsou",zs,"zrec",zr);
    get_param("randomness",randomness);

    default_random_engine generator;
    uniform_real_distribution<double> distribution(0.0,1.0);
      
    vector<float> souloc;
    vector<float> recloc;

    int nssofar=0,nrsofar=0;

    for(int is=0;is<ns;is++){
        double rs=distribution(generator);
        if(rs>=randomness){
            nssofar++;
            float sx=os+is*ds;
            int nrtrue=0;
            for(int ir=0;ir<nr;ir++){
                double rr=distribution(generator);
                if(rr>=randomness){
                    nrtrue++;
                    float rx=sx+minoffset+ir*dr;
                    recloc.push_back(rx);
                    recloc.push_back(zr);
                }
            }
            souloc.push_back(sx);
            souloc.push_back(zs);
            souloc.push_back(nrtrue);
            souloc.push_back(nrsofar);
            nrsofar+=nrtrue;
            fprintf(stderr,"so far %d sous %d recs. sou %d has %d recs.\n",nssofar,nrsofar,is,nrtrue);
        }
    }
    
    if(nssofar!=souloc.size()/4) fprintf(stderr,"sth is wrong with souloc\n");

    to_header("souloc","n1",4,"o1",1.,"d1",1.);
    to_header("souloc","n2",nssofar,"o2",1.,"d1",1.);
    write("souloc",&souloc[0],souloc.size());
    
    if(nrsofar!=recloc.size()/2) fprintf(stderr,"sth is wrong with recloc\n");

    to_header("recloc","n1",2,"o1",1.,"d1",1.);
    to_header("recloc","n2",nrsofar,"o2",1.,"d1",1.);
    write("recloc",&recloc[0],recloc.size());
    
    myio_close();
    
    return 0;
}
