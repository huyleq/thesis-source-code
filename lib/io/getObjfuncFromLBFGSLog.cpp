#include <cstdio>
#include <vector>

#include "myio.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
    string logfile=get_s("logfile");
    
    float init_obj;
    get_first("logfile","f",init_obj);
//    fprintf(stderr,"initial objfunc %.10f\n",init_obj);

    vector<int> iter;
    get_all("logfile","ITER",iter);
    int niter=iter.back();
    fprintf(stderr,"total # of iterates %d\n",niter);

    vector<float> obj;
    obj.push_back(init_obj);
    get_all("logfile","F",obj);

    vector<int> nfun;
    get_all("logfile","NFUN",nfun);

    if(obj.size()==niter+1 && nfun.size()==niter) fprintf(stderr,"size of obj and nfun is equal niter\n");
    else fprintf(stderr,"something is wrong\n");

//    for(int i=0;i<niter;i++) fprintf(stderr,"iterate %d after %d has obj %.10f\n",i+1,nfun[i],obj[i]);
    write("objfunc",&obj[0],niter+1);
    to_header("objfunc","n1",niter+1,"o1",0,"d1",1);

    myio_close();
    return 0;
}
