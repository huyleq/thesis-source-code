#include <iostream>
#include <vector>
#include <algorithm>

#include "myio.h"

using namespace std;

int main(int argc,char **argv){
    myio_init(argc,argv);
    
    long long n; get_param("nelem",n);

    float *output=new float[n];
    float *temp=new float[n];
    
    string basefilename=get_s("basefilename");

    vector<int> shotrange; get_array("shotrange",shotrange);

    for(int i=shotrange[0];i<shotrange[1];i++){
        string filename=basefilename+"_"+to_string(i)+".H";
        if(readFromHeader(filename,temp,n)){
            cout<<"summing output from file "<<filename<<endl;
            #pragma omp parallel for num_threads(16)
            for(size_t j=0;j<n;j++) output[j]+=temp[j];
        }
    }

    write("output",output,n);
    to_header("output","n1",n,"o1",0,"d1",1);
    to_header("output","n2",1,"o2",0,"d2",1);

    delete []output;delete []temp;
    
    myio_close();
    return 0;
}
