//same as randomB.cpp but without seplib
#include <cstdlib>
#include <iostream>

#include "randomBound.h"
#include "myio.h"

using namespace std;
using namespace SEPMY;

int main(int argc, char **argv) {
    myio_init(argc, argv);
   	
	// read parameters
	int seed, pad;
	float pctG;
    get_param("seed",seed,"pad",pad,"pctG",pctG);
	cout << "Random boundary seed : " << seed << endl;
	cout << "Random boundary pad : " << pad << endl;

    // read in data
	int n1,n2,n3=1;
    float o1,o2,o3,d1,d2,d3;
    from_header("modelIn","n1",n1,"o1",o1,"d1",d1);
    from_header("modelIn","n2",n2,"o2",o2,"d2",d2);
    from_header("modelIn","n3",n3,"o3",o3,"d3",d3);
    
	size_t modelSize = n1*n2*n3;
    float *velIn = new float[modelSize];
    cout << "vel size = " << modelSize << endl;
    read("modelIn", velIn, modelSize);
    cout << "Printing axis info for input " << endl;
    cout<<"n1="<<n1<<" o1="<<o1<<" d1="<<d1<<endl;
    cout<<"n2="<<n2<<" o2="<<o2<<" d2="<<d2<<endl;
    cout<<"n3="<<n3<<" o3="<<o3<<" d3="<<d3<<endl;

    // basically you can call the header file with the following function call
    //void RandomYM(float * field, float bval, float eval, const std::vector<int>& dims, const std::vector<int>& thick = std::vector<int>{50, 50, 50}, float pctG = 0.99, int seed = 2017);
    vector<int> dims = {n1, n2, n3};
    vector<int> thick = {pad, pad, pad};
    RandomMY(velIn, 1, 0.2, dims, thick, pctG, seed);
    
    //output vector to tag c .H file
    write("modelOut", velIn, modelSize);
    to_header("modelOut","n1",n1,"o1",o1,"d1",d1);
    to_header("modelOut","n2",n2,"o2",o2,"d2",d2);
    to_header("modelOut","n3",n3,"o3",o3,"d3",d3);
    
    delete []velIn;
    myio_close();

    return 0;

}
