#include <iostream>
#include "seplib.h"
extern "C" {
#include "sep3d.h"
}
#include "Vector.h"
#include "randomBound.h"
using namespace std;
using namespace SEPMY;

int main(int argc, char *argv[]) {
    initpar(argc, argv);
   	
	// read parameters
	int seed, pad;
	float pctG;
	getch("seed", "d", &seed);
	getch("pad", "d", &pad);
	getch("pctG", "f",&pctG);
	cout << "Random boundary seed : " << seed << endl;
	cout << "Random boundary pad : " << pad << endl;

    // read in data
	hyperCube modelCube;
	modelCube.readAxes("modelIn");
	int modelSize = modelCube.getSize();
    float *velIn = new float[modelSize];
    cout << "vel size = " << modelSize << endl;
    sreed("modelIn", velIn, sizeof(float) * modelSize);
    cout << "Printing axis info for input " << endl;
    modelCube.axesInfo();

    // extract axis dimension
    vector<axis> axesIn = modelCube.getAxes();
    int nd = axesIn.size();
    int n1 = axesIn[0].n;
    int n2 = axesIn[1].n;
    int n3 = (nd == 3) ? axesIn[2].n : 1;


    // basically you can call the header file with the following function call
    //void RandomYM(float * field, float bval, float eval, const std::vector<int>& dims, const std::vector<int>& thick = std::vector<int>{50, 50, 50}, float pctG = 0.99, int seed = 2017);
    vector<int> dims = {n1, n2, n3};
    vector<int> thick = {pad, pad, pad};
    RandomMY(velIn, 1, 0.2, dims, thick, pctG, seed);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    //output vector to tag c .H file
    srite("modelOut", velIn, sizeof(float) * modelSize);
    modelCube.writeAxes("modelOut");
    return 0;

}
