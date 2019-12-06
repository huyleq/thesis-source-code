#ifndef RANDYM_H
#define RANDYM_H 1
#include <exception>
#include <vector>
#include <set>
#include <algorithm>
/*
 * Random boundary generator, 2D 3D
 * To call the function, use:
 * RandomYM(field, bval, eval, dims, thicks, pctG, seed);
 * Input
 *      field: a pointer to an array that contains the physical model, velocity, density, etc
 *      bval: control the strength of randomness, fill later
 *      eval: control the strength of randomness, fill later
 *      dims: vector of integer, with size 2 or 3, depends on the dimension of the physical model
 *      thicks: vector of integer, boundary thickness, optional, default to 50 on each size
 *      pctG: float 0~1, possibility that we continue the random walk, size ~ 1 - 1/pctG for the first cluster, reduce later
 *      seed: integer, seed of random number, in case we need to regenerate the same realization of random boundary
 * Output
 *      field: the array with modified random boundary
 *
 * A brief history of random boundary:
 *   2009 Bob created the first random boundary, highly cited, algorithm worked for arbitrary dimension
 *   2013 Bob and Xukai wrote the second random boundary to introduce randomness at low frequency, 2D or 3D
 *   2015 Bob and Gustavo wrote a different implementation for elastic wave propagation, 2D
 *   2017 we extend it to 3D, to meet SEP's graduation criteria.
 *
 * Development record:
 * 	 2018-03-06 Fix bug such that vel goes to zero
 *   2018-03-08 FIx bug related to memory leakage
 *
 * First Created by Yinbin Ma, 2017-04-26
 * Last Modified by Yinbin Ma, 2018-03-08
 * Any concerns and bugs could be sent to: yinbin@stanford.edu before December, 2018. Or sent to bob@sep.stanford.edu
 */

using namespace std;
namespace SEPMY {

// main function call
void RandomMY(float * field, float bval, float eval, const std::vector<int>& dims, const std::vector<int>& thick = std::vector<int>{50, 50, 50}, float pctG = 0.99, int seed = 2017);

// function call specific to this program, to make RandomYM function more readable
void initDir(std::vector<std::vector<int> >& dir, float * PDF, int ndir);
void resetRnd(int& irand, int rndMax, float * rnd);
void initialDistanceFunction(const std::vector<int>& dims, const std::vector<int>& thick, int& countBound, int& countGrid, int * clusterLabels, float * dist, int * boundToGrid, int * gridToBound);
int firstUnusedPoint(bool * used, int * minSearchRad, float * rnd, int& irand, int rndMax, int boundSize);

// below is actual implementation

void RandomMY(float *field, float bval, float eval, const std::vector<int>& dims, const std::vector<int>& thick, float pctG, int seed) {
	srand (seed);
	if ((int)dims.size() < 3) {
		//throw std::exception("2D random boundary generator, input dimension incorrect!");
		cout << "illegal input" << endl;
		return;
	}
    int ndir = dims[2] == 1 ? 8 : 26;
    //cout << "ndir = " << ndir << endl;
    if (ndir == 26) {
    	cout << "Working on 3D random boundary" << endl;
    } else {
    	cout << "Working on 2D random boundary" << endl;
    }
    // initializting search direction
    std::vector<std::vector<int> > dir;
    float * PDF = new float[ndir];
    initDir(dir , PDF, ndir);  // initialize the direction

    // initializing temp array, I do not like this part
    int modelSize = dims[0] * dims[1] * dims[2];
    int boundSize = modelSize - (dims[0] - thick[0] * 2) * (dims[1] - thick[1] * 2) * std::max(dims[2] - thick[2] * 2, 1);
    int * clusterLabels  = new int [modelSize];
    int * boundToGrid    = new int[boundSize];
    int * gridToBound    = new int[modelSize];
    float * dist         = new float[boundSize];
	float *start         = new float[boundSize];
	float *prop          = new float[boundSize]; for (int i = 0; i < boundSize; i++) prop[i] = 1.;
    bool * used          = new bool[boundSize];
    int * minSearchRad   = new int [boundSize];
    bool triedDir[ndir];

    for (int i = 0; i < boundSize; i++) {
    	used[i] = false;
    	minSearchRad[i] = 1;
    }

    // initializing the distance function
    int n1 = dims[0]; int n2 = dims[1]; int n3 = dims[2];
    int thick1 = thick[0]; int thick2 = thick[1]; int thick3 = thick[2];
    int countBound = 0, countGrid = 0;
    initialDistanceFunction(dims, thick, countBound, countGrid, clusterLabels, dist, boundToGrid, gridToBound);

    // return if fail to precompute the boundary
    if (countBound != boundSize || countGrid != modelSize) {
		std::cout << "illogical boundary calc" << ", countBound = " << countBound << ", boundSize = " << boundSize << endl;
		std::cout << "illogical boundary calc" << ", countGrid = " << countGrid << ", modelSize = " << modelSize << endl;
		return;
    }

    // random assignment to different clusters
    int rndMax = 50000, irand = 1;
    float *rnd = new float[rndMax];
    resetRnd(irand, rndMax, rnd);
    int clusterId = 0;
    int *residualDir = new int[ndir];
    float *CDF = new float[ndir];
    int leftDirNum = 0;
    
    // loop until we fill all the data
    while (countBound > 0) {
		// Find the first point that has not been used
		int iloc = firstUnusedPoint(used, minSearchRad, rnd, irand, rndMax, boundSize);

		// now we have a point, assign it first before random walk
	    std::set<int> currentCluster; currentCluster.clear();
	    currentCluster.insert(iloc);
		used[iloc] = true;
		clusterId++;
		int i1 = boundToGrid[iloc] % n1;
		int i2 = (boundToGrid[iloc] / n1) % n2;
		int i3 = boundToGrid[iloc] / (n1 * n2);
		clusterLabels[boundToGrid[iloc]] = clusterId;
		start[clusterId] = dist[iloc];
		//cout << "Flag " << clusterId << ", countBound = " << countBound;
		countBound--;

		// now expand, the strategy is that we choose a random point, then a random direction
        bool grow = true;
        bool reselectPoint = false;
        while (grow && countBound > 0 && currentCluster.size() > 0) {
            // stop with when we are not reslecting point and probability (1-pctG)
        	if (!reselectPoint && rnd[irand] > pctG) {
        		grow = false;
        		++irand;
        		if (irand >= rndMax - 10) resetRnd(irand, rndMax, rnd);
        		continue;
        	}
        	// we decided to grow, find a point to grow
        	reselectPoint = false;
        	//int shift = std::round((currentCluster.size() - 1) * rnd[++irand]);
        	int shift = round((currentCluster.size() - 1) * rnd[++irand]);
   	        std::set<int>::const_iterator it(currentCluster.begin());
   	        //std::cout<< "first element = " << *it << ", select element = ";
   	        advance(it, shift);
   	        iloc = 	*it;
   	        //std::cout << *it << ", total size = " << currentCluster.size() << std::endl;
   	        i1 = boundToGrid[iloc] % n1;
		    i2 = (boundToGrid[iloc] / n1) % n2;
		    i3 = boundToGrid[iloc] / (n1 * n2);
   	        
        	// from the selected point, find possible dir
        	leftDirNum = 0;
        	CDF[leftDirNum] = 0;
        	for (int k = 0; k < ndir; k++) {
        		triedDir[k] = true;
    			int ia = i1 + dir[k][0];
    			int ib = i2 + dir[k][1];
    			int ic = i3 + dir[k][2];
    			if (ia >= 0 && ib >= 0 && ic >= 0 && ia < n1 && ib < n2 && ic < n3) {
    				int coord = ic * n1 * n2 + ib * n1 + ia;
    				if (clusterLabels[coord] == 0) {
    					triedDir[k] = false;
    					residualDir[leftDirNum] = k;
    					CDF[leftDirNum] = CDF[max(leftDirNum - 1, 0)] + PDF[k];
    					leftDirNum++;
    				}
    			}
        	}

        	// continue to find another point if we choose a "dead" point
        	if (leftDirNum == 0) {
        		reselectPoint = true;
        		currentCluster.erase(iloc);
        		continue;
        	}

        	//cout << "number of possible dir = " << leftDirNum << endl;
        	bool triedAll = false;
        	bool search   = true;
        	while (!triedAll && search) {
        		float prob = CDF[leftDirNum - 1] * rnd[irand++];
        		if (irand >= rndMax) resetRnd(irand, rndMax, rnd);
        		//int idir = std::round((leftDirNum - 1) * rnd[irand++]);
        		int idir = 0;
        		while (CDF[idir] < prob) idir++;
        		idir = residualDir[idir];
        		if (!triedDir[idir]) {
        			triedDir[idir] = true;
        			int ia = i1 + dir[idir][0];
        			int ib = i2 + dir[idir][1];
        			int ic = i3 + dir[idir][2];
        			if (ia >= 0 && ib >= 0 && ic >= 0 && ia < n1 && ib < n2 && ic < n3) {
        				int coord = ic * n1 * n2 + ib * n1 + ia;
        				// finally we make a legal move, move and add to the current cluster
        				if (clusterLabels[coord] == 0) {
        					clusterLabels[coord] = clusterId;
        					used[gridToBound[coord]] = true;
        					countBound--;
        					search = false;
        					i1 = ia; i2 = ib; i3 = ic;
        					int newloc = gridToBound[i3*n2*n1+i2*n1+i1];
        					currentCluster.insert(newloc);
        				}
        			}
        		}
        		triedAll = true;
            	for (int k = 0; k < ndir; k++) {
            		if (!triedDir[k]) {
            			triedAll = false;
            			break;
            		}
            	}
            	if (triedAll && search) {
            		grow = false;
            	}
        	}
        }
		//cout << ", countBound = " << countBound << endl;
    }
    //cout << "now assignment" << endl;
    
    // now assignment
    float dMax = std::max(std::max(thick[0], thick[1]), thick[2]);
    for (int i = 0; i < clusterId; i++) {
    	float pct = start[i] / dMax;
    	float dbase = (1. - pct) * bval + pct * eval;
    	bool found = false;
		if (pct == 0) {
			prop[i] = 1;
			continue;
		}
    	while (!found) {
    		float r = (float) rand() / (float) RAND_MAX;
    		r = (r - 0.5) * pct + dbase;
    		if (eval > bval) {
    			// density case
    			if (pct == 0 || (r > 1. && r < eval * 3)) {
    				found = true;
    				prop[i] = r;
    			}
    		} else {
    			if (pct == 0 || (r > eval / 3. && r < 1.)) {
    				prop[i] = r;
    				found = true;
    			}
    		}
    	}
    }

    // finally adjust the field
    for (int i3 = 0; i3 < n3; i3++) {
    	for (int i2 = 0; i2 < n2; i2++) {
    		for (int i1 = 0; i1 < n1; i1++) {
				int loc = i3 * n2 * n1 + i2 * n1 + i1;
    			if (clusterLabels[loc] > 0) {
    				field[loc] = field[loc] * prop[clusterLabels[loc]];
    			}
    		}
    	}
    }


    delete[] clusterLabels;
    delete[] boundToGrid;
    delete[] gridToBound;
    delete[] dist;
    delete[] start;
    delete[] prop;
    delete[] used;
    delete[] minSearchRad;
    delete[] rnd;
    delete[] residualDir;
    delete[] CDF;

}

void initDir(std::vector<std::vector<int> >& dir, float * PDF, int ndir) {
	int count = 0;
	if (ndir == 8) {
		float scale1 = exp(-1.);
		float scale2 = exp(-2.);
		float scale3 = exp(-3.);
		dir.push_back(std::vector<int>{-1, -1, 0}); PDF[count] = scale2; count++;
		dir.push_back(std::vector<int>{-1, 0, 0}); PDF[count] = scale1; count++;
		dir.push_back(std::vector<int>{-1, 1, 0}); PDF[count] = scale2; count++;

		dir.push_back(std::vector<int>{0, -1, 0}); PDF[count] = scale1; count++;
		dir.push_back(std::vector<int>{0, 1, 0}); PDF[count] = scale1; count++;

		dir.push_back(std::vector<int>{1, -1, 0}); PDF[count] = scale2; count++;
		dir.push_back(std::vector<int>{1, 0, 0}); PDF[count] = scale1; count++;
		dir.push_back(std::vector<int>{1, 1, 0}); PDF[count] = scale2; count++;
	}
	if (ndir == 26) {
		float scale1 = 1.; //exp(-1.);
		float scale2 = 1.; //exp(-2.);
		float scale3 = 1.; //exp(-3.);
		dir.push_back(std::vector<int>{-1, -1, -1}); PDF[count] = scale3; count++;
		dir.push_back(std::vector<int>{-1, -1, 0}); PDF[count] = scale2; count++;
		dir.push_back(std::vector<int>{-1, -1, 1}); PDF[count] = scale3; count++;

		dir.push_back(std::vector<int>{-1, 0, -1}); PDF[count] = scale2; count++;
		dir.push_back(std::vector<int>{-1, 0, 0}); PDF[count] = scale1; count++;
		dir.push_back(std::vector<int>{-1, 0, 1}); PDF[count] = scale2; count++;

		dir.push_back(std::vector<int>{-1, 1, -1}); PDF[count] = scale3; count++;
		dir.push_back(std::vector<int>{-1, 1, 0}); PDF[count] = scale2; count++;
		dir.push_back(std::vector<int>{-1, 1, 1}); PDF[count] = scale3; count++;

		dir.push_back(std::vector<int>{0, -1, -1}); PDF[count] = scale2; count++;
		dir.push_back(std::vector<int>{0, -1, 0}); PDF[count] = scale1; count++;
		dir.push_back(std::vector<int>{0, -1, 1}); PDF[count] = scale2; count++;

		dir.push_back(std::vector<int>{0, 0, -1}); PDF[count] = scale1; count++;
		dir.push_back(std::vector<int>{0, 0, 1}); PDF[count] = scale1; count++;

		dir.push_back(std::vector<int>{0, 1, -1}); PDF[count] = scale2; count++;
		dir.push_back(std::vector<int>{0, 1, 0}); PDF[count] = scale1; count++;
		dir.push_back(std::vector<int>{0, 1, 1}); PDF[count] = scale2; count++;

		dir.push_back(std::vector<int>{1, -1, -1}); PDF[count] = scale3; count++;
		dir.push_back(std::vector<int>{1, -1, 0}); PDF[count] = scale2; count++;
		dir.push_back(std::vector<int>{1, -1, 1}); PDF[count] = scale3; count++;

		dir.push_back(std::vector<int>{1, 0, -1}); PDF[count] = scale2; count++;
		dir.push_back(std::vector<int>{1, 0, 0}); PDF[count] = scale1; count++;
		dir.push_back(std::vector<int>{1, 0, 1}); PDF[count] = scale2; count++;

		dir.push_back(std::vector<int>{1, 1, -1}); PDF[count] = scale3; count++;
		dir.push_back(std::vector<int>{1, 1, 0}); PDF[count] = scale2; count++;
		dir.push_back(std::vector<int>{1, 1, 1}); PDF[count] = scale3; count++;
	}
}
void resetRnd(int& irand, int rndMax, float * rnd) {
	for (int i = 0; i < rndMax; i++) rnd[i] = (1. * rand()) / RAND_MAX;
	irand = 0;
}

void initialDistanceFunction(const std::vector<int>& dims, const  std::vector<int>& thick, int& countBound, int& countGrid, int * clusterLabels, float * dist, int * boundToGrid, int * gridToBound) {
    int n1 = dims[0]; int n2 = dims[1]; int n3 = dims[2];
    int thick1 = thick[0]; int thick2 = thick[1]; int thick3 = thick[2];
	for (int i3 = 0; i3 < n3; i3++) {
    	int in3 = std::min(std::max(i3, thick3), n3 - thick3 - 1);
    	in3 = std::min(std::max(in3, 0), n3 - 1);
    	float d3 = (i3 - in3) * (i3 - in3);
    	for (int i2 = 0; i2 < n2; i2++) {
    		int in2 = std::min(std::max(i2, thick2), n2 - thick2 - 1);
        	float d2 = (i2 - in2) * (i2 - in2);
    		for (int i1 = 0; i1 < n1; i1++) {
    			int in1 = std::min(std::max(i1, thick1), n1 - thick1 - 1);
    	    	float d1 = (i1 - in1) * (i1 - in1);
    	    	dist[countBound] = sqrt(d1 + d2 + d3);
    	    	clusterLabels[countGrid] = -1;
    	    	if (dist[countBound] > 0.01) {
    	    		clusterLabels[countGrid] = 0;
    	    		boundToGrid[countBound] = countGrid;
    	    		gridToBound[countGrid] = countBound;
    	    	    countBound++;
    	    	}
    	    	countGrid++;
    		}
    	}
    }
}

int firstUnusedPoint(bool * used, int * minSearchRad, float * rnd, int& irand, int rndMax, int boundSize) {
	//int iloc = std::round((boundSize - 1) * rnd[irand]);
	int iloc = round((boundSize - 1) * rnd[irand]);
	irand++;
	if (irand >= rndMax) resetRnd(irand, rndMax, rnd);
	if (used[iloc]) {
		bool found = false;
		int idist = minSearchRad[iloc];
		while (!found) {
			if (!used[std::max(iloc - idist, 0)]) {
				found = true;
				minSearchRad[iloc] = idist - 1;
				iloc = std::max(iloc - idist, 0);
		        break;
			}
			if (!used[std::min(iloc + idist, boundSize - 1)]) {
				found = true;
				minSearchRad[iloc] = idist - 1;
				iloc = std::min(iloc + idist, boundSize - 1);
		        break;
			}
			idist++;
		}
	}
	return iloc;
}


} // end of namespace
#endif
