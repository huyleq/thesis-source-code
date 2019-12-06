
####################################################################

Source code in Src
Fault model in Models

####################################################################

make exe to compile all executables

####################################################################

all inversions need:
    wavelet made with ricker.x,
    source and receiver locations made with uniformGeom.x or streamerGeom.x
    and synthetic observed data, made with synthetic.x

look at Makefile to see parameters and how to run these executables

####################################################################

Figures 1,2,4,5,6,7,14,15 are made by fwi-lbfgs-c-param.x with different parameter options

Figure 3 is made by rtm-param.x with options like following

Figures 8,9 are made by hessian.x

Figures 10 is made by hessianMatrix.x

Figures 11,12,13 are made with fwi-newton-param.x and fwi-newtonGN-param.x 

look at run.sh to see parameters and how to run these executables

####################################################################

