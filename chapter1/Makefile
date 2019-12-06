#####################################################################
# Chapter 1
#####################################################################

CODEDIR=/homes/sep/huyle/research/thesis-source-code/lib
PLOT=$(CODEDIR)/
IO=$(CODEDIR)/io
VTI2D=$(CODEDIR)/vti-fwi-2d

FC=gfortran -Ofast -Wall -fbounds-check -g -Wno-uninitialized
CC=g++ -std=c++11 -Ofast -fopenmp -I$(IO) -I$(VTI2D)
NVCC=/usr/local/cuda/bin/nvcc --std=c++11 -Xcompiler -fopenmp -lgomp -O3 -arch=sm_70 -I$(IO) -I$(VTI2D)

S=./Src
O=./Obj
B=./Bin
P=./Par

#####################################################################

.SECONDARY:

clean:
	rm -rf $(B)/* $(O)/*

burn:
	make clean
	Rm *.H
	rm -rf $(R)/*

#####################################################################

$(O)/%.o: $(IO)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(S)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(VTI2D)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(VTI2D)/%.cu
	$(NVCC) -c $^ -o $@

$(B)/%Geom.x: $(O)/%Geom.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/model%.x: $(O)/model%.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/ricker.x: $(O)/ricker.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/synthetic.x: $(O)/synthetic.o $(O)/synthetic_f.o $(O)/conversions.o $(O)/kernels.o $(O)/init.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/test-nmo.x: $(O)/test-nmo.o $(O)/processing.o $(O)/myio.o
	$(CC) $^ -o $@ 

####################################################################

exe: $(B)/model1.x $(B)/streamerGeom.x $(B)/ricker.x $(B)/synthetic.x $(B)/test-nmo.x

####################################################################

v.1.H eps.1.H del.1.H: $(B)/model1.x $(P)/par.1.p
	./$< par=$(P)/par.1.p v1=1500 d1=500 v2=1700 v3=2200 d2=1500 v=v.1.H
	./$< par=$(P)/par.1.p v1=0 d1=500 v2=0.35 v3=0.35 d2=1500 v=eps.1.H
	./$< par=$(P)/par.1.p v1=0 d1=500 v2=0.1 v3=0.1 d2=1500 v=del.1.H

souloc.oneshot.1.H recloc.oneshot.1.H: $(B)/streamerGeom.x $(P)/par.1.p
	$< par=$(P)/par.1.p souloc=souloc.oneshot.1.H recloc=recloc.oneshot.1.H 

ricker10hz.H: $(B)/ricker.x
	./$< par=$(P)/par.1.p opt=0 wavelet=$@

# data.oneshot.1.10.H for Figure 1.2(b) 
data.oneshot.1.%.H: $(B)/synthetic.x $(P)/par.1.p v.1.H souloc.oneshot.1.H 
	./$< par=$(P)/par.1.p souloc=souloc.oneshot.1.H recloc=recloc.oneshot.1.H wavelet=ricker10hz.H parameter=vepsdel v=v.1.H eps=eps.1.H del=del.1.H data=$@

# anmo-prestack.H for Figures 1.3 and 1.4
# anmo-sem.H for Figures 1.5
test-nmo: $(B)/test-nmo.x $(P)/par.1.p
	./$< par=$(P)/par.1.p v=v.1.H vrms=vrms.1.H dataout=anmo-prestack.H data=data.oneshot.1.10.H souloc=souloc.oneshot.1.H recloc=recloc.oneshot.1.H minv=1500 maxv=1900 dv=10. mineta=0 maxeta=0.3 deta=0.01 semblance=anmo-sem.H halfwindow=20

####################################################################

%.pdf: %.v
	pstexpen $*.v $*.ps fat=1 fatmult=1.5 invras=n force=  invras=y color=y
	ps2pdf \
	-dDownsampleColorImages=false \
	-dDownsampleGrayImages=false \
	-dDownsampleMonoImages=false \
	-dColorImageFilter=/FlateEncode \
	-dGrayImageFilter=/FlateEncode \
	-dMonoImageFilter=/CCITTFaxEncode \
	-dAutoFilterColorImages=false \
	-dAutoFilterGrayImages=false \
	-dAutoFilterMonoImages=false \
	-dEPSCrop $*.ps $*.pdf
	rm -f $*.ps

