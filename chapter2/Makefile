#####################################################################
# Chapter 2
#####################################################################

CODEDIR=/homes/sep/huyle/research/thesis-source-code/lib
PLOT=$(CODEDIR)/
IO=$(CODEDIR)/io
VTI2D=$(CODEDIR)/vti-fwi-2d
LS=$(CODEDIR)/LinearSolver
LBFGS=$(CODEDIR)/lbfgs

CC=g++ -std=c++11 -Ofast -fopenmp -I$(IO) -I$(VTI2D) -I$(LS) -I$(LBFGS)
NVCC=/usr/local/cuda/bin/nvcc --std=c++11 -Xcompiler -fopenmp -lgomp -O3 -arch=sm_70 -I$(IO) -I$(VTI2D) -I$(LS) -I$(LBFGS)
ISPC=$(HOME)/ispc/ispc-v1.9.1-linux/ispc

S=./Src
O=./Obj
B=./Bin
P=./Par

BP2007=$(HOME)/BP2007

#####################################################################

.SECONDARY:

clean:
	rm -rf $(B)/* $(O)/*

burn:
	make clean
	Rm *.H
	rm -rf $(R)/*

#####################################################################

$(O)/%.o: $(S)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(S)/%.ispc
	$(ISPC) $^ -o $@

$(O)/%.o: $(IO)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(VTI2D)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(VTI2D)/%.cu
	$(NVCC) -c $^ -o $@

$(O)/%.o: $(LS)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(LBFGS)/%.cpp
	$(CC) -c $^ -o $@

$(B)/ricker.x: $(O)/ricker.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/uniformGeom.x: $(O)/uniformGeom.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/modeling.x: $(O)/modeling.o $(O)/modeling_f.o $(O)/conversions.o $(O)/kernels.o $(O)/init.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/synthetic.x: $(O)/synthetic.o $(O)/synthetic_f.o $(O)/conversions.o $(O)/kernels.o $(O)/init.o $(O)/myio.o
	$(NVCC) $^ -o $@

ricker.%hz.bp.H: $(B)/ricker.x
	./$< par=$(P)/par.bp.p freq=$* tdelay=0.08 scalefactor=1e5 wavelet=$@ opt=0 

souloc.uniform.bp.H recloc.uniform.bp.H: $(B)/uniformGeom.x
	$< nsou=100 osou=100 dsou=200 zsou=12.5 nrec=800 orec=0 drec=25 zrec=25 souloc=souloc.uniform.bp.H recloc=recloc.uniform.bp.H 

data.10hz.uniform.bp.H: $(B)/synthetic.x 
	./$< par=$(P)/par.bp.p opt=1 souloc=souloc.uniform.bp.H recloc=recloc.uniform.bp.H wavelet=ricker.10hz.bp.H v=$(BP)/v.bp2007.H eps=$(BP)/eps.bp2007.H del=$(BP)/del.bp2007.H data=$@ 

wavefield.15hz.bp.%.H: $(B)/modeling.x 
	./$< par=$(P)/par.bp.p system=$* wavelet=ricker.15hz.bp.H v=$(BP)/v.bp2007.H eps=$(BP)/eps.bp2007.H del=$(BP)/del.bp2007.H wavefield=$@ 

$(B)/vtifd3d8o-forward.x: $(O)/vtifd3d8o-forward.o $(O)/laplacian3d.o $(O)/myio.o
	$(CC) $^ -ltbb -std=c++11 -o $@

$(B)/fwi-lbfgs-c-param.x: $(O)/fwi-lbfgs-c-param.o $(O)/objFuncGradient.o $(O)/objFuncGradient_f.o $(O)/kernels.o $(O)/init.o $(O)/check.o $(O)/conversions.o $(O)/lbfgs_c.o $(O)/la.o $(O)/lsearch.o $(O)/myio.o
	$(NVCC) $^ -o $@  

####################################################################

exe: $(B)/uniformGeom.x $(B)/ricker.x $(B)/synthetic.x $(B)/modeling.x $(B)/fwi-lbfgs-c-param.x $(B)/vtifd3d8o-forward.x

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

