#####################################################################
# Chapter 3
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

FAULT=./Models

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

$(O)/%.o: $(LBFGS)/%.cpp
	$(CC) -c $< -o $@

$(O)/%.o: $(S)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(LS)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(VTI2D)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(VTI2D)/%.cu
	$(NVCC) -c $^ -o $@

$(B)/getObjfuncFromLBFGSLog.x: $(O)/getObjfuncFromLBFGSLog.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/modelResidual.x: $(O)/modelResidual.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/ricker.x: $(O)/ricker.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/uniformGeom.x: $(O)/uniformGeom.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/streamerGeom.x: $(O)/streamerGeom.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/synthetic.x: $(O)/synthetic.o $(O)/synthetic_f.o $(O)/conversions.o $(O)/kernels.o $(O)/init.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/fwi-lbfgs-c-param.x: $(O)/fwi-lbfgs-c-param.o $(O)/objFuncGradient.o $(O)/objFuncGradient_f.o $(O)/kernels.o $(O)/init.o $(O)/check.o $(O)/conversions.o $(O)/lbfgs_c.o $(O)/la.o $(O)/lsearch.o $(O)/myio.o
	$(NVCC) $^ -o $@  

$(B)/fwi-newton-param.x: $(O)/fwi-newton-param.o $(O)/objFuncGradient.o $(O)/objFuncGradient_f.o $(O)/kernels.o $(O)/init.o $(O)/check.o $(O)/conversions.o $(O)/newton.o $(O)/la.o $(O)/lsearch.o $(O)/LinearSolver.o $(O)/HessianOp.o $(O)/hessians.o $(O)/hessian_f.o $(O)/myio.o
	$(NVCC) $^ -o $@  

$(B)/fwi-newtonGN-param.x: $(O)/fwi-newtonGN-param.o $(O)/objFuncGradient.o $(O)/objFuncGradient_f.o $(O)/kernels.o $(O)/init.o $(O)/check.o $(O)/conversions.o $(O)/newton.o $(O)/la.o $(O)/lsearch.o $(O)/LinearSolver.o $(O)/HessianOp.o $(O)/hessians.o $(O)/hessian_f.o $(O)/myio.o
	$(NVCC) $^ -o $@  

$(B)/gradient-param.x: $(O)/gradient-param.o $(O)/objFuncGradient.o $(O)/objFuncGradient_f.o $(O)/kernels.o $(O)/check.o $(O)/conversions.o $(O)/init.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/rtm-param.x: $(O)/rtm-param.o $(O)/rtm_f.o $(O)/kernels.o $(O)/conversions.o $(O)/init.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/odcig-param.x: $(O)/odcig-param.o $(O)/odcig_f.o $(O)/kernels.o $(O)/conversions.o $(O)/init.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/hessian.x: $(O)/hessian.o $(O)/hessians.o $(O)/hessian_f.o $(O)/HessianOp.o $(O)/kernels.o $(O)/check.o $(O)/conversions.o $(O)/init.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/hessianMatrix.x: $(O)/hessianMatrix.o $(O)/hessians.o $(O)/hessian_f.o $(O)/HessianOp.o $(O)/kernels.o $(O)/check.o $(O)/conversions.o $(O)/init.o $(O)/myio.o
	$(NVCC) $^ -o $@

####################################################################

exe: $(B)/ricker.x $(B)/uniformGeom.x $(B)/streamerGeom.x $(B)/getObjfuncFromLBFGSLog.x $(B)/modelResidual.x $(B)/synthetic.x $(B)/fwi-lbfgs-c-param.x $(B)/fwi-newton-param.x $(B)/fwi-newtonGN-param.x $(B)/gradient-param.x $(B)/rtm-param.x $(B)/odcig-param.x $(B)/hessian.x $(B)/hessianMatrix.x 

####################################################################

ricker.5hz.fault.H: $(B)/ricker.x
	./$< par=$(P)/par.fault.p scalefactor=1e5 wavelet=$@ opt=0

ricker.20hz.fault.H: $(B)/ricker.x
	./$< par=$(P)/par.fault20.p scalefactor=1e5 wavelet=$@ opt=0

souloc.uniform.fault.H recloc.uniform.fault.H: $(B)/uniformGeom.x
	$< nsou=100 osou=100 dsou=200 zsou=12.5 nrec=800 orec=0 drec=25 zrec=25 souloc=souloc.uniform.fault.H recloc=recloc.uniform.fault.H 

data.%hz.uniform.fault.H: $(B)/synthetic.x 
	./$< par=$(P)/par.fault$*.p opt=1 souloc=souloc.uniform.fault.H recloc=recloc.uniform.fault.H wavelet=ricker.$*hz.fault.H v=$(FAULT)/Vpz.Fault.H eps=$(FAULT)/Eps.Fault.H del=$(FAULT)/Del.Fault.H data=$@ 

souloc.shortoff.fault.H recloc.shortoff.fault.H: $(B)/streamerGeom.x
	$< nsou=100 osou=50 dsou=150 zsou=12.5 minoffset=50 nrec=200 drec=25 zrec=25 souloc=souloc.shortoff.fault.H recloc=recloc.shortoff.fault.H 

data.%hz.shortoff.fault.H: $(B)/synthetic.x 
	./$< par=$(P)/par.fault$*.p opt=1 souloc=souloc.shortoff.fault.H recloc=recloc.shortoff.fault.H wavelet=ricker.$*hz.fault.H v=$(FAULT)/Vpz.Fault.H eps=$(FAULT)/Eps.Fault.H del=$(FAULT)/Del.Fault.H data=$@ 

souloc.longoff.fault.H recloc.longoff.fault.H: $(B)/streamerGeom.x
	$< nsou=100 osou=50 dsou=100 zsou=12.5 minoffset=5000 nrec=200 drec=25 zrec=25 souloc=souloc.longoff.fault.H recloc=recloc.longoff.fault.H 

data.%hz.longoff.fault.H: $(B)/synthetic.x 
	./$< par=$(P)/par.fault$*.p opt=1 souloc=souloc.longoff.fault.H recloc=recloc.longoff.fault.H wavelet=ricker.$*hz.fault.H v=$(FAULT)/Vpz.Fault.H eps=$(FAULT)/Eps.Fault.H del=$(FAULT)/Del.Fault.H data=$@ 

souloc.alloff.fault.H recloc.alloff.fault.H: $(B)/streamerGeom.x
	$< nsou=100 osou=50 dsou=100 zsou=12.5 minoffset=50 nrec=400 drec=25 zrec=25 souloc=souloc.alloff.fault.H recloc=recloc.alloff.fault.H 

data.%hz.alloff.fault.H: $(B)/synthetic.x 
	./$< par=$(P)/par.fault$*.p opt=1 souloc=souloc.alloff.fault.H recloc=recloc.alloff.fault.H wavelet=ricker.$*hz.fault.H v=$(FAULT)/Vpz.Fault.H eps=$(FAULT)/Eps.Fault.H del=$(FAULT)/Del.Fault.H data=$@ 

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

