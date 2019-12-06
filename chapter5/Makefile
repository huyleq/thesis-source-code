#####################################################################
# Chapter 5
#####################################################################

CODEDIR=/homes/sep/huyle/research/thesis-source-code/lib
PLOT=$(CODEDIR)/
IO=$(CODEDIR)/io
VTI2D=$(CODEDIR)/vti-fwi-2d
LS=$(CODEDIR)/LinearSolver
LBFGS=$(CODEDIR)/lbfgs
LBFGSB=$(CODEDIR)/lbfgs/Lbfgsb.3.0
VTI3DCPU=$(CODEDIR)/vti-fwi-3d/cpu
VTI3DGPU=$(CODEDIR)/vti-fwi-3d/gpu
RANDB=$(CODEDIR)/random-boundary
NETWORK=$(CODEDIR)/network

S=./Src
O=./Obj
B=./Bin
P=./Par
M=./models
D=./data
W=./wavelets
I=./images

FC=gfortran -Ofast -Wall -fbounds-check -g -Wno-uninitialized
CC=g++ -std=c++11 -Ofast -fopenmp -I$(IO) -I$(VTI2D) -I$(VTI3DCPU) -I$(VTI3DGPU) -I$(LBFGS) -I$(LS) -I$(NETWORK) -I$(RANDB) 
NVCC=/usr/local/cuda/bin/nvcc --std=c++11 -D_FORCE_INLINES -Xcompiler -fopenmp -lgomp -O3 -arch=sm_70 -I$(IO) -I$(VTI2D) -I$(VTI3DCPU) -I$(VTI3DGPU) -I$(LBFGS) -I$(LS) -I$(NETWORK) -I$(RANDB) -lssh
ISPC=$(HOME)/ispc/ispc-v1.9.1-linux/ispc

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

$(O)/%.o: $(IO)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(LBFGSB)/%.f
	$(FC) -c $< -o $@

$(O)/%.o: $(LS)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(VTI2D)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(VTI2D)/%.cu
	$(NVCC) -c $^ -o $@

$(O)/%.o: $(VTI3DCPU)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(VTI3DCPU)/%.ispc
	$(ISPC) -O3 --target=avx $< -o $@ 

$(O)/%.o: $(VTI3DGPU)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(VTI3DGPU)/%.cu
	$(NVCC) -c $^ -o $@

$(O)/%.o: $(RANDB)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(NETWORK)/%.cpp
	$(CC) -c $^ -o $@

####################################################################

$(B)/apply_gaussian_filter.x: $(O)/apply_gaussian_filter.o $(O)/gaussian_filter.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/job_calc.x: $(O)/job_calc.o $(O)/cluster.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/sort-trace.x: $(O)/sort-trace.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/clip.x: $(O)/clip.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/correctBounds.x: $(O)/correctBounds.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/checkBound3d.x: $(O)/checkBound3d.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/checkBoundLoc3d.x: $(O)/checkBoundLoc3d.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/firstArrivalMute.x: $(O)/firstArrivalMute.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/salt_mask.x: $(O)/salt_mask.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/putVSalt.x: $(O)/putVSalt.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/tpow.x: $(O)/tpow.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/trace-normalize.x: $(O)/trace-normalize.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/ricker.x: $(O)/ricker.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/fwi3d-lbfgsb-cluster.x: $(O)/fwi3d-lbfgsb-cluster.o $(O)/cluster.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/windowShot3d.x: $(O)/windowShot3d.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/residualAGCShot.x: $(O)/residualAGCShot.o $(O)/agc.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/applyAGC.x: $(O)/applyAGC.o $(O)/agc.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/getboundary.x: $(O)/getboundary.o $(O)/boundary.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/putboundary.x: $(O)/putboundary.o $(O)/boundary.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/randomBound.x: $(O)/randomBound.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/processimage.x: $(O)/processimage.o $(O)/boundary.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/modelData3d.x: $(O)/modelData3d.o $(O)/modelData3d_f.o $(O)/kernel3d.o $(O)/injectRecord3d.o $(O)/abc3d.o $(O)/memcpy3d.o $(O)/conversions.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/rtm3d.x: $(O)/rtm3d.o $(O)/rtm3d_f.o $(O)/kernel3d.o $(O)/injectRecord3d.o $(O)/abc3d.o $(O)/memcpy3d.o $(O)/conversions.o $(O)/boundary.o $(O)/check.o $(O)/cluster.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/rtm3d-cij-shot.x: $(O)/rtm3d-cij-shot.o $(O)/rtm3d_f.o $(O)/kernel3d.o $(O)/injectRecord3d.o $(O)/abc3d.o $(O)/memcpy3d.o $(O)/conversions.o $(O)/boundary.o $(O)/check.o $(O)/cluster.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/rtm3d-vepsdel-cluster.x: $(O)/rtm3d-vepsdel-cluster.o $(O)/rtm3d_f.o $(O)/kernel3d.o $(O)/injectRecord3d.o $(O)/abc3d.o $(O)/memcpy3d.o $(O)/conversions.o $(O)/boundary.o $(O)/check.o $(O)/cluster.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/sourceInv-cg.x: $(O)/sourceInv-cg.o $(O)/ModelingOp3d.o $(O)/modelData3d_f.o $(O)/kernel3d.o $(O)/injectRecord3d.o $(O)/abc3d.o $(O)/memcpy3d.o $(O)/conversions.o $(O)/boundary.o $(O)/check.o $(O)/LinearSolver.o $(O)/myio.o
	$(NVCC) $^ -o $@ 

$(B)/runScriptRemote.x: $(O)/runScriptRemote.o $(O)/sshtunneling.o 
	$(CC) $^ -o $@ -lssh

$(B)/objFuncGradient3d-cij-cpu.x: $(O)/objFuncGradient3d-cij-cpu.o $(O)/laplacian3d.o $(O)/boundary.o $(O)/conversions.o $(O)/agc.o $(O)/myio.o
	$(CC) $^ -ltbb -std=c++11 -o $@

$(B)/objFuncGradient3d-cij-cluster.x: $(O)/objFuncGradient3d-cij-cluster.o $(O)/cluster.o $(O)/cees_job_submit.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/objFuncGradient3d-vepsdel-jarvis-cees.x: $(O)/objFuncGradient3d-vepsdel-jarvis-cees.o $(O)/objFuncGradient3d_f.o $(O)/objFuncGradientCij3d_network.o $(O)/kernel3d.o $(O)/injectRecord3d.o $(O)/abc3d.o $(O)/memcpy3d.o $(O)/conversions.o $(O)/boundary.o $(O)/check.o $(O)/cluster.o $(O)/sshtunneling.o $(O)/agc.o $(O)/gaussian_filter.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/fwi3d-lbfgsb-jarvis-cees.x: $(O)/fwi3d-lbfgsb-jarvis-cees.o $(O)/objFuncGradient3d_f.o $(O)/objFuncGradientCij3d_network.o $(O)/kernel3d.o $(O)/injectRecord3d.o $(O)/abc3d.o $(O)/memcpy3d.o $(O)/conversions.o $(O)/boundary.o $(O)/check.o $(O)/cluster.o $(O)/sshtunneling.o $(O)/agc.o $(O)/gaussian_filter.o $(O)/lbfgsb.o $(O)/minilinpack.o $(O)/timer.o $(O)/miniblas.o $(O)/myio.o
	$(NVCC) $^ -o $@ -lgfortran

$(B)/modeling3d.x: $(O)/modeling3d.o $(O)/modeling3d_f.o $(O)/kernel3d.o $(O)/injectRecord3d.o $(O)/abc3d.o $(O)/memcpy3d.o $(O)/conversions.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/odcig3d.x: $(O)/odcig3d.o $(O)/odcig3d_f.o $(O)/kernel3d.o $(O)/injectRecord3d.o $(O)/abc3d.o $(O)/memcpy3d.o $(O)/conversions.o $(O)/boundary.o $(O)/check.o $(MYIO)
	$(NVCC) $^ -o $@

##################################################################

exe: $(B)/apply_gaussian_filter.x $(B)/job_calc.x $(B)/sort-trace.x $(B)/clip.x $(B)/correctBounds.x $(B)/checkBound3d.x $(B)/checkBoundLoc3d.x $(B)/firstArrivalMute.x $(B)/salt_mask.x $(B)/putVSalt.x $(B)/tpow.x $(B)/trace-normalize.x $(B)/residualAGCShot.x $(B)/applyAGC.x $(B)/getboundary.x $(B)/putboundary.x $(B)/randomBound.x $(B)/processimage.x $(B)/modeling3d.x $(B)/modelData3d.x $(B)/rtm3d.x $(B)/rtm3d-cij-shot.x $(B)/rtm3d-vepsdel-cluster.x $(B)/sourceInv-cg.x $(B)/runScriptRemote.x $(B)/objFuncGradient3d-cij-cpu.x $(B)/objFuncGradient3d-cij-cluster.x $(B)/objFuncGradient3d-vepsdel-jarvis-cees.x $(B)/fwi3d-lbfgsb-jarvis-cees.x

##################################################################

ricker.dragon3d..H: $(B)/ricker.x 
	./$< freq=15 scalefactor=1e6 tdelay=0.08 opt=0 wavelet=$@

paddragon:
	Pad < dragon.vp.rtm.H beg1=28 beg2=28 beg3=28 end1=28 end2=28 end3=28 extend=1 > dragon.vp.rtm.pad.H
	Pad < dragon.eps.rtm.H beg1=28 beg2=28 beg3=28 end1=28 end2=28 end3=28 extend=1 > dragon.eps.rtm.pad.H
	Pad < dragon.del.rtm.H beg1=28 beg2=28 beg3=28 end1=28 end2=28 end3=28 extend=1 > dragon.del.rtm.pad.H
	$(WORK)/sharedLibrary/randomB.x modelIn=dragon.vp.rtm.pad.H seed=20 pad=28 pctG=0.97 modelOut=dragon.vp.rtm.random.H

boundarydragon: $(B)/getboundary.x $(P)/par.dragon3d.p
	./$< par=$(P)/par$*.p v=dragon.vp.rtm.pad.H boundary=padboundary.dragon.vp.rtm.H
	./$< par=$(P)/par$*.p v=dragon.vp.rtm.random.H boundary=randboundary.dragon.vp.rtm.H

##################################################################

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

