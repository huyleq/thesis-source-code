#####################################################################
# Appendix C
#####################################################################

CODEDIR=/homes/sep/huyle/research/thesis-source-code/lib
IO=$(CODEDIR)/io
VTI2D=$(CODEDIR)/vti-fwi-2d
ELAS2D=$(CODEDIR)/elastic2d

CC=g++ -std=c++11 -Ofast -fopenmp -I$(IO) -I$(VTI2D) -I$(ELAS2D) 
NVCC=/usr/local/cuda/bin/nvcc --std=c++11 -Xcompiler -fopenmp -lgomp -O3 -arch=sm_70 -I$(IO) -I$(VTI2D) -I$(ELAS2D)

S=./Src
O=./Obj
B=./Bin
P=./Par

BP2007=$(HOME)/BP2007
HESS=$(HOME)/Hess

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

$(O)/%.o: $(VTI2D)/%.cpp
	$(CC) -c $^ -o $@

$(O)/%.o: $(VTI2D)/%.cu
	$(NVCC) -c $^ -o $@

$(O)/%.o: $(ELAS2D)/%.cpp
	$(CC) -c $^ -o $@

$(B)/ricker.x: $(O)/ricker.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/uniformGeom.x: $(O)/uniformGeom.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/elastic_modeling.x: $(O)/elastic_modeling.o $(O)/elastic_modeling_f.o $(O)/elastic.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/viscoelastic_modeling.x: $(O)/viscoelastic_modeling.o $(O)/viscoelastic_modeling_f.o $(O)/elastic.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/elastic_synthetic.x: $(O)/elastic_synthetic.o  $(O)/elastic_synthetic_f.o $(O)/elastic.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/viscoelastic_synthetic.x: $(O)/viscoelastic_synthetic.o $(O)/viscoelastic_synthetic_f.o $(O)/elastic.o $(O)/myio.o
	$(CC) $^ -o $@

$(B)/synthetic.x: $(O)/synthetic.o $(O)/synthetic_f.o $(O)/conversions.o $(O)/kernels.o $(O)/init.o $(O)/myio.o
	$(NVCC) $^ -o $@

$(B)/vp2vsrho.x: $(O)/vp2vsrho.o $(O)/myio.o
	$(CC) $^ -o $@

####################################################################

exe: $(B)/ricker.x $(B)/uniformGeom.x $(B)/elastic_modeling.x $(B)/viscoelastic_modeling.x $(B)/elastic_synthetic.x $(B)/viscoelastic_synthetic.x $(B)/synthetic.x $(B)/vp2vsrho.x

####################################################################

ricker.10Hz.0.H: $(B)/ricker.x
	./$< par=$(P)/par0.p wavelet=$@

vp.H:
	Spike n1=200 n2=200 o1=0 o2=0 d1=12.5 d2=12.5 mag=2000 > vp.H
	Spike n1=200 n2=200 o1=0 o2=0 d1=12.5 d2=12.5 mag=1200 > vs.H
	Spike n1=200 n2=200 o1=0 o2=0 d1=12.5 d2=12.5 mag=1000 > rho.H
	Spike n1=200 n2=200 o1=0 o2=0 d1=12.5 d2=12.5 mag=0.2 > eps.0.H
	Spike n1=200 n2=200 o1=0 o2=0 d1=12.5 d2=12.5 mag=-0.2 > eps.1.H
	Spike n1=200 n2=200 o1=0 o2=0 d1=12.5 d2=12.5 mag=1 > del.0.H
	Spike n1=200 n2=200 o1=0 o2=0 d1=12.5 d2=12.5 mag=0.6 > del.1.H
	Spike n1=200 n2=200 o1=0 o2=0 d1=12.5 d2=12.5 mag=0.1 > del.2.H
	Spike n1=200 n2=200 o1=0 o2=0 d1=12.5 d2=12.5 mag=-0.1 > del.3.H
	Spike n1=200 n2=200 o1=0 o2=0 d1=12.5 d2=12.5 mag=-0.3 > del.4.H

Qinv.H:
	Spike n1=200 n2=200 o1=0 o2=0 d1=12.5 d2=12.5 mag=0.02 > $@

sx_wfld.00.H: $(B)/elastic_modeling.x
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.0.H del=del.0.H sx_wfld=sx_wfld.00.H sz_wfld=sz_wfld.00.H sxz_wfld=sxz_wfld.00.H vx_wfld=vx_wfld.00.H vz_wfld=vz_wfld.00.H 
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.0.H del=del.2.H sx_wfld=sx_wfld.02.H sz_wfld=sz_wfld.02.H sxz_wfld=sxz_wfld.02.H vx_wfld=vx_wfld.02.H vz_wfld=vz_wfld.02.H 
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.0.H del=del.4.H sx_wfld=sx_wfld.04.H sz_wfld=sz_wfld.04.H sxz_wfld=sxz_wfld.04.H vx_wfld=vx_wfld.04.H vz_wfld=vz_wfld.04.H 
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.1.H del=del.1.H sx_wfld=sx_wfld.11.H sz_wfld=sz_wfld.11.H sxz_wfld=sxz_wfld.11.H vx_wfld=vx_wfld.11.H vz_wfld=vz_wfld.11.H 
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.1.H del=del.3.H sx_wfld=sx_wfld.13.H sz_wfld=sz_wfld.13.H sxz_wfld=sxz_wfld.13.H vx_wfld=vx_wfld.13.H vz_wfld=vz_wfld.13.H 
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.1.H del=del.4.H sx_wfld=sx_wfld.14.H sz_wfld=sz_wfld.14.H sxz_wfld=sxz_wfld.14.H vx_wfld=vx_wfld.14.H vz_wfld=vz_wfld.14.H 

sx_wfld.a.00.H: $(B)/viscoelastic_modeling.x
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.0.H del=del.0.H Qinv=Qinv.H gbar=50 g=0.03029737,0.01856114,0.03029737 omega=6.28318531,34.41442326,188.49555922 sx_wfld=sx_wfld.a.00.H sz_wfld=sz_wfld.a.00.H sxz_wfld=sxz_wfld.a.00.H vx_wfld=vx_wfld.a.00.H vz_wfld=vz_wfld.a.00.H 
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.0.H del=del.2.H Qinv=Qinv.H gbar=50 g=0.03029737,0.01856114,0.03029737 omega=6.28318531,34.41442326,188.49555922 sx_wfld=sx_wfld.a.02.H sz_wfld=sz_wfld.a.02.H sxz_wfld=sxz_wfld.a.02.H vx_wfld=vx_wfld.a.02.H vz_wfld=vz_wfld.a.02.H 
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.0.H del=del.4.H Qinv=Qinv.H gbar=50 g=0.03029737,0.01856114,0.03029737 omega=6.28318531,34.41442326,188.49555922 sx_wfld=sx_wfld.a.04.H sz_wfld=sz_wfld.a.04.H sxz_wfld=sxz_wfld.a.04.H vx_wfld=vx_wfld.a.04.H vz_wfld=vz_wfld.a.04.H 
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.1.H del=del.1.H Qinv=Qinv.H gbar=50 g=0.03029737,0.01856114,0.03029737 omega=6.28318531,34.41442326,188.49555922 sx_wfld=sx_wfld.a.11.H sz_wfld=sz_wfld.a.11.H sxz_wfld=sxz_wfld.a.11.H vx_wfld=vx_wfld.a.11.H vz_wfld=vz_wfld.a.11.H 
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.1.H del=del.3.H Qinv=Qinv.H gbar=50 g=0.03029737,0.01856114,0.03029737 omega=6.28318531,34.41442326,188.49555922 sx_wfld=sx_wfld.a.13.H sz_wfld=sz_wfld.a.13.H sxz_wfld=sxz_wfld.a.13.H vx_wfld=vx_wfld.a.13.H vz_wfld=vz_wfld.a.13.H 
	$< par=$(P)/par0.p wavelet=ricker.10Hz.0.H parameter=vepsdel vp=vp.H vs=vs.H rho=rho.H eps=eps.1.H del=del.4.H Qinv=Qinv.H gbar=50 g=0.03029737,0.01856114,0.03029737 omega=6.28318531,34.41442326,188.49555922 sx_wfld=sx_wfld.a.14.H sz_wfld=sz_wfld.a.14.H sxz_wfld=sxz_wfld.a.14.H vx_wfld=vx_wfld.a.14.H vz_wfld=vz_wfld.a.14.H 

%.500ms.H: %.H
	< $*.H Window3d n3=1 f3=50 > $@

compare_%_wfld.H: %_wfld.00.500ms.H %_wfld.02.500ms.H %_wfld.04.500ms.H %_wfld.11.500ms.H %_wfld.13.500ms.H %_wfld.14.500ms.H
	Cat3d $^ axis=3 > $@

compare_%_wfld.a.H: %_wfld.a.00.500ms.H %_wfld.a.02.500ms.H %_wfld.a.04.500ms.H %_wfld.a.11.500ms.H %_wfld.a.13.500ms.H %_wfld.a.14.500ms.H
	Cat3d $^ axis=3 > $@

compare_all_wfld.H: compare_sx_wfld.H compare_sz_wfld.H compare_sxz_wfld.H compare_vx_wfld.H compare_vz_wfld.H
	Math file1=compare_vx_wfld.H exp='file1*75e4' > compare_vx_wfld_scaled.H
	Math file1=compare_vz_wfld.H exp='file1*75e4' > compare_vz_wfld_scaled.H
	Cat3d compare_sx_wfld.H compare_sz_wfld.H compare_sxz_wfld.H compare_vx_wfld_scaled.H compare_vz_wfld_scaled.H axis=1 > $@

####################################################################

ricker.10Hz.1.H: $(B)/ricker.x
	./$< par=$(P)/par1.p wavelet=$@

souloc.uniform.1.H recloc.uniform.1.H: $(B)/uniformGeom.x
	$< par=$(P)/par1.p souloc=souloc.uniform.1.H recloc=recloc.uniform.1.H

vp1.H:
	Spike n1=200 n2=16 o1=0 o2=0 d1=12.5 d2=12.5 mag=1500 > a.H
	Spike n1=200 n2=144 o1=0 o2=0 d1=12.5 d2=12.5 mag=2000 > b.H
	Spike n1=200 n2=40 o1=0 o2=0 d1=12.5 d2=12.5 mag=2500 > c.H
	Cat3d a.H b.H c.H axis=2 > abc.H
	Pad < abc.H beg1=28 beg2=28 end1=28 end2=28 extend=1 > vp1.H
	Rm a.H b.H c.H
	Spike n1=200 n2=16 o1=0 o2=0 d1=12.5 d2=12.5 mag=0 > a.H
	Spike n1=200 n2=184 o1=0 o2=0 d1=12.5 d2=12.5 mag=1200 > b.H
	Cat3d a.H b.H axis=2 > ab.H
	Pad < ab.H beg1=28 beg2=28 end1=28 end2=28 extend=1 > vs1.H
	Rm b.H ab.H
	Spike n1=256 n2=256 o1=-350 o2=-350 d1=12.5 d2=12.5 mag=1000 > rho1.H
	Spike n1=200 n2=184 o1=0 o2=0 d1=12.5 d2=12.5 mag=0.2 > b.H
	Cat3d a.H b.H axis=2 > ab.H
	Pad < ab.H beg1=28 beg2=28 end1=28 end2=28 extend=1 > eps1.H
	Rm b.H ab.H
	Spike n1=200 n2=184 o1=0 o2=0 d1=12.5 d2=12.5 mag=0.1 > b.H
	Cat3d a.H b.H axis=2 > ab.H
	Pad < ab.H beg1=28 beg2=28 end1=28 end2=28 extend=1 > del1.H
	Rm b.H ab.H
	Spike n1=256 n2=256 o1=-350 o2=-350 d1=12.5 d2=12.5 mag=0.02 > Qinv1.H

pdata.1.H: $(B)/elastic_synthetic.x	
	$< par=$(P)/par1.p parameter=vepsdel vp=vp1.H vs=vs1.H rho=rho1.H eps=eps1.H del=del1.H wavelet=ricker.10Hz.1.H souloc=souloc.uniform.1.H recloc=recloc.uniform.1.H pdata=pdata.1.H vxdata=vxdata.1.H vzdata=vzdata.1.H

pdata.1.a.H: $(B)/viscoelastic_synthetic.x	
	$< par=$(P)/par1.p parameter=vepsdel vp=vp1.H vs=vs1.H rho=rho1.H eps=eps1.H del=del1.H Qinv=Qinv1.H gbar=50 g=0.03029737,0.01856114,0.03029737 omega=6.28318531,34.41442326,188.49555922 wavelet=ricker.10Hz.1.H souloc=souloc.uniform.1.H recloc=recloc.uniform.1.H pdata=pdata.1.a.H vxdata=vxdata.1.a.H vzdata=vzdata.1.a.H

data.1.H: $(B)/synthetic.x 
	./$< par=$(P)/par1.p parameter=vepsdel souloc=souloc.uniform.1.H recloc=recloc.uniform.1.H wavelet=ricker.10Hz.1.H v=vp1.H eps=eps1.H del=del1.H data=$@ gpu=0,1,2,3,4,5

####################################################################

ricker.10Hz.hess.acoustic.H: $(B)/ricker.x
	./$< par=$(P)/par.hess.p freq=10. scalefactor=1e5 tdelay=0.12 wavelet=$@

ricker.10Hz.hess.elastic.H: $(B)/ricker.x
	./$< par=$(P)/par.hess.p freq=10. scalefactor=1e3 tdelay=0.12 wavelet=$@

ricker1.10Hz.hess.elastic.H: $(B)/ricker.x
	./$< par=$(P)/par.hess.p opt=1 freq=10. scalefactor=1e3 tdelay=0.12 wavelet=$@

souloc.uniform.hess.H recloc.uniform.hess.H: $(B)/uniformGeom.x
	$< par=$(P)/par.hess.p souloc=souloc.uniform.hess.H recloc=recloc.uniform.hess.H

data.hess.acoustic.H: $(B)/synthetic.x 
	./$< par=$(P)/par.hess.p opt=1 souloc=souloc.uniform.hess.H recloc=recloc.uniform.hess.H wavelet=ricker.10Hz.hess.acoustic.H v=$(HESS)/v.hess.small.H eps=$(HESS)/eps.hess.small.H del=$(HESS)/del.hess.small.H data=$@ gpu=0

data.hess.acoustic.ricker1.H: $(B)/synthetic.x 
	./$< par=$(P)/par.hess.p opt=1 souloc=souloc.uniform.hess.H recloc=recloc.uniform.hess.H wavelet=ricker1.10Hz.hess.elastic.H v=$(HESS)/v.hess.small.H eps=$(HESS)/eps.hess.small.H del=$(HESS)/del.hess.small.H data=$@ gpu=0

tdata.hess.acoustic.H: $(B)/synthetic.x 
	./$< par=$(P)/par.hess.p opt=1 souloc=souloc.uniform.hess.H recloc=recloc.uniform.hess.H wavelet=ricker.10Hz.hess.acoustic.H v=$(HESS)/v.hess.small.H eps=$(HESS)/eps.hess.small.H del=$(HESS)/del.hess.small.H data=$@ gpu=0

pdata.hess.acoustic.H: $(B)/elastic_synthetic.x	
	$< par=$(P)/par.hess.p parameter=vepsdel souloc=souloc.uniform.hess.H recloc=recloc.uniform.hess.H wavelet=ricker.10Hz.hess.elastic.H vp=$(HESS)/v.hess.small.pad.H eps=$(HESS)/eps.hess.small.pad.H del=$(HESS)/del.hess.small.pad.H vs=$(HESS)/zero.hess.small.pad.H rho=$(HESS)/constrho.hess.small.pad.H pdata=$@ vxdata=vxdata.hess.acoustic.H vzdata=vzdata.hess.acoustic.H

pdata.hess.elastic.H: $(B)/elastic_synthetic.x	
	$< par=$(P)/par.hess.p parameter=vepsdel souloc=souloc.uniform.hess.H recloc=recloc.uniform.hess.H wavelet=ricker.10Hz.hess.elastic.H vp=$(HESS)/v.hess.small.pad.H eps=$(HESS)/eps.hess.small.pad.H del=$(HESS)/del.hess.small.pad.H vs=$(HESS)/vs.hess.small.pad.H rho=$(HESS)/rho.hess.small.pad.H pdata=$@ vxdata=vxdata.hess.elastic.H vzdata=vzdata.hess.elastic.H

pdata.hess.viscoelastic.H: $(B)/viscoelastic_synthetic.x	
	$< par=$(P)/par.hess.p parameter=vepsdel souloc=souloc.uniform.hess.H recloc=recloc.uniform.hess.H wavelet=ricker.10Hz.hess.elastic.H vp=$(HESS)/v.hess.small.pad.H eps=$(HESS)/eps.hess.small.pad.H del=$(HESS)/del.hess.small.pad.H vs=$(HESS)/vs.hess.small.pad.H rho=$(HESS)/rho.hess.small.pad.H Qinv=$(HESS)/Qinv.hess.small.pad.H gbar=50 g=0.03029737,0.01856114,0.03029737 omega=6.28318531,34.41442326,188.49555922 pdata=$@ vxdata=vxdata.hess.viscoelastic.H vzdata=vzdata.hess.viscoelastic.H

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

