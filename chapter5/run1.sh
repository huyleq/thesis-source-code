#!/usr/bin/env tcsh

#./Bin/modelData3d.x par=Par/par.dragon3d.p wavelet=wavelets/ricker.dragon3d.15hz.H souloc=data/S4_ph4_data_cmph_subset_7s_recline12_1rec_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_recline12_1rec_souloc.H v=models/dragon.vp.rtm.pad.H eps=models/dragon.eps.rtm.pad.H del=models/dragon.del.rtm.pad.H data=data/testdata.H wbottom=50 gpu=0,1,2,3 wavefield=testwfld.H >& dragon-modelData3d.log

#./Bin/rtm3d.x par=Par/par.dragon3d.p wavelet=wavelets/ricker.dragon3d.15hz.H souloc=data/S4_ph4_data_cmph_subset_7s_recline111213_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_recline111213_souloc.H v=models/dragon.vp.rtm.pad.H eps=models/dragon.eps.rtm.pad.H del=models/dragon.del.rtm.pad.H randomboundary=models/randboundary.dragon.vp.rtm.H data=data/S4_ph4_data_cmph_subset_7s_recline111213_sorted_3_30hz_clip2_tpow1.5.rtm.H image=images/image.recline111213.H image1=images/image1.recline111213.H image2=images/image2.recline111213.H image3=images/image3.recline111213.H wbottom=50 gpu=0,1,2,3 badshot=148 >& dragon-rtm3d-reclin111213.log

#./Bin/processimage.x image=images/image1.recline12.H npad=29 wbottom=50 imageout=images/pimage1.recline12.H
#./Bin/processimage.x image=images/image2.recline12.H npad=29 wbottom=50 imageout=images/pimage2.recline12.H
#./Bin/processimage.x image=images/image3.recline12.H npad=29 wbottom=50 imageout=images/pimage3.recline12.H

#./Bin/rtm3d.x par=Par/par.dragon3d.p wavelet=wavelets/ricker.dragon3d.15hz.H souloc=data/S4_ph4_data_cmph_subset_7s_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_souloc.H v=models/dragon.vz.rtm.pad.H eps=models/dragon.del.rtm.pad.H del=models/dragon.del.rtm.pad.H randomboundary=models/randboundary.dragon.vz.rtm.H data=data/S4_ph4_data_cmph_subset_7s_sorted_3_30hz_clip2_tpow_normalize_rtm_NUPDATE45.H image=images/image2.dragon.vision.vzdeldel.H wbottom=50 shotrange=1750,1900 gpu=2,3 >& dragon-rtm-vision-vzdeldel2.log

#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.H geps=geps.dragon.lbfgsb.agc.test.H gdel=gdel.dragon.lbfgsb.agc.test.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_5s_shift_NUPDATE8.H gpu=0,1,2,3,4,5,6,7 ngpugroup=8 shotrange=0,572 server=cees-mazama.stanford.edu remoteworkdir=/data/cees/huyle/dragon3dc/ remotedatapath=/data/cees/huyle/scratch/ remotescript="Bin/objFuncGradient3d-cij-cluster.x exec=./Bin/objFuncGradient3d-cij-cpu.x par=Par/par.dragon3d.fwi.p wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_5s_shift_NUPDATE8.H shotrange=572,2004 workdir=/data/cees/huyle/dragon3dc/ datapath=/data/cees/huyle/scratch/" remotecommand=/bin/bash >& objfuncgradient-jarvis-cees-test.log

#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.H geps=geps.dragon.lbfgsb.agc.test.H gdel=gdel.dragon.lbfgsb.agc.test.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_5s_shift_NUPDATE8.H gpu=0,1,2,3,4,5,6,7 ngpugroup=2 shotrange=1000,1008 >& objfuncgradient-jarvis-cees-test.2group.16update.2.log

#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.1.H geps=geps.dragon.lbfgsb.agc.test.1.H gdel=gdel.dragon.lbfgsb.agc.test.1.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_1s_shift_NUPDATE8.H gpu=0 ngpugroup=1 shotrange=1000,1008 forwardAbcWfld=forwardAbcWfld.shot1004.1.H forwardRandWfld=forwardRandWfld.shot1004.1.H adjointWfld=adjointWfld.shot1004.1.H modeleddata=modeleddata.shot1004.1.H adjsou=adjsou.shot1004.1.H >& objfuncgradient-jarvis-cees-test.1group.64update.1.log
#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.2.H geps=geps.dragon.lbfgsb.agc.test.2.H gdel=gdel.dragon.lbfgsb.agc.test.2.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_1s_shift_NUPDATE8.H gpu=0 ngpugroup=1 shotrange=1000,1008 forwardAbcWfld=forwardAbcWfld.shot1004.2.H forwardRandWfld=forwardRandWfld.shot1004.2.H adjointWfld=adjointWfld.shot1004.2.H modeleddata=modeleddata.shot1004.2.H adjsou=adjsou.shot1004.2.H >& objfuncgradient-jarvis-cees-test.1group.64update.2.log
#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.3.H geps=geps.dragon.lbfgsb.agc.test.3.H gdel=gdel.dragon.lbfgsb.agc.test.3.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_1s_shift_NUPDATE8.H gpu=0 ngpugroup=1 shotrange=1000,1008 forwardAbcWfld=forwardAbcWfld.shot1004.3.H forwardRandWfld=forwardRandWfld.shot1004.3.H adjointWfld=adjointWfld.shot1004.3.H modeleddata=modeleddata.shot1004.3.H adjsou=adjsou.shot1004.3.H >& objfuncgradient-jarvis-cees-test.1group.64update.3.log
#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.4.H geps=geps.dragon.lbfgsb.agc.test.4.H gdel=gdel.dragon.lbfgsb.agc.test.4.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_1s_shift_NUPDATE8.H gpu=0 ngpugroup=1 shotrange=1000,1008 forwardAbcWfld=forwardAbcWfld.shot1004.4.H forwardRandWfld=forwardRandWfld.shot1004.4.H adjointWfld=adjointWfld.shot1004.4.H modeleddata=modeleddata.shot1004.4.H adjsou=adjsou.shot1004.4.H >& objfuncgradient-jarvis-cees-test.1group.64update.4.log
#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.5.H geps=geps.dragon.lbfgsb.agc.test.5.H gdel=gdel.dragon.lbfgsb.agc.test.5.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_1s_shift_NUPDATE8.H gpu=0 ngpugroup=1 shotrange=1000,1008 forwardAbcWfld=forwardAbcWfld.shot1004.5.H forwardRandWfld=forwardRandWfld.shot1004.5.H adjointWfld=adjointWfld.shot1004.5.H modeleddata=modeleddata.shot1004.5.H adjsou=adjsou.shot1004.5.H >& objfuncgradient-jarvis-cees-test.1group.64update.5.log

#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.1.H geps=geps.dragon.lbfgsb.agc.test.1.H gdel=gdel.dragon.lbfgsb.agc.test.1.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_5s_shift_NUPDATE8.H gpu=0,1,2,3 ngpugroup=4 shotrange=1000,1008 forwardAbcWfld=forwardAbcWfld.1.H forwardRandWfld=forwardRandWfld.1.H adjointWfld=adjointWfld.1.H modeleddata=modeleddata.1.H adjsou=adjsou.1.H >& objfuncgradient-jarvis-cees-test.4group.64udate.19.log
#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.2.H geps=geps.dragon.lbfgsb.agc.test.2.H gdel=gdel.dragon.lbfgsb.agc.test.2.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_5s_shift_NUPDATE8.H gpu=0,1,2,3 ngpugroup=4 shotrange=1000,1008 forwardAbcWfld=forwardAbcWfld.2.H forwardRandWfld=forwardRandWfld.2.H adjointWfld=adjointWfld.2.H modeleddata=modeleddata.2.H adjsou=adjsou.2.H >& objfuncgradient-jarvis-cees-test.4group.64udate.29.log
#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.3.H geps=geps.dragon.lbfgsb.agc.test.3.H gdel=gdel.dragon.lbfgsb.agc.test.3.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_5s_shift_NUPDATE8.H gpu=0,1,2,3 ngpugroup=4 shotrange=1000,1008 forwardAbcWfld=forwardAbcWfld.3.H forwardRandWfld=forwardRandWfld.3.H adjointWfld=adjointWfld.3.H modeleddata=modeleddata.3.H adjsou=adjsou.3.H >& objfuncgradient-jarvis-cees-test.4group.64udate.39.log
#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.4.H geps=geps.dragon.lbfgsb.agc.test.4.H gdel=gdel.dragon.lbfgsb.agc.test.4.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_5s_shift_NUPDATE8.H gpu=0,1,2,3 ngpugroup=4 shotrange=1000,1008 forwardAbcWfld=forwardAbcWfld.4.H forwardRandWfld=forwardRandWfld.4.H adjointWfld=adjointWfld.4.H modeleddata=modeleddata.4.H adjsou=adjsou.4.H >& objfuncgradient-jarvis-cees-test.4group.64udate.49.log
#./Bin/objFuncGradient3d-vepsdel-jarvis-cees.x par=Par/par.dragon3d.fwi.p icall=9876 wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_200spacing_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_200spacing_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H gv=gv.dragon.lbfgsb.agc.test.5.H geps=geps.dragon.lbfgsb.agc.test.5.H gdel=gdel.dragon.lbfgsb.agc.test.5.H wbottom=50 mask=models/dragon.saltmask.25m.5km.pad.H padboundary=models/padboundary.dragon.cij.25m.5km.H randomboundary=models/randboundary.dragon.cij.25m.5km.H data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_fwi_5s_shift_NUPDATE8.H gpu=0,1,2,3 ngpugroup=4 shotrange=1000,1008 forwardAbcWfld=forwardAbcWfld.5.H forwardRandWfld=forwardRandWfld.5.H adjointWfld=adjointWfld.5.H modeleddata=modeleddata.5.H adjsou=adjsou.5.H >& objfuncgradient-jarvis-cees-test.4group.64udate.59.log

#./Bin/modelData3d.x par=Par/par.dragon3d.fwi.p wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_rec1200_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_rec1200_clip_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H data=data/modeleddata.vzepsdel.rec1200.H wbottom=50 gpu=0 
#./Bin/modelData3d.x par=Par/par.dragon3d.fwi.p wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_rec1200_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_rec1200_clip_souloc.H v=v.dragon.lbfgsb.nobound.agc.alldata.iter12.scale.H eps=eps.dragon.lbfgsb.nobound.agc.alldata.iter12.scale.H del=del.dragon.lbfgsb.nobound.agc.alldata.iter12.scale.H data=data/modeleddata.nobound.rec1200.H wbottom=50 gpu=0 
#./Bin/modelData3d.x par=Par/par.dragon3d.fwi.p wavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_rec1200_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_rec1200_clip_souloc.H v=v.dragon.lbfgsb.bound.agc.alldata.iter13.scale.H eps=eps.dragon.lbfgsb.bound.agc.alldata.iter13.scale.H del=del.dragon.lbfgsb.bound.agc.alldata.iter13.scale.H data=data/modeleddata.bound.rec1200.H wbottom=50 gpu=0 
./Bin/applyAGC.x halfwidth=624 data=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_rec1200.5s.H agcdata=data/S4_ph4_data_cmph_subset_7s_200spacing_3_15hz_clip_mute_rec1200.5s.agc.H
./Bin/applyAGC.x halfwidth=624 data=data/modeleddata.vzepsdel.rec1200.H agcdata=data/modeleddata.vzepsdel.rec1200.agc.H
./Bin/applyAGC.x halfwidth=624 data=data/modeleddata.nobound.rec1200.H agcdata=data/modeleddata.nobound.rec1200.agc.H
./Bin/applyAGC.x halfwidth=624 data=data/modeleddata.bound.rec1200.H agcdata=data/modeleddata.bound.rec1200.agc.H