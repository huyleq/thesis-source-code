#!/usr/bin/env tcsh

#./Bin/modelData3d.x par=Par/par.dragon3d.p wavelet=wavelets/ricker.dragon3d.15hz.H souloc=data/S4_ph4_data_cmph_subset_7s_recline12_1rec_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_recline12_1rec_souloc.H v=models/dragon.vp.rtm.pad.H eps=models/dragon.eps.rtm.pad.H del=models/dragon.del.rtm.pad.H data=data/testdata.H wbottom=50 gpu=0,1,2,3 wavefield=testwfld.H >& dragon-modelData3d.log

#./Bin/rtm3d.x par=Par/par.dragon3d.p wavelet=wavelets/ricker.dragon3d.15hz.H souloc=data/S4_ph4_data_cmph_subset_7s_recline111213_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_recline111213_souloc.H v=models/dragon.vp.rtm.pad.H eps=models/dragon.eps.rtm.pad.H del=models/dragon.del.rtm.pad.H randomboundary=models/randboundary.dragon.vp.rtm.H data=data/S4_ph4_data_cmph_subset_7s_recline111213_sorted_3_30hz_clip2_tpow1.5.rtm.H image=images/image.recline111213.H image1=images/image1.recline111213.H image2=images/image2.recline111213.H image3=images/image3.recline111213.H wbottom=50 gpu=0,1,2,3 badshot=148 >& dragon-rtm3d-reclin111213.log

#./Bin/processimage.x image=images/image1.recline12.H npad=29 wbottom=50 imageout=images/pimage1.recline12.H
#./Bin/processimage.x image=images/image2.recline12.H npad=29 wbottom=50 imageout=images/pimage2.recline12.H
#./Bin/processimage.x image=images/image3.recline12.H npad=29 wbottom=50 imageout=images/pimage3.recline12.H

#./Bin/rtm3d.x par=Par/par.dragon3d.p wavelet=wavelets/ricker.dragon3d.15hz.H souloc=data/S4_ph4_data_cmph_subset_7s_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_souloc.H v=models/dragon.vz.rtm.pad.H eps=models/dragon.del.rtm.pad.H del=models/dragon.del.rtm.pad.H randomboundary=models/randboundary.dragon.vz.rtm.H data=data/S4_ph4_data_cmph_subset_7s_sorted_3_30hz_clip2_tpow_normalize_rtm_NUPDATE45.H image=images/image1.dragon.vision.vzdeldel.H wbottom=50 shotrange=1600,1750 gpu=0,1 >& dragon-rtm-vision-vzdeldel1.log

#./Bin/sourceInv-cg.x par=Par/par.dragon3d.fwi.p souloc=data/S4_ph4_data_cmph_subset_7s_6rec_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_clip_6rec_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H iwavelet=wavelets/iwavelet1.dragon.6rec.5s.vzepsdel.H data=data/S4_ph4_data_cmph_subset_7s_sorted_3_15hz_clip_5s_6rec.H gpu=0,1,2,3 ngpugroup=4 niter=50 >& sourceInv1-dragon.log

#./Bin/modelData3d.x par=Par/par.dragon3d.fwi.p par=Par/par.dragon3d.fwi.p wavelet=wavelets/iwavelet1.dragon.6rec.5s.vzepsdel.0_15hz_pad.H souloc=data/S4_ph4_data_cmph_subset_7s_400spacing_rec477_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_400spacing_clip_rec477_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H data=data/bgdata_rec477.H gpu=0 datapath=/net/vision/scr1/huyle/ >& dragon-modelData3d.log

#./Bin/sourceInv-cg.x par=Par/par.dragon3d.fwi.p souloc=data/S4_ph4_data_cmph_subset_7s_6rec_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_clip_6rec_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H iwavelet=wavelets/iwavelet2.dragon.6rec.3s.vzepsdel.H data=data/S4_ph4_data_cmph_subset_7s_sorted_3_15hz_clip_shift_6rec.H gpu=0,1,2,3 ngpugroup=4 niter=25 >& sourceInv2-dragon.log

#./Bin/sourceInv-cg.x par=Par/par.dragon3d.fwi.p souloc=data/S4_ph4_data_cmph_subset_7s_6rec_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_clip_6rec_souloc.H v=models/dragon.vz.25m.5km.pad.H eps=models/dragon.eps.25m.5km.pad.H del=models/dragon.del.25m.5km.pad.H iwavelet=wavelets/iwavelet3.dragon.6rec.3s.vzepsdel.H data=data/S4_ph4_data_cmph_subset_7s_sorted_3_15hz_clip_shift_6rec.H gpu=0,1,2,3 ngpugroup=4 niter=10 wavelet=wavelets/wavelet.0phase.6rec.3_15hz.1ms.pad.H >& sourceInv3-dragon.log

./Bin/sourceInv-cg.x par=Par/par.dragon3d.p souloc=data/S4_ph4_data_cmph_subset_7s_6rec_recloc.H recloc=data/S4_ph4_data_cmph_subset_7s_sorted_3_30hz_clip_6rec_souloc.H v=models/dragon.vz.rtm.16m.5km.pad.H eps=models/dragon.eps.rtm.16m.5km.pad.H del=models/dragon.del.rtm.16m.5km.pad.H iwavelet=wavelets/iwavelet5.dragon.rtm.6rec.3s.vzepsdel.H data=data/S4_ph4_data_cmph_subset_7s_sorted_3_30hz_clip2_shift_3s_6rec.H gpu=0,1,2,3,4,5 ngpugroup=6 niter=10 >& sourceInv5-dragon.log

