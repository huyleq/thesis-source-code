addpath('/data/cees/huyle/sep/sep-matlab-io');
tempdata=sepread('../line1/ddata1.dragon2d.30.H',8333,2501);
data=tempdata';

ntrace=size(data,2);

samplingRate=0.004;       
Fmax=0.5/samplingRate;
nt=2501; 
t=(-(nt-1)/2:(nt-1)/2)*samplingRate;
df=2*Fmax/(nt-1);
f=(0:(nt-1)/2)*df;

spectra=abs(fft(data,[],1));
spectrum=sum(spectra,2)/ntrace;

wavelet=ifft(spectrum);

%spectrum1=spectrum(1:(nt+1)/2)/nt;
%spectrum1(2:end-1)=2*spectrum1(2:end-1);
%
temp=wavelet(1:(nt+1)/2);
wavelet(1:(nt-1)/2)=wavelet((nt+3)/2:end);
wavelet((nt+1)/2:end)=temp;

figure
subplot(4,1,1)
plot(spectrum)
subplot(4,1,2)
plot(wavelet)

spectra_smooth=spectra;
Fmin=5;
b=round(Fmin/df);
w=21;
spectra_smooth(b:(nt+1)/2,:)=movmean(spectra_smooth(b:(nt+1)/2,:),w,1);
spectra_smooth((nt+1)/2+1:nt,:)=flip(spectra_smooth(2:(nt+1)/2,:),1);

spectrum=sum(spectra_smooth,2)/ntrace;

wavelet=ifft(spectrum);

temp=wavelet(1:(nt+1)/2);
wavelet(1:(nt-1)/2)=wavelet((nt+3)/2:end);
wavelet((nt+1)/2:end)=temp;

subplot(4,1,3)
plot(spectrum)
subplot(4,1,4)
plot(wavelet)

half_width=100;
mid=(nt+1)/2;
temp=wavelet(mid-half_width:mid+half_width);
sepwrite('../line1/wavelet.ddata1.30.zero.H',temp,[2*half_width+1;1],[-half_width*0.004;0],[0.004;1])
