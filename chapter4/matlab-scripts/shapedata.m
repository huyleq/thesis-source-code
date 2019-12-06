addpath('/net/server2/homes/sep/huyle/sep/sep-matlab-io');
nt=2501;
ntrace=10717;
datafile='data.1.30.H';

w0=sepread('wavelet.1.30.smooth.H',nt,1);
wmin=sepread('wavelet.1.30.smooth.min.bandpass.H',nt,1);

half=100;

w01=zeros(2*half+1,1);
w01(half+1:end)=w0(1:half+1);
w01(1:half)=w0(nt-half+1:end);

wmin1=zeros(2*half+1,1);
wmin1(half+1:end)=wmin(1:half+1);

nPosLag=70;
nNegLag=70;
filter=shaping_filter(w01,wmin1,half+1,nPosLag,nNegLag);

w=zeros(nt,1);
temp=myconv(w01,filter,nNegLag+1);
w(2:half+2-12)=temp(half+1:end-12);
sepwrite('wavelet.1.30.smooth.shape.H',w,[nt;1],[0;0],[0.004;1]);

data=sepread(datafile,nt,ntrace);

ddata=zeros(nt,ntrace);

for i=1:ntrace
    ddata(:,i)=myconv(data(:,i),filter,nNegLag+1);
end

sepwrite('data.1.30.shape.H',ddata,[size(ddata,1);size(ddata,2)],[0;0],[0.004;1]);

