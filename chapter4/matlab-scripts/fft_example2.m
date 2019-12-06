samplingRate=0.004;       
Fmax=0.5/samplingRate;
nt=2501; 
t=(-(nt-1)/2:(nt-1)/2)*samplingRate;
df=2*Fmax/(nt-1);
f=(0:(nt-1)/2)*df;

s=sinc(t);

S=fft(s);
A=abs(S);
amp=A(1:(nt+1)/2)/nt;
amp(2:end-1)=2*amp(2:end-1);
phase=atan2(imag(S(1:(nt+1)/2)),real(S(1:(nt+1)/2)))*180/pi;

%figure 
%subplot(2,1,1)
%plot(real(S))
%subplot(2,1,2)
%plot(imag(S))

ss=ifft(A);
temp=ss(1:(nt+1)/2);
ss(1:(nt-1)/2)=ss((nt+3)/2:end);
ss((nt+1)/2:end)=temp;

figure
subplot(3,1,1)
plot(t,s,'b',t,ss,'k')
subplot(3,1,2)
plot(f,amp)
subplot(3,1,3)
plot(f,phase)
