samplingRate=0.004;       
Fmax=0.5/samplingRate;
nt=1000; 
t=(0:nt-1)*samplingRate;
df=2*Fmax/nt;
f=(0:nt/2)*df;

F=10;
s=0.7*cos(2*pi*F*t);

S=fft(s);
A=abs(S/nt);
amp=A(1:nt/2+1);
amp(2:end-1)=2*amp(2:end-1);
phase=atan2(imag(S(1:nt/2+1)),real(S(1:nt/2+1)))*180/pi;

figure 
subplot(2,1,1)
plot(real(S))
subplot(2,1,2)
plot(imag(S))

ss=ifft(1000*A);

figure
subplot(4,1,1)
plot(t,s)
subplot(4,1,2)
plot(f,amp)
subplot(4,1,3)
plot(f,phase)
subplot(4,1,4)
plot(t,ss)
