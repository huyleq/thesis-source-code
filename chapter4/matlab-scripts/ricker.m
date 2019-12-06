function w=ricker(t,f)
n=size(t,1);
w=zeros(n,1);
for i=1:n
	pi2f2t2=pi^2*f^2*t(i)^2;
	w(i)=(1-2*pi2f2t2)*exp(-pi2f2t2);
end
end
