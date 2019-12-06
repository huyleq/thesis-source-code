set(0, 'DefaultLineLineWidth', 2.5);

nt=1001;
mid=(nt+1)/2;
mint=-10.;
maxt=10.;
t=linspace(mint,maxt,nt)';
dt=(maxt-mint)/(nt-1.);

forward=[469/90;-223/10;879/20;-949/18;41;-201/10;1019/180;-7/10];
center=[1/90;-3/20;3/2;-49/18;3/2;-3/20;1/90];

x=sinc(t);
d2x_forward=myconv(x,forward,1);
d2x_center=myconv(x,center,4);
d2x=-pi^2*x-2*(cos(pi*t)-x)./(t.^2);
d2x(mid)=-1/3*pi^2;
d2x=d2x*dt^2;

f=0.1;
y=ricker(t,f);
d2y_forward=myconv(y,forward,1);
d2y_center=myconv(y,center,4);
pi2f2=pi^2*f^2;
pi2f2t2=pi2f2*t.^2;
d2y=-2*pi2f2*(3-12*pi2f2t2+4*pi2f2t2.^2).*exp(-pi2f2t2);
d2y=d2y*dt^2;


