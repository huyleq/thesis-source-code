beta=sepread('../line1/beta1.H',1067,401);
dtm=64.5; % in us/ft
X=1.97;
sigma0=26000; %in psi
pp=sepread('../line1/pp.basin.interp.H',1067,201);
wdepth=25.*3.28084;
depth=linspace(0,5000,201)*3.28084;
v=zeros(1067,201);
for i=1:1067
	dt=pp2dt(pp(i,:),dtm,X,sigma0,beta(i,1:201),wdepth,depth);
	v(i,:)=0.3048e6./dt;
end
sepwrite('../line1/v.basin.H',v,[1067;201],[0;0],[25;25]);



