function ppandrpt(wellname)
set(0,'defaultLineLineWidth',3)
set(0,'DefaultAxesFontSize',15);
%load 'thermal_history_wells.mat'
kb_ST168=130; wd_ST168=70; 
lat_ST168=28.57276479;
bht_depths=[17060;17552;19361];
bht_depths=[0;bht_depths-(wd_ST168+kb_ST168)];
Tswi_ST168=tswi_calc(lat_ST168,wd_ST168);

bht_ST168=[265;280;300];
ages=[2.58;5.33];
historydepths=[5.9538e3,7.2277e3,7.4678e3,7.7944e3,6.9199e3,6.9781e3;...
			   8.2555e3,1.0304e4,1.0755e4,1.1877e4,9.5231e3,9.7039e3];	

depth=0:20:10000;
depth=transpose(depth);
depthft=depth*3.28084;
lat=[28.67544544;28.62160218;28.58814302;28.45741797;28.60183226;28.57276479];
wdepth=[50;60;72;131;75;70]; 
kb=[94.5;96.3;94;93;97;130];

params.smec_ini=1;
params.arr=0.4*10^11; % Arrhenius factor in Myr^-1
params.del_e=20; %Activation energy in kcal/mol
params.R=1.986*10^-3; % Gas constant

if strcmp(wellname,'SS160')
	%SS160
	mudweight=[9.3;9.7;11;15.4;16.6;17;17.7;17.9];
	muddepth=[3500;9775;11800;12870;14343;14671;15961;17500];
	vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_SS160.H',1500,1);
	wellid=1;
	[~,data,~,~]=loadlas('177114095100_Orig+Edit+RckPhys.las');
	soniclog=17;
end

if strcmp(wellname,'SS187')
	%SS187
	mudweight=[9.1;9.3;10.4;14.6;15.1;16.1];
    muddepth=[2440;9960;13000;13368;14950;16190];
	vel=sepread('../velocity_SS187.H',1500,1);
	wellid=2;
	[~,data,~,~]=loadlas('../177114129700_Orig+Edit+RckPhys.las');
	soniclog=37;
end

if strcmp(wellname,'SS191')
	%SS191
	mudweight=[10.5;10.5;14.8;15.8];
    muddepth=[10927;13150;13325;15078];
	vel=sepread('../velocity_SS191.H',1500,1);
	wellid=3;
	[~,data,~,~]=loadlas('177114136300_Orig+Edit+RckPhys.las');
	soniclog=31;
end

if strcmp(wellname,'ST200')
	%ST200
	mudweight=[10;11.8;12.6;13.2;13.5;14.3;15.4;16.2;16;15.3;16.8];
    muddepth=[3553;6025;9305;10415;10957;11500;12425;13955;14263;15205;15592];
	vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_ST200.H',1500,1);
	wellid=4;
	[~,data,~,~]=loadlas('177154042100_Orig+Edit+RckPhys.las');
	soniclog=21;
end

if strcmp(wellname,'ST143')
	%ST143
	mudweight=[8.9;10.5;11.1;14.1;14.8];
    muddepth=[1035;7285;12250;14224;15925];
	vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_ST143.H',1500,1);
	wellid=5;
	[~,data,~,~]=loadlas('177154098500_Orig+Edit+RckPhys.las');
	soniclog=29;
end

if strcmp(wellname,'ST168')
	%ST168
	mudweight=[14.7;14.5;14.5;14.5;14.8;15;15;15;15;14.5;14.7];
    muddepth=[17052;17552;19361;19930;20357;20723;20898;21056;21105;15342;17020];
	vel=sepread('../wells/velocity_ST168.H',1500,1);
	wellid=6;
	[~,data,~,~]=loadlas('../wells/177154117301.las');
	soniclog=8;
end

ppmud=0.0519*mudweight.*muddepth;

Tswi=tswi_calc(lat(wellid),wdepth(wellid));

params.beta0=6.5;
params.beta1=14;
[beta,smecfr]=beta_function(depthft,ages,historydepths(:,wellid),[Tswi;bht_ST168+(Tswi-Tswi_ST168)],bht_depths,params);

dtm=64.5; % in us/ft
x=1.97;
rhow=1.05;
sigma0=26000; %in psi

n=0;
for i=1:size(data(:,1))
    if isfinite(data(i,soniclog))
        n=n+1;
    end
end
dtsonic=zeros(n,1);
depthsonic=dtsonic;
j=0;
for i=1:size(data(:,1))
    if isfinite(data(i,soniclog))
        j=j+1;
        dtsonic(j)=data(i,soniclog);
        depthsonic(j)=data(i,1)-kb(wellid)-wdepth(wellid);
    end
end
w=120/(data(2,1)-data(1,1))+1;
dtsonic=movmean(dtsonic,w);
betafsonic=interp1(depthft,beta,depthsonic,'linear');
[~,~,ppsonic]=dt2pp(dtsonic,dtm,x,sigma0,betafsonic,wdepth(wellid),depthsonic);

dtseismic=0.3048e6./vel(1:1001);
depthseismic=transpose(linspace(0,10000,1001));
betafseismic=interp1(depth,beta,depthseismic,'linear');
[~,~,ppseismic]=dt2pp(dtseismic,dtm,x,sigma0,betafseismic,wdepth(wellid),depthseismic*3.28084);

pphydro=(depthft+wdepth(wellid))*rhow*0.433;

s=0.000005432*depthft.^2+0.8783*depthft+0.455*wdepth(wellid)+14.7;
ppfrac=0.975*s;

plot(ppseismic,depthseismic*0.001,'k',pphydro,depth*0.001,'b',ppfrac,depth*0.001,'r',ppsonic,depthsonic*0.3048*0.001,'g',s,depth*0.001,'m')
hold on
scatter(ppmud,muddepth*0.3048*0.001,80,'filled','k')
legend('seismic','hydro','fracture','sonic','overburden','mud weight')
title(wellname)
xlabel('pressure (psi)')
ylabel('depth (km)')
title('(a)')
set(gca,'Ydir','reverse')
%figure('units','normalized','outerposition',[0 0 1 1])
%subplot(1,3,1)
%plot(ppseismic,depthseismic*0.001,'k',pphydro,depth*0.001,'b',ppfrac,depth*0.001,'r',ppsonic,depthsonic*0.3048*0.001,'g',s,depth*0.001,'m')
%%title(wellname)
%xlabel('pressure (psi)')
%ylabel('depth (km)')
%title('(a)')
%set(gca,'Ydir','reverse')
%hold on
%scatter(ppmud,muddepth*0.3048*0.001,80,'filled','k')
%legend('seismic','hydro','fracture','sonic','overburden','mud weight')
%xlim([0 4e4])
%ylim([0 10])
%hold off
%figname=strcat('../Fig/pp-',wellname,'.pdf');
%%export_fig(figname)
%
%%figure('units','normalized','outerposition',[0 0 1 1])
%subplot(1,3,3)
%plot(smecfr,depth*0.001)
%set(gca,'Ydir','reverse')
%%title(wellname)
%xlabel('Smectite fraction')
%ylabel('depth (km)')
%ylim([0 10])
%title('(c)')
%
%ppgrad=[8.7625;10;11;12;13;14;15;16;17];
%dt=zeros(size(beta,1),10);
%for i=1:9
%    pp=ppgrad(i)*0.0519*depth*3.28084;
%    sigma=s-pp;
%    dt(:,i)=dtm*(1+log(sigma0./sigma)./beta).^x;
%    xi=log(sigma0./sigma)./beta;
%    phi=xi./(1+xi);
%    ix=phi<0.38;
%    dt(~ix,i)=nan;
%end
%sigma=s-ppfrac;
%dt(:,10)=dtm*(1+log(sigma0./sigma)./beta).^x;
%xi=log(sigma0./sigma)./beta;
%phi=xi./(1+xi);
%ix=phi<0.38;
%dt(~ix,10)=nan;
%
%%figure('units','normalized','outerposition',[0 0 1 1])
%subplot(1,3,2)
%plot(0.3048e6./dtsonic,depthsonic*0.3048*0.001,'g',vel(1:1001),depthseismic*0.001,'k')
%title('(b)')
%xlabel('velocity (m/s)')
%ylabel('depth (km)')
%set(gca,'Ydir','reverse')
%hold on
%plot(0.3048e6./dt(:,1),depth*0.001,'b')
%for i=2:9
%    plot(0.3048e6./dt(:,i),depth*0.001,'--')
%end
%plot(0.3048e6./dt(:,10),depth*0.001,'r')
%hold off
%legend('sonic','seismic','hydro','10 ppg','11 ppg','12 ppg','13 ppg','14 ppg','15 ppg',...
%    '16 ppg','17 ppg','fracture')
%xlim([1500 4500])
%figname=strcat('../Fig/pprptsmecfr',wellname,'.pdf');
%export_fig(figname)

end
