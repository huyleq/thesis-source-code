function betaandsmectite()
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

depth=0:20:7000;
depth=transpose(depth);
depthft=depth*3.28084;
lat=[28.67544544;28.62160218;28.58814302;28.45741797;28.60183226;28.57276479];
wdepth=[50;60;72;131;75;70]; 
kb=[94.5;96.3;94;93;97;130];

params.smec_ini=1;
params.arr=0.4*10^11; % Arrhenius factor in Myr^-1
params.del_e=20; %Activation energy in kcal/mol
params.R=1.986*10^-3; % Gas constant


params.beta0=6.5;
params.beta1=14;
beta=zeros(size(depth,1),6);
smectite=beta;
for i=1:6
	wellid=i;
	Tswi=tswi_calc(lat(wellid),wdepth(wellid));
	[beta(:,i),smectite(:,i)]=beta_function(depthft,ages,historydepths(:,wellid),[Tswi;bht_ST168+(Tswi-Tswi_ST168)],bht_depths,params);
end

figure('units','normalized','outerposition',[0 0 1 1])
plot(beta(:,1),depth,'k',beta(:,2),depth,'m',beta(:,3),depth,'c',beta(:,4),depth,'r',beta(:,5),depth,'g',beta(:,6),depth,'b')
xlabel('Beta')
ylabel('Depth (m)')
set(gca,'Ydir','reverse')
legend('SS160','SS187','SS191','ST200','ST143','ST168')
figname=strcat('beta.pdf');
export_fig(figname)

figure('units','normalized','outerposition',[0 0 1 1])
plot(smectite(:,1),depth,'k',smectite(:,2),depth,'m',smectite(:,3),depth,'c',smectite(:,4),depth,'r',smectite(:,5),depth,'g',smectite(:,6),depth,'b')
xlabel('Smectite fraction')
ylabel('Depth (m)')
set(gca,'Ydir','reverse')
legend('SS160','SS187','SS191','ST200','ST143','ST168')
figname=strcat('smectite.pdf');
export_fig(figname)
