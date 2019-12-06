function [as,ai,b,smec_SS187,depth_SS187ft]=rhodtSS187
set(0,'defaultLineLineWidth',3)
set(0,'DefaultAxesFontSize',15);
%rho=a+b*dt
[~,~,~,~,~,shalerho1]=selectsandshale('177114129700_Orig+Edit+RckPhys.las',20,5,90,25,3300,12830);
[~,~,~,~,~,shalerho2]=selectsandshale('177114129700_Orig+Edit+RckPhys.las',21,5,90,3,12920,15960);
[~,~,~,shaledepth1,shalegamma1,shaledt1]=selectsandshale('177114129700_Orig+Edit+RckPhys.las',20,5,90,37,3300,12830);
[~,~,~,shaledepth2,shalegamma2,shaledt2]=selectsandshale('177114129700_Orig+Edit+RckPhys.las',21,5,90,37,12920,15960);
depth=[shaledepth1;shaledepth2];
kb=96.3;
wd=60;
depth=depth-kb-wd;
rho=[shalerho1;shalerho2];
dt=[shaledt1;shaledt2]*3.28084e-6;
gamma=[shalegamma1;shalegamma2];

[~,data,~,~]=loadlas('177114129700_Orig+Edit+RckPhys.las');
w=120/(data(2,1)-data(1,1))+1;
m1=floor((3300-data(1,1))/(data(2,1)-data(1,1)))+1;
n1=floor((12830-data(1,1))/(data(2,1)-data(1,1)))+1;
m2=floor((12920-data(1,1))/(data(2,1)-data(1,1)))+1;
n2=floor((15960-data(1,1))/(data(2,1)-data(1,1)))+1;

figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,3,1)
plot(data(m1:n1,20),(data(m1:n1,1)-kb-wd)*0.3048,'b',movmean(data(m1:n1,20),w),(data(m1:n1,1)-kb-wd)*0.3048,'g');
hold on
plot(data(m2:n2,21),(data(m2:n2,1)-kb-wd)*0.3048,'b',movmean(data(m2:n2,21),w),(data(m2:n2,1)-kb-wd)*0.3048,'g');
scatter(gamma,depth*0.3048,'r')
hold off
set(gca,'Ydir','reverse')
xlabel('Gamma ray');
ylabel('Depth (m)');
subplot(1,3,2)
plot(data(m1:n1,25),(data(m1:n1,1)-kb-wd)*0.3048,'b',movmean(data(m1:n1,25),w),(data(m1:n1,1)-kb-wd)*0.3048,'g');
hold on
plot(data(m2:n2,3),(data(m2:n2,1)-kb-wd)*0.3048,'b',movmean(data(m2:n2,3),w),(data(m2:n2,1)-kb-wd)*0.3048,'g');
scatter(rho,depth*0.3048,'r')
hold off
set(gca,'Ydir','reverse')
xlabel('Density (g/cc)');
subplot(1,3,3)
scatter(dt,depth*0.3048,'r')
hold on
plot(data(m1:n1,37)*3.28084e-6,(data(m1:n1,1)-kb-wd)*0.3048,'b',movmean(data(m1:n1,37),w)*3.28084e-6,(data(m1:n1,1)-kb-wd)*0.3048,'g');
plot(data(m2:n2,37)*3.28084e-6,(data(m2:n2,1)-kb-wd)*0.3048,'b',movmean(data(m2:n2,37),w)*3.28084e-6,(data(m2:n2,1)-kb-wd)*0.3048,'g');
scatter(dt,depth*0.3048,'r')
hold off
legend('Shale data point','Log','Averaged log','Location','southeast')
set(gca,'Ydir','reverse')
xlabel('Slowness (s/m)');
box on
export_fig('../Fig/shalerhodtSS187.pdf')

kb_ST168=130; wd_ST168=70; 
lat_ST168=28.57276479;
bht_depths=[17060;17552;19361];
bht_depths=[0;bht_depths-(wd_ST168+kb_ST168)];
Tswi_ST168=tswi_calc(lat_ST168,wd_ST168);

bht_ST168=[265;280;300];
temp_ST168=[Tswi_ST168;bht_ST168];

ages=[2.58;5.33];
historydepths=[7.2277e3;1.0304e4];	

lat=28.62160218; 

params.smec_ini=1;
params.arr=0.4*10^11; % Arrhenius factor in Myr^-1
params.del_e=20; %Activation energy in kcal/mol
params.R=1.986*10^-3; % Gas constant

Tswi=tswi_calc(lat,wd);

params.beta0=7;
params.beta1=10;

vel_SS187=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_SS187.H',1500,1);
dt_SS187=1./vel_SS187;
depth_SS187=linspace(0,14990,1500);
depth_SS187ft=depth_SS187*3.28084;

[~,smec_SS187]=beta_function(depth_SS187ft,ages,historydepths,[Tswi;bht_ST168+(Tswi-Tswi_ST168)],bht_depths,params);

smec=interp1(depth_SS187ft,smec_SS187,depth,'linear');
A=[smec,1-smec,dt];
b=rho;
x=A\b;
as=x(1);
ai=x(2);
b=x(3);

figure('units','normalized','outerposition',[0 0 1 1])
plot(smec_SS187,depth_SS187,'k')
xlabel('Smectite fraction');
ylabel('Depth (m)');
set(gca,'Ydir','reverse')
ylim([0 10000])
%export_fig('../Fig/smectiteSS187.pdf')

dt1=linspace(2.3e-4,4.3e-4,100);
depth1=interp1(dt_SS187(1:450),depth_SS187ft(1:450),dt1,'linear','extrap');
smec1=interp1(depth_SS187ft,smec_SS187,depth1,'linear');

figure('units','normalized','outerposition',[0 0 1 1])
plot(dt1,as+b*dt1,'k--',dt1,ai+b*dt1,'k--o',dt1,smec1.*(as+b*dt1)+(1-smec1).*(ai+b*dt1),'k')
hold on
scatter(dt,rho,[],depth*0.3048)
hold off
xlabel('Slowness (s/m)')
ylabel('Density (g/cc)')
c = colorbar;
c.Label.String = 'Depth (m)';
xlim([2.3e-4 4.3e-4])
legend('Smectite density line','Illite density line','Diagenetic model')
%export_fig('../Fig/rhodtSS187.pdf')

end
