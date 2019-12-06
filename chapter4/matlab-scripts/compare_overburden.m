function compare_overburden
set(0,'defaultLineLineWidth',3)
set(0,'DefaultAxesFontSize',15);

maxz=16404; % 5 km deep
dz=0.5;
z=transpose(0:dz:maxz);
n=size(z,1);

vel0=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_SS187.H',1500,1);
z0=transpose(0:32.8084:32.8084*1499);
vel=interp1(z0,vel0,z);


data=loadlas('177114129700_Orig+Edit+RckPhys.las');
kb=96.5;
wdepth=60;
data.depth=data.depth-kb-wdepth;

rho=zeros(n,1);
zz=data.depth(isfinite(data.rhoz));
rho(min(zz)/dz+1:min(zz)/dz+size(zz,1))=data.rhoz(isfinite(data.rhoz));
zb=data.depth(isfinite(data.rhob));
rho(min(zb)/dz+1:min(zb)/dz+size(zb,1))=data.rhob(isfinite(data.rhob));
for i=1:min(zz)/dz
    rho(i)=0.31*vel(i)^0.25;
end
for i=min(zz)/dz+1+size(zz,1):min(zb)/dz
    rho(i)=0.31*vel(i)^0.25;
end
for i=min(zb)/dz+1+size(zb,1):n
    rho(i)=0.31*vel(i)^0.25;
end

rhogard=zeros(n,1);
zgard=data.depth(isfinite(data.rhogard));
rhogard(min(zgard)/dz+1:min(zgard)/dz+size(zgard,1))=data.rhogard(isfinite(data.rhogard));
for i=1:min(zgard)/dz
    rhogard(i)=0.31*vel(i)^0.25;
end
for i=min(zgard)/dz+1+size(zgard,1):n
    rhogard(i)=0.31*vel(i)^0.25;
end

kb_ST168=130; wd_ST168=70; 
lat_ST168=28.57276479;
bht_depths=[17060;17552;19361];
bht_depths=[0;bht_depths-(wd_ST168+kb_ST168)];
Tswi_ST168=tswi_calc(lat_ST168,wd_ST168);

bht_ST168=[265;280;300];
ages=[2.58;5.33];
historydepths=[7.2277e3;1.0304e4];	

lat=28.62160218; 

params.smec_ini=1;
params.arr=0.4*10^11; % Arrhenius factor in Myr^-1
params.del_e=20; %Activation energy in kcal/mol
params.R=1.986*10^-3; % Gas constant

Tswi=tswi_calc(lat,wdepth);

params.beta0=7;
params.beta1=10;

[~,smec]=beta_function(z,ages,historydepths,[Tswi;bht_ST168+(Tswi-Tswi_ST168)],bht_depths,params);

rhod=zeros(n,1);
for i=1:n
    rhod(i)=smec(i)*(2.8535-1.6858e3/vel(i))+(1-smec(i))*(3.253-1.6858e3/vel(i));
end

figure('units','normalized','outerposition',[0 0 1 1])
plot(rho,z*0.3048,'k',rhogard,z*0.3048,'b',rhod,z*0.3048,'g')
xlabel('Density (g/cc)');
ylabel('Depth (m)');
set(gca,'Ydir','reverse')
legend('Bulk density log','Gardner model','Diagenetic model')
export_fig('../Fig/densityModelsSS187.pdf')

sbulk=overburden(wdepth,rho,z);
sgardner=overburden(wdepth,rhogard,z);
sd=overburden(wdepth,rhod,z);
semperical=0.000005432*z.^2+0.8783*z+0.455*wdepth;

figure('units','normalized','outerposition',[0 0 1 1])
plot(sbulk,z*0.3048,'k',sgardner,z*0.3048,'b',sd,z*0.3048,'g',semperical,z*0.3048,'r')
xlabel('Overburden pressure (psi)');
ylabel('Depth (m)');
legend('Bulk density','Gardner model','Diagenetic model','Empirical model')
set(gca,'Ydir','reverse')
export_fig('../Fig/overburdenModelsSS187.pdf')


end
