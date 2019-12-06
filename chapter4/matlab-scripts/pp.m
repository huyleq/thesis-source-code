function pp(betaf,betadepth)
dtm=65; % in us/ft
x=2.1;
rhow=1.05;
sigma0=26500; %in psi

%SS160
data=loadlas('177114095100_Orig+Edit+RckPhys.las');
kb=94.5;
wdepth=50; %in ft

n=0;
for i=1:size(data.depth)
    if isfinite(data.dt_ed7(i))
        n=n+1;
    end
end
dtsonic=zeros(n,1);
depthsonic=dtsonic;
j=0;
for i=1:size(data.depth)
    if isfinite(data.dt_ed7(i))
        j=j+1;
        dtsonic(j)=data.dt_ed7(i);
        depthsonic(j)=data.depth(i)-kb-wdepth;
    end
end
dtsonic=movmean(dtsonic,130);
betafsonic=interp1(betadepth,betaf(1:101),depthsonic*0.3048,'linear');
[~,~,ppsonic]=dt2pp(dtsonic,dtm,x,sigma0,betafsonic,wdepth,depthsonic);

mudweight=[9.3;9.7;11;15.4;16.6;17;17.7;17.9];
muddepth=[3500;9775;11800;12870;14343;14671;15961;17500];
ppmud=0.0519*mudweight.*muddepth;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_SS160.H',1500,1);
dtseismic=0.3048e6./vel(1:601);
depthseismic=transpose(linspace(0,6000*3.28084,601));
betafseismic=interp1(betadepth,betaf(1:101),depthseismic*0.3048,'linear');
[s,~,ppseismic]=dt2pp(dtseismic,dtm,x,sigma0,betafseismic,wdepth,depthseismic);

pphydro=(depthseismic+wdepth)*rhow*0.433;
ppfrac=0.975*s;

figure
plot(ppseismic,depthseismic*0.3048,'b',pphydro,depthseismic*0.3048,'g',ppfrac,depthseismic*0.3048,'k',ppsonic,depthsonic*0.3048,'r',s,depthseismic*0.3048,'m')
title('SS160')
xlabel('pressure (psi)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
scatter(ppmud,muddepth*0.3048,'filled','k')
legend('seismic','hydro','fracture','sonic','overburden','mud weight')
hold off

%SS187
data=loadlas('177114129700_Orig+Edit+RckPhys.las');
kb=96.3;
wdepth=60;

n=0;
for i=1:size(data.depth)
    if isfinite(data.dt_ed5(i))
        n=n+1;
    end
end
dtsonic=zeros(n,1);
depthsonic=dtsonic;
j=0;
for i=1:size(data.depth)
    if isfinite(data.dt_ed5(i))
        j=j+1;
        dtsonic(j)=data.dt_ed5(i);
        depthsonic(j)=data.depth(i)-kb-wdepth;
    end
end
dtsonic=movmean(dtsonic,130);
betafsonic=interp1(betadepth,betaf(102:202),depthsonic*0.3048,'linear');
[~,~,ppsonic]=dt2pp(dtsonic,dtm,x,sigma0,betafsonic,wdepth,depthsonic);

mudweight=[9.1;9.3;10.4;14.6;15.1;16.1];
muddepth=[2440;9960;13000;13368;14950;16190];
ppmud=0.0519*mudweight.*muddepth;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_SS187.H',1500,1);
dtseismic=0.3048e6./vel(1:601);
depthseismic=transpose(linspace(0,6000*3.28084,601));
betafseismic=interp1(betadepth,betaf(102:202),depthseismic*0.3048,'linear');

[s,~,ppseismic]=dt2pp(dtseismic,dtm,x,sigma0,betafseismic,wdepth,depthseismic);

pphydro=(depthseismic+wdepth)*rhow*0.433;
ppfrac=0.975*s;

figure
plot(ppseismic,depthseismic*0.3048,'b',pphydro,depthseismic*0.3048,'g',ppfrac,depthseismic*0.3048,'k',ppsonic,depthsonic*0.3048,'r',s,depthseismic*0.3048,'m')
title('SS187')
xlabel('pressure (psi)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
scatter(ppmud,muddepth*0.3048,'filled','k')
legend('seismic','hydro','fracture','sonic','overburden','mud weight')
hold off

%SS191
data=loadlas('177114136300_Orig+Edit+RckPhys.las');
kb=94;
wdepth=72;

n=0;
for i=1:size(data.depth)
    if isfinite(data.dtln_ed2(i))
        n=n+1;
    end
end
dtsonic=zeros(n,1);
depthsonic=dtsonic;
j=0;
for i=1:size(data.depth)
    if isfinite(data.dtln_ed2(i))
        j=j+1;
        dtsonic(j)=data.dtln_ed2(i);
        depthsonic(j)=data.depth(i)-kb-wdepth;
    end
end
dtsonic=movmean(dtsonic,130);
betafsonic=interp1(betadepth,betaf(203:303),depthsonic*0.3048,'linear');
[~,~,ppsonic]=dt2pp(dtsonic,dtm,x,sigma0,betafsonic,wdepth,depthsonic);

mudweight=[10.5;10.5;14.8;15.8];
muddepth=[10927;13150;13325;15078];
ppmud=0.0519*mudweight.*muddepth;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_SS191.H',1500,1);
dtseismic=0.3048e6./vel(1:601);
depthseismic=transpose(linspace(0,6000*3.28084,601));
betafseismic=interp1(betadepth,betaf(203:303),depthseismic*0.3048,'linear');
[s,~,ppseismic]=dt2pp(dtseismic,dtm,x,sigma0,betafseismic,wdepth,depthseismic);

pphydro=(depthseismic+wdepth)*rhow*0.433;
ppfrac=0.975*s;

figure
plot(ppseismic,depthseismic*0.3048,'b',pphydro,depthseismic*0.3048,'g',ppfrac,depthseismic*0.3048,'k',ppsonic,depthsonic*0.3048,'r',s,depthseismic*0.3048,'m')
title('SS191')
xlabel('pressure (psi)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
scatter(ppmud,muddepth*0.3048,'filled','k')
legend('seismic','hydro','fracture','sonic','overburden','mud weight')
hold off

%ST200
data=loadlas('177154042100_Orig+Edit+RckPhys.las');
kb=93;
wdepth=131;

n=0;
for i=1:size(data.depth)
    if isfinite(data.dt_ed4(i))
        n=n+1;
    end
end
dtsonic=zeros(n,1);
depthsonic=dtsonic;
j=0;
for i=1:size(data.depth)
    if isfinite(data.dt_ed4(i))
        j=j+1;
        dtsonic(j)=data.dt_ed4(i);
        depthsonic(j)=data.depth(i)-kb-wdepth;
    end
end
dtsonic=movmean(dtsonic,130);
betafsonic=interp1(betadepth,betaf(304:404),depthsonic*0.3048,'linear');
[~,~,ppsonic]=dt2pp(dtsonic,dtm,x,sigma0,betafsonic,wdepth,depthsonic);

mudweight=[10;11.8;12.6;13.2;13.5;14.3;15.4;16.2;16;15.3;16.8];
muddepth=[3553;6025;9305;10415;10957;11500;12425;13955;14263;15205;15592];
ppmud=0.0519*mudweight.*muddepth;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_ST200.H',1500,1);
dtseismic=0.3048e6./vel(1:601);
depthseismic=transpose(linspace(0,6000*3.28084,601));
betafseismic=interp1(betadepth,betaf(304:404),depthseismic*0.3048,'linear');
[s,~,ppseismic]=dt2pp(dtseismic,dtm,x,sigma0,betafseismic,wdepth,depthseismic);

pphydro=(depthseismic+wdepth)*rhow*0.433;
ppfrac=0.975*s;

figure
plot(ppseismic,depthseismic*0.3048,'b',pphydro,depthseismic*0.3048,'g',ppfrac,depthseismic*0.3048,'k',ppsonic,depthsonic*0.3048,'r',s,depthseismic*0.3048,'m')
title('ST200')
xlabel('pressure (psi)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
scatter(ppmud,muddepth*0.3048,'filled','k')
legend('seismic','hydro','fracture','sonic','overburden','mud weight')
hold off

%ST143
data=loadlas('177154098500_Orig+Edit+RckPhys.las');
kb=97;
wdepth=75;

n=0;
for i=1:size(data.depth)
    if isfinite(data.dt_ed1(i))
        n=n+1;
    end
end
dtsonic=zeros(n,1);
depthsonic=dtsonic;
j=0;
for i=1:size(data.depth)
    if isfinite(data.dt_ed1(i))
        j=j+1;
        dtsonic(j)=data.dt_ed1(i);
        depthsonic(j)=data.depth(i)-kb-wdepth;
    end
end
dtsonic=movmean(dtsonic,130);
betafsonic=interp1(betadepth,betaf(405:505),depthsonic*0.3048,'linear');
[~,~,ppsonic]=dt2pp(dtsonic,dtm,x,sigma0,betafsonic,wdepth,depthsonic);

mudweight=[8.9;10.5;11.1;14.1;14.8];
muddepth=[1035;7285;12250;14224;15925];
ppmud=0.0519*mudweight.*muddepth;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_ST143.H',1500,1);
dtseismic=0.3048e6./vel(1:601);
depthseismic=transpose(linspace(0,6000*3.28084,601));
betafseismic=interp1(betadepth,betaf(405:505),depthseismic*0.3048,'linear');
[s,~,ppseismic]=dt2pp(dtseismic,dtm,x,sigma0,betafseismic,wdepth,depthseismic);

pphydro=(depthseismic+wdepth)*rhow*0.433;
ppfrac=0.975*s;

figure
plot(ppseismic,depthseismic*0.3048,'b',pphydro,depthseismic*0.3048,'g',ppfrac,depthseismic*0.3048,'k',ppsonic,depthsonic*0.3048,'r',s,depthseismic*0.3048,'m')
title('ST143')
xlabel('pressure (psi)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
scatter(ppmud,muddepth*0.3048,'filled','k')
legend('seismic','hydro','fracture','sonic','overburden','mud weight')
hold off

%ST168
data=loadlas('177154117301.las');
kb=130;
wdepth=70;

n=0;
for i=1:size(data.dept_f)
    if isfinite(data.dtco_us_f(i))
        n=n+1;
    end
end
dtsonic=zeros(n,1);
depthsonic=dtsonic;
j=0;
for i=1:size(data.dept_f)
    if isfinite(data.dtco_us_f(i))
        j=j+1;
        dtsonic(j)=data.dtco_us_f(i);
        depthsonic(j)=data.dept_f(i)-kb-wdepth;
    end
end
dtsonic=movmean(dtsonic,130);
betafsonic=interp1(betadepth,betaf(506:606),depthsonic*0.3048,'linear');
[~,~,ppsonic]=dt2pp(dtsonic,dtm,x,sigma0,betafsonic,wdepth,depthsonic);

mudweight=[14.7;14.5;14.5;14.5;14.8;15;15;15;15;14.5;14.7];
muddepth=[17052;17552;19361;19930;20357;20723;20898;21056;21105;15342;17020];
ppmud=0.0519*mudweight.*muddepth;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_ST168.H',1500,1);
dtseismic=0.3048e6./vel(1:601);
depthseismic=transpose(linspace(0,6000*3.28084,601));
betafseismic=interp1(betadepth,betaf(405:505),depthseismic*0.3048,'linear');
[s,~,ppseismic]=dt2pp(dtseismic,dtm,x,sigma0,betafseismic,wdepth,depthseismic);

pphydro=(depthseismic+wdepth)*rhow*0.433;
ppfrac=0.975*s;

figure
plot(ppseismic,depthseismic*0.3048,'b',pphydro,depthseismic*0.3048,'g',ppfrac,depthseismic*0.3048,'k',ppsonic,depthsonic*0.3048,'r',s,depthseismic*0.3048,'m')
title('ST168')
xlabel('pressure (psi)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
scatter(ppmud,muddepth*0.3048,'filled','k')
legend('seismic','hydro','fracture','sonic','overburden','mud weight')
hold off
end
