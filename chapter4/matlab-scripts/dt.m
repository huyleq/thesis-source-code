function dt(betaf,betadepth)
dtm=60.0310; % in us/ft
x=2.0748;
sigma0=24807;

%SS160
data=loadlas('177114095100_Orig+Edit+RckPhys.las');
kb=94.5;
wdepth=50; %in ft
data.depth=data.depth-kb-wdepth;

betadepthft=betadepth(1:61)*3.28084;
s=0.000005432*betadepthft.^2+0.8783*betadepthft+0.455*wdepth+14.7;
ppgrad=[8.7625;10;11;12;13;14;15;16;17];
dt=zeros(61,10);
for i=1:9
    pp=ppgrad(i)*0.0519*betadepthft;
    sigma=s-pp;
    dt(:,i)=dtm*(1+log(sigma0./sigma)./betaf(1:61)).^x;
    xi=log(sigma0./sigma)./betaf(1:61);
    phi=xi./(1+xi);
    ix=phi<0.38;
    dt(~ix,i)=nan;
end
sigma=s-0.975*s;
dt(:,10)=dtm*(1+log(sigma0./sigma)./betaf(1:61)).^x;
xi=log(sigma0./sigma)./betaf(1:61);
phi=xi./(1+xi);
ix=phi<0.38;
dt(~ix,10)=nan;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_SS160.H',1500,1);
depthseismic=transpose(linspace(0,6000*3.28084,601));

figure
plot(0.3048e6./data.dt_ed7,data.depth*0.3048,'g',vel(1:601),depthseismic*0.3048,'k')
title('SS160')
xlabel('velocity (m/s)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
for i=1:10
    plot(0.3048e6./dt(:,i),betadepth(1:61),'--')
end
hold off
legend('sonic','seismic','hydro','10 ppg','11 ppg','12 ppg','13 ppg','14 ppg','15 ppg',...
    '16 ppg','17 ppg','fracture')

%SS187
data=loadlas('177114129700_Orig+Edit+RckPhys.las');
kb=96.3;
wdepth=60;
data.depth=data.depth-kb-wdepth;

betadepthft=betadepth(1:61)*3.28084;
s=0.000005432*betadepthft.^2+0.8783*betadepthft+0.455*wdepth+14.7;
sigma=s-0.975*s;
dt(:,10)=dtm*(1+log(sigma0./sigma)./betaf(102:162)).^x;
xi=log(sigma0./sigma)./betaf(102:162);
phi=xi./(1+xi);
ix=phi<0.38;
dt(~ix,10)=nan;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_SS187.H',1500,1);
depthseismic=transpose(linspace(0,6000*3.28084,601));

figure
plot(0.3048e6./data.dt_ed5,data.depth*0.3048,'g',vel(1:601),depthseismic*0.3048,'k')
title('SS187')
xlabel('velocity (m/s)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
for i=1:10
    plot(0.3048e6./dt(:,i),betadepth(1:61),'--')
end
hold off
legend('sonic','seismic','hydro','10 ppg','11 ppg','12 ppg','13 ppg','14 ppg','15 ppg',...
    '16 ppg','17 ppg','fracture')

%SS191
data=loadlas('177114136300_Orig+Edit+RckPhys.las');
kb=94;
wdepth=72;
data.depth=data.depth-kb-wdepth;

betadepthft=betadepth(1:61)*3.28084;
s=0.000005432*betadepthft.^2+0.8783*betadepthft+0.455*wdepth+14.7;
sigma=s-0.975*s;
dt(:,10)=dtm*(1+log(sigma0./sigma)./betaf(203:263)).^x;
xi=log(sigma0./sigma)./betaf(203:263);
phi=xi./(1+xi);
ix=phi<0.38;
dt(~ix,10)=nan;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_SS191.H',1500,1);
depthseismic=transpose(linspace(0,6000*3.28084,601));

figure
plot(0.3048e6./data.dtln_ed2,data.depth*0.3048,'g',vel(1:601),depthseismic*0.3048,'k')
title('SS191')
xlabel('velocity (m/s)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
for i=1:10
    plot(0.3048e6./dt(:,i),betadepth(1:61),'--')
end
hold off
legend('sonic','seismic','hydro','10 ppg','11 ppg','12 ppg','13 ppg','14 ppg','15 ppg',...
    '16 ppg','17 ppg','fracture')

%ST200
data=loadlas('177154042100_Orig+Edit+RckPhys.las');
kb=93;
wdepth=131;
data.depth=data.depth-kb-wdepth;

betadepthft=betadepth(1:61)*3.28084;
s=0.000005432*betadepthft.^2+0.8783*betadepthft+0.455*wdepth+14.7;
sigma=s-0.975*s;
dt(:,10)=dtm*(1+log(sigma0./sigma)./betaf(304:364)).^x;
xi=log(sigma0./sigma)./betaf(304:364);
phi=xi./(1+xi);
ix=phi<0.38;
dt(~ix,10)=nan;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_ST200.H',1500,1);
depthseismic=transpose(linspace(0,6000*3.28084,601));

figure
plot(0.3048e6./data.dt_ed4,data.depth*0.3048,'g',vel(1:601),depthseismic*0.3048,'k')
title('ST200')
xlabel('velocity (m/s)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
for i=1:10
    plot(0.3048e6./dt(:,i),betadepth(1:61),'--')
end
hold off
legend('sonic','seismic','hydro','10 ppg','11 ppg','12 ppg','13 ppg','14 ppg','15 ppg',...
    '16 ppg','17 ppg','fracture')

%ST143
data=loadlas('177154098500_Orig+Edit+RckPhys.las');
kb=97;
wdepth=75;
data.depth=data.depth-kb-wdepth;

betadepthft=betadepth(1:61)*3.28084;
s=0.000005432*betadepthft.^2+0.8783*betadepthft+0.455*wdepth+14.7;
sigma=s-0.975*s;
dt(:,10)=dtm*(1+log(sigma0./sigma)./betaf(405:465)).^x;
xi=log(sigma0./sigma)./betaf(405:465);
phi=xi./(1+xi);
ix=phi<0.38;
dt(~ix,10)=nan;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_ST143.H',1500,1);
depthseismic=transpose(linspace(0,6000*3.28084,601));

figure
plot(0.3048e6./data.dt_ed1,data.depth*0.3048,'g',vel(1:601),depthseismic*0.3048,'k')
title('ST143')
xlabel('velocity (m/s)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
for i=1:10
    plot(0.3048e6./dt(:,i),betadepth(1:61),'--')
end
hold off
legend('sonic','seismic','hydro','10 ppg','11 ppg','12 ppg','13 ppg','14 ppg','15 ppg',...
    '16 ppg','17 ppg','fracture')

%ST168
data=loadlas('177154117301.las');
kb=130;
wdepth=70;
data.dept_f=data.dept_f-kb-wdepth;

betadepthft=betadepth(1:61)*3.28084;
s=0.000005432*betadepthft.^2+0.8783*betadepthft+0.455*wdepth+14.7;
sigma=s-0.975*s;
dt(:,10)=dtm*(1+log(sigma0./sigma)./betaf(506:566)).^x;
xi=log(sigma0./sigma)./betaf(506:566);
phi=xi./(1+xi);
ix=phi<0.38;
dt(~ix,10)=nan;

vel=sepread('/net/server2/homes/sep/huyle/dragon/velocities/velocity_ST168.H',1500,1);
depthseismic=transpose(linspace(0,6000*3.28084,601));

figure
plot(0.3048e6./data.dtco_us_f,data.dept_f*0.3048,'g',vel(1:601),depthseismic*0.3048,'k')
title('ST168')
xlabel('velocity (m/s)')
ylabel('depth (m)')
set(gca,'Ydir','reverse')
hold on
for i=1:10
    plot(0.3048e6./dt(:,i),betadepth(1:61),'--')
end
hold off
legend('sonic','seismic','hydro','10 ppg','11 ppg','12 ppg','13 ppg','14 ppg','15 ppg',...
    '16 ppg','17 ppg','fracture')

end