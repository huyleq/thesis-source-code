function [dtm0,x0]=dtmx
%SS160
[~,~,~,~,~,shaledt1]=selectsandshale('../wells/177114095100_Orig+Edit+RckPhys.las',4,5,95,17,6000,7000);
[~,~,~,~,~,shalephi1]=selectsandshale('../wells/177114095100_Orig+Edit+RckPhys.las',4,5,95,9,6000,7000);
[~,~,~,~,~,shaledt2]=selectsandshale('../wells/177114095100_Orig+Edit+RckPhys.las',4,5,95,17,2945*3.28084,4308*3.28084);
[~,~,~,~,~,shalephi2]=selectsandshale('../wells/177114095100_Orig+Edit+RckPhys.las',4,5,95,9,2945*3.28084,4308*3.28084);
shaledt_ss160=[shaledt1;shaledt2];
shalephi_ss160=[shalephi1;shalephi2]*0.01;

%SS187
[~,~,~,~,~,shaledt_ss187]=selectsandshale('../wells/177114129700_Orig+Edit+RckPhys.las',20,5,95,37,1000*3.28084,3890*3.28084);
[~,~,~,~,~,shalephi_ss187]=selectsandshale('../wells/177114129700_Orig+Edit+RckPhys.las',20,5,95,23,1000*3.28084,3890*3.28084);
shalephi_ss187=shalephi_ss187(shalephi_ss187<0.6);
shaledt_ss187=shaledt_ss187(shalephi_ss187<0.6);

%SS191
[~,~,~,~,~,shaledt_ss191]=selectsandshale('../wells/177114136300_Orig+Edit+RckPhys.las',2,5,95,31,0,0);
[~,~,~,~,~,shalephi_ss191]=selectsandshale('../wells/177114136300_Orig+Edit+RckPhys.las',2,5,95,7,0,0);

%ST200
[~,~,~,~,~,shaledt_st200]=selectsandshale('../wells/177154042100_Orig+Edit+RckPhys.las',5,5,95,21,2900*3.2084,4237*3.2084);
[~,~,~,~,~,shalephi_st200]=selectsandshale('../wells/177154042100_Orig+Edit+RckPhys.las',5,5,95,18,2900*3.2084,4237*3.2084);

%ST143
[~,~,~,~,~,shaledt_st143]=selectsandshale('../wells/177154098500_Orig+Edit+RckPhys.las',4,5,95,29,0,15770);
[~,~,~,~,~,shalephi_st143]=selectsandshale('../wells/177154098500_Orig+Edit+RckPhys.las',4,5,95,16,0,15770);
shalephi_st143=shalephi_st143*0.01;

figure('units','normalized','outerposition',[0 0 1 1])
scatter(shalephi_ss160,shaledt_ss160*3.28084e-6,'k')
hold on
scatter(shalephi_ss187,shaledt_ss187*3.28084e-6,'m')
scatter(shalephi_ss191,shaledt_ss191*3.28084e-6,'c')
scatter(shalephi_st200,shaledt_st200*3.28084e-6,'r')
scatter(shalephi_st143,shaledt_st143*3.28084e-6,'g')

phi=transpose(linspace(0,0.6,1000));

[dtm,x]=testdtmx('SS160');
plot(phi,(dtm*(1-phi).^(-x))*3.28084e-6,'k')

[dtm,x]=testdtmx('SS187');
plot(phi,(dtm*(1-phi).^(-x))*3.28084e-6,'m')

[dtm,x]=testdtmx('SS191');
plot(phi,(dtm*(1-phi).^(-x))*3.28084e-6,'c')

[dtm,x]=testdtmx('ST200');
plot(phi,(dtm*(1-phi).^(-x))*3.28084e-6,'r')

[dtm,x]=testdtmx('ST143');
plot(phi,(dtm*(1-phi).^(-x))*3.28084e-6,'g')

shaledt=[shaledt_ss160;shaledt_ss187;shaledt_ss191;shaledt_st200;shaledt_st143];

p=0.07
shaledt=1/(1-p)*shaledt;

shalephi=[shalephi_ss160;shalephi_ss187;shalephi_ss191;shalephi_st200;shalephi_st143];
n=size(shaledt,1);
G=[-log(shaledt),ones(n,1)];
d=log(1-shalephi);
m=(transpose(G)*G)\(transpose(G)*d);
x=1/m(1);
dtm=exp(m(2)/m(1));

plot(phi,(dtm*(1-phi).^(-x))*3.28084e-6,'b')
fprintf('For all wells dtm=%f x=%f\n',dtm,x);

xlabel('Porosity')
ylabel('Slowness (s/m)')
ylim([0 200*3.28084e-6])
legend('SS160','SS187','SS191','ST200','ST143','Location','southeast')
box on
hold off        
%export_fig('../Fig/dtmx.pdf')
