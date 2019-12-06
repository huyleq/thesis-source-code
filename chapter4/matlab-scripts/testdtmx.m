function [dtm,x]=testdtmx(wellname)
if strcmp(wellname,'SS160')
    %SS160
    [~,~,~,~,~,shaledt1]=selectsandshale('../wells/177114095100_Orig+Edit+RckPhys.las',4,5,95,17,6000,7000);
    [~,~,~,~,~,shalephi1]=selectsandshale('../wells/177114095100_Orig+Edit+RckPhys.las',4,5,95,9,6000,7000);
    [~,~,~,~,~,shaledt2]=selectsandshale('../wells/177114095100_Orig+Edit+RckPhys.las',4,5,95,17,2945*3.28084,4308*3.28084);
    [~,~,~,~,~,shalephi2]=selectsandshale('../wells/177114095100_Orig+Edit+RckPhys.las',4,5,95,9,2945*3.28084,4308*3.28084);
    shaledt_ss160=[shaledt1;shaledt2];
    shalephi_ss160=[shalephi1;shalephi2]*0.01;
    shaledt=shaledt_ss160;
    shalephi=shalephi_ss160;
end

if strcmp(wellname,'SS187')
	%SS187
	[~,~,~,~,~,shaledt1]=selectsandshale('../wells/177114129700_Orig+Edit+RckPhys.las',20,5,95,37,1000*3.28084,3890*3.28084);
	[~,~,~,~,~,shalephi1]=selectsandshale('../wells/177114129700_Orig+Edit+RckPhys.las',20,5,95,23,1000*3.28084,3890*3.28084);
    %[~,~,~,~,~,shaledt2]=selectsandshale('../wells/177114129700_Orig+Edit+RckPhys.las',2,5,95,37,4000*3.28084,4800*3.28084);
	%[~,~,~,~,~,shalephi2]=selectsandshale('../wells/177114129700_Orig+Edit+RckPhys.las',2,5,95,30,4000*3.28084,4800*3.28084);
    shaledt=[shaledt1];%;shaledt2];
    shalephi=[shalephi1];%;shalephi2];
    shalephi=shalephi(shalephi<0.5);
	shaledt=shaledt(shalephi<0.5);
end

if strcmp(wellname,'SS191')
	%SS191
	[~,~,~,~,~,shaledt_ss191]=selectsandshale('../wells/177114136300_Orig+Edit+RckPhys.las',2,5,95,31,0,0);
	[~,~,~,~,~,shalephi_ss191]=selectsandshale('../wells/177114136300_Orig+Edit+RckPhys.las',2,5,95,7,0,0);
    shaledt=shaledt_ss191;
    shalephi=shalephi_ss191;
end

if strcmp(wellname,'ST200')
	%ST200
	[~,~,~,~,~,shaledt_st200]=selectsandshale('../wells/177154042100_Orig+Edit+RckPhys.las',5,5,95,21,2900*3.2084,4237*3.2084);
	[~,~,~,~,~,shalephi_st200]=selectsandshale('../wells/177154042100_Orig+Edit+RckPhys.las',5,5,95,18,2900*3.2084,4237*3.2084);
    shaledt=shaledt_st200;
    shalephi=shalephi_st200;
end

if strcmp(wellname,'ST143')
	%ST143
	[~,~,~,~,~,shaledt_st143]=selectsandshale('../wells/177154098500_Orig+Edit+RckPhys.las',4,5,95,29,0,15770);
	[~,~,~,~,~,shalephi_st143]=selectsandshale('../wells/177154098500_Orig+Edit+RckPhys.las',4,5,95,16,0,15770);
	shalephi_st143=shalephi_st143*0.01;
    shaledt=shaledt_st143;
    shalephi=shalephi_st143;
end

p=0.07
shaledt=1/(1-p)*shaledt;

n=size(shaledt,1);
G=[-log(shaledt),ones(n,1)];
d=log(1-shalephi);
m=(transpose(G)*G)\(transpose(G)*d);
x=1/m(1);
dtm=exp(m(2)/m(1));

% figure
% scatter(log(shaledt),log(1-shalephi))
% xlabel('log(dt)')
% ylabel('log(1-phi)')
rho=corr(log(shaledt),log(1-shalephi));
fprintf('corr coef between log(1-phi) and log(dt) is %f\n',rho);

% phi=transpose(linspace(0,0.7,1000));
% figure
% scatter(shalephi,shaledt)
% hold on
% plot(phi,dtm*(1-phi).^(-x))
fprintf('%s dtm=%f vm=%f x=%f\n',wellname,dtm,0.3028e6/dtm,x);

obj1=sum((G*m-d).^2);
fprintf('obj func at mimimum is %f\n',obj1);

m=[1/2.1;log(50)/2.1];
obj2=sum((G*m-d).^2);
fprintf('obj func at dtm=50 x=2.1 is %f\n',obj2);

m=[1/2.1;log(60)/2.1];
obj3=sum((G*m-d).^2);
fprintf('obj func at dtm=60 x=2.1 is %f\n',obj3);

m=[1/2.3;log(50)/2.1];
obj4=sum((G*m-d).^2);
fprintf('obj func at dtm=50 x=2.3 is %f\n',obj4);

m=[1/2.3;log(60)/2.1];
obj5=sum((G*m-d).^2);
fprintf('obj func at dtm=60 x=2.3 is %f\n',obj5);


