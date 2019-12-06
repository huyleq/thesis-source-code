function Tswi=tswi_calc(lat,water_depth)
% SWIT % Hantschel&Kauerauf, pg. 128
Tf = -1.90-7.64*10^-4*(water_depth*0.3048);
a = 4.63+8.84*10^-4*lat-7.24*10^-4*lat^2;
b = -0.32+1.04*10^-4*lat-7.08*10^-5*lat^2;
Tswi=Tf+exp(a+b*log(water_depth*0.3048));% in 0C
Tswi=Tswi*(9/5)+32; % in 0F
end