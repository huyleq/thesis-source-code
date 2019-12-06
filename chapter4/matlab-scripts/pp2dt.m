function [dt,s]=pp2dt(pp,dtm,x,sigma0,beta,wdepth,depth)
s=0.000005432*depth.^2+0.8783*depth+0.455*wdepth+14.7;
sigma=s-pp;
dt=dtm*(1+log(sigma0./sigma)./beta).^x;
end