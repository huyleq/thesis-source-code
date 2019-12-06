function [s,sigma,pp]=dt2pp(dt,dtm,x,sigma0,beta,wdepth,depth)
phig=(dt/dtm).^(-1/x);
phi=1-phig;

s=0.000005432*depth.^2+0.8783*depth+0.455*wdepth+14.7;

si=phi./phig;
sigma=sigma0*exp(-si.*beta);

pp=s-sigma;
end