function s=overburden(wdepth,rho,depth)
%s in psi, wdepth in ft, rho in g/cc, depth in ft
%rho and depth from water bottom
g=9.82;
pair=14.7;
pwater=0.000145038*g*1e3*wdepth*0.3048;
n=size(depth,1);
s=zeros(n,1);
s(1)=pair+pwater;
for i=2:n
    s(i)=s(i-1)+0.000145038*g*0.5*(rho(i)+rho(i-1))*1e3*(depth(i)-depth(i-1))*0.3048;
end
end