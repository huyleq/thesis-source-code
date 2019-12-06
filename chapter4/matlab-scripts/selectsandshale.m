function [sanddepth,sandgamma,sand,shaledepth,shalegamma,shale]=selectsandshale(file,gammalog,sandprctile,shaleprctile,sthlog,mindepthinfeet,maxdepthinfeet)
[~,data,~,~]=loadlas(file);
if mindepthinfeet~=0
    m=floor((mindepthinfeet-data(1,1))/(data(2,1)-data(1,1)))+1;
else
    m=1;
end
if maxdepthinfeet~=0
    n=floor((maxdepthinfeet-data(1,1))/(data(2,1)-data(1,1)))+1;
else
    n=size(data,1);
end
sanddepth0=zeros(n,1);
sandgamma0=zeros(n,1)-1;
sand0=zeros(n,1)-1;
shaledepth0=zeros(n,1);
shalegamma0=zeros(n,1)-1;
shale0=zeros(n,1)-1;
nsand=0;
nshale=0;
w=120/(data(2,1)-data(1,1))+1;
gamma=movmean(data(m:n,gammalog),w);
sth=movmean(data(m:n,sthlog),w);
gammasand=prctile(gamma,sandprctile);
gammashale=prctile(gamma,shaleprctile);
for i=1:n-m+1
    if isfinite(sth(i)) && isfinite(gamma(i)) && sth(i)>0
        if gamma(i)<=gammasand
            sanddepth0(i)=data(i+m-1,1);
            sandgamma0(i)=gamma(i);
            sand0(i)=sth(i);
            nsand=nsand+1;
        end
        if gamma(i)>=gammashale
            shaledepth0(i)=data(i+m-1,1);
            shalegamma0(i)=gamma(i);
            shale0(i)=sth(i);
            nshale=nshale+1;
        end
    end
end
sanddepth=zeros(nsand,1);
sandgamma=sanddepth;
sand=sanddepth;
shaledepth=zeros(nshale,1);
shalegamma=shaledepth;
shale=shaledepth;
isand=0;
ishale=0;
for i=1:n-m+1
    if sanddepth0(i)~=0 
        isand=isand+1;
        sanddepth(isand)=sanddepth0(i);
        sandgamma(isand)=sandgamma0(i);
        sand(isand)=sand0(i);
    end
    if shaledepth0(i)~=0 
        ishale=ishale+1;
        shaledepth(ishale)=shaledepth0(i);
        shalegamma(ishale)=shalegamma0(i);
        shale(ishale)=shale0(i);
    end
end

%figure
%plot(data(m:n,gammalog),data(m:n,1)*0.3048,gamma,data(m:n,1)*0.3048)
%set(gca,'Ydir','reverse')
%ylim([data(1,1)*0.3048 data(size(data,1),1)])
%hold on
%scatter(sandgamma,sanddepth*0.3048,'r')
%scatter(shalegamma,shaledepth*0.3048,'g')
%hold off
end
