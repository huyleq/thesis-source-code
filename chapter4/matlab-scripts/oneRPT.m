function [dt,beta,smec_fr]=oneRPT(ages,historydepths,depth)
    kb_ST168=130; wd_ST168=70; 
    lat_ST168=28.57276479;
    bht_depths=[17060;17552;19361];
    bht_depths=[0;bht_depths-(wd_ST168+kb_ST168)];
    Tswi_ST168=tswi_calc(lat_ST168,wd_ST168);
    bht_ST168=[265;280;300];
%    ages=[2.58;5.33];
    depthft=depth*3.28084;
    params.smec_ini=1;
    params.arr=0.4*10^11; % Arrhenius factor in Myr^-1
    params.del_e=20; %Activation energy in kcal/mol
    params.R=1.986*10^-3; % Gas constant
    
    %       dtm     x       b0      b1  sigma
    %redo1  64.5    1.97    6.5     10  26e3
    %redo2  64.5    1.97    6       10  26e3
    %redo3  69.35   1.97    6.7     13  26e3
    %
    
    dtm=69.35; % in us/ft
    x=1.97;
    params.beta0=6.7; %6.5;
    params.beta1=13; %13; %14;
    sigma0=26000; %in psi
    
    [beta,smec_fr]=beta_function(depthft,ages,historydepths,[Tswi_ST168;bht_ST168],bht_depths,params);
    ppgrad=[8.7625;9;9.5;10;10.5;11;11.5;12;12.5;13;13.5;14;14.5;15;15.5;16;16.5;17];
    s=0.000005432*depthft.^2+0.8783*depthft+0.455*wd_ST168+14.7;
    ppfrac=0.975*s;
    for i=1:size(ppgrad,1)
        pp=ppgrad(i)*0.0519*depthft;
        sigma=s-pp;
        dt(:,i)=dtm*(1+log(sigma0./sigma)./beta).^x;
        xi=log(sigma0./sigma)./beta;
        phi=xi./(1+xi);
        ix=phi<0.38;
    %    dt(~ix,i)=nan;
    end
    sigma=s-ppfrac;
    dt(:,size(ppgrad,1)+1)=dtm*(1+log(sigma0./sigma)./beta).^x;
    xi=log(sigma0./sigma)./beta;
    phi=xi./(1+xi);
    ix=phi<0.38;
   % dt(~ix,10)=nan;
end
