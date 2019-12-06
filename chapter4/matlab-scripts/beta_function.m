function [beta,smec_fr]=beta_function(depth,ages,burial_depths,bht,bht_depths,params)

[burial_depths,idx]=sort(burial_depths); ages=ages(idx);
ages=[0;ages(:)]; burial_depths=[0;burial_depths(:)];
[burial_depths,i]=unique(burial_depths);
ages=ages(i);
geol_time=interp1(burial_depths,ages,depth,'linear','extrap');
smec_fr=cellfun(@(t1) params.smec_ini*exp(-integral(@(t) kin_fun(t,ages,burial_depths,bht,bht_depths,...
    params.arr,params.del_e,params.R),0,t1)),num2cell(geol_time)); % Smectite fraction using kinetics
beta=params.beta0*smec_fr+params.beta1*(1-smec_fr);

end

function res=kin_fun(time,ages,burial_depths,bht,bht_depths,arr,del_e,R)
z=interp1(ages,burial_depths,time,'linear','extrap');
temp=(interp1(bht_depths,bht,z,'linear','extrap')+459.67)*(5/9);
res=arr*exp(-del_e./(R*temp));
end
