function vel=velFromRPT1(mud,muddepth,rpt,nz,mindepth,dz)
	vel=zeros(nz,1);
% depths in meter of mudweight [8.7625;9;9.5;10;10.5;11;11.5;12;12.5;13;13.5;14;14.5;15;15.5;16;16.5;17;17.5;18;18.5] ppg
    ppgrad=[8.7625;9;9.5;10;10.5;11;11.5;12;12.5;13;13.5;14;14.5;15;15.5;16;16.5;17];
    nmud=size(muddepth,1);
    imuddepth=zeros(nmud,1);
    for i=1:nmud-1
        imuddepth(i)=floor((0.5*(muddepth(i)+muddepth(i+1))-mindepth)/dz+1);
    end
    imuddepth(nmud)=floor((muddepth(nmud)-mindepth)/dz+1);
    vel(1:imuddepth(1))=rpt(1,1,1:imuddepth(1),find(ppgrad==mud(1)));
    for i=2:nmud
	    vel(imuddepth(i-1):imuddepth(i))=rpt(1,1,imuddepth(i-1):imuddepth(i),find(ppgrad==mud(i)));
    end
    vel(imuddepth(nmud):nz)=rpt(1,1,imuddepth(nmud):nz,find(ppgrad==mud(nmud)));
    vel(vel<1500)=1500;
    vel=smooth(vel,20);
end
