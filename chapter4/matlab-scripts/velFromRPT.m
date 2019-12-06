function vel=velFromRPT(velfile,mudfile,rptfile)
    mud=textread(mudfile);
    mud(:,1)=mud(:,1)/3.28084;
    [nx,ox,dx]=get_par3(rptfile,'n1','o1','d1');
    [nz,oz,dz]=get_par3(rptfile,'n2','o2','d2');
    nrpt=get_par1(rptfile,'n3');
    rpt=sepread(rptfile,nx*nz,nrpt);
    rpt=reshape(rpt,[nx,nz,nrpt]);
    vel0=zeros(nx,nz);
    for i=1:nx
        vel0(i,:)=velFromRPT1(mud(:,2),mud(:,1),rpt(i,:,:),nz,oz,dz);
    end
    vel=zeros(nx,nz+1);
	vel(:,1)=1500;
    vel(:,2:nz+1)=vel0;
    sepwrite(velfile,vel,[nx;nz+1],[ox;oz],[dx;dz]);
%    depth=transpose(oz:dz:oz+dz*(nz-1));
%    figure
%    plot(vel(500,:),depth,'linewidth',3)
%    hold on;
%    for i=1:nrpt
%        plot(rpt(500,:,i),depth)
%    end
%    hold off;
%    set(gca,'Ydir','reverse');
end

function vel=velFromRPT1(mud,muddepth,rpt,nz,mindepth,dz)
% depths in meter of mudweight [8.7625;9;9.5;10;10.5;11;11.5;12;12.5;13;13.5;14;14.5;15;15.5;16;16.5;17;17.5;18;18.5] ppg
    ppgrad=[8.7625;9;9.5;10;10.5;11;11.5;12;12.5;13;13.5;14;14.5;15;15.5;16;16.5;17];
    nmud=size(muddepth,1);
    imuddepth=zeros(nmud,1);
    for i=1:nmud-1
        imuddepth(i)=floor((0.5*(muddepth(i)+muddepth(i+1))-mindepth)/dz+1);
    end
    imuddepth(nmud)=floor((muddepth(nmud)-mindepth)/dz+1);
    vel(1:imuddepth(1))=rpt(1,1:imuddepth(1),find(ppgrad==mud(1)));
    for i=2:nmud
        vel(imuddepth(i-1):imuddepth(i))=rpt(1,imuddepth(i-1):imuddepth(i),find(ppgrad==mud(i)));
    end
    vel(imuddepth(nmud):nz)=rpt(1,imuddepth(nmud):nz,find(ppgrad==mud(nmud)));
    vel(vel<1500)=1500;
    vel=smooth(vel,20);
end
