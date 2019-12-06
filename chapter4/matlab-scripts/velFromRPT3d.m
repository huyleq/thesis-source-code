function vel=velFromRPT3d(velfile,mudfile,rptfile)
    mud=textread(mudfile);
    mud(:,1)=mud(:,1)/3.28084;
    [nx,ox,dx]=get_par3(rptfile,'n1','o1','d1');
    [ny,oy,dy]=get_par3(rptfile,'n2','o2','d2');
    [nz,oz,dz]=get_par3(rptfile,'n3','o3','d3');
    nrpt=get_par1(rptfile,'n4');
    rpt=sepread(rptfile,nx*ny*nz,nrpt);
    rpt=reshape(rpt,[nx,ny,nz,nrpt]);
    vel0=zeros(nx,ny,nz);
    for i=1:nx
		for j=1:ny
%			fprintf('ix=%d iy=%d\n',i,j)
			vel0(i,j,:)=velFromRPT1(mud(:,2),mud(:,1),rpt(i,j,:,:),nz,oz,dz);
		end
    end
    vel=zeros(nx,ny,nz+1);
	vel(:,:,1)=1500;
    vel(:,:,2:nz+1)=vel0;
    sepwrite(velfile,vel,[nx;ny;nz+1],[ox;oy;oz],[dx;dy;dz]);
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

