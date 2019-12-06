function populatePPandRPT3d()
	nx=151;ox=650000;dx=200;
	ny=151;oy=-20000;dy=200;
    x=transpose(linspace(ox,ox+(nx-1)*dx,nx));
    y=transpose(linspace(oy,oy+(ny-1)*dy,ny));
    
	plio=textread('../surfaces/E-Dragon_Top_Pliocene_pts');
	plio=plio*0.3048;
	mio=textread('../surfaces/E-Dragon_Top_Miocene_pts');
	mio=mio*0.3048;
	
    ages0=[2.58;5.33];
    
	mindepth=0;
    maxdepth=6000;
    dz=200;
    depth=transpose(mindepth:dz:maxdepth);
    nz=size(depth,1);

    dt=zeros(nx,ny,nz,19);
    beta=zeros(nx,ny,nz);
    smec_fr=zeros(nx,ny,nz);
    
    for i=1:nx
		for j=1:ny
%			fprintf('ix=%d iy=%d\n',i,j)
			z_plio=getHistoryDepth(x(i),y(j),plio)*3.28084;
			z_mio=getHistoryDepth(x(i),y(j),mio)*3.28084;
			historydepths0=[z_plio;z_mio];
			[dt(i,j,:,:),beta(i,j,:),smec_fr(i,j,:)]=oneRPT(ages0,historydepths0,depth);
		end
    end
    dt=0.3048e6./dt;
    sepwrite('RPT3d.H',dt,[nx;ny;nz;19],[ox;oy;mindepth;1],[dx;dy;dz;1]);
    sepwrite('beta3d.H',beta,[nx;ny;nz],[ox;oy;mindepth],[dx;dy;dz]);
    sepwrite('smecfr3d.H',smec_fr,[nx;ny;nz],[ox;oy;mindepth],[dx;dy;dz]);
end

