function populatePPandRPT()
    y=-10320;
    x=transpose(linspace(648867,648867+25*1066,1067));
    nx=size(x,1);
    
	Plio=textread('../wells/E-Dragon_Top_Pliocene');
	zPlio=extract2DHor(Plio/3.28084,x,y);
	Mio=textread('../wells/E-Dragon_Top_Miocene');
	zMio=extract2DHor(Mio/3.28084,x,y);
	UMio2=textread('../wells/NewModel_manylayer_Miocene_Upper2_Depth_0Ma_Top_xyz.txt');
	zUMio2=interp1(UMio2(:,1),UMio2(:,3),x);
	zUMio2(173:196)=zUMio2(173)+(zUMio2(196)-zUMio2(173))/(196.-173.+1.)*linspace(1,23,24)';
	sepwrite('../line1/TopPlio1.H',zPlio,[nx;1],[0;0],[25;1]);
	sepwrite('../line1/TopMio1.H',zMio,[nx;1],[0;0],[25;1]);
	sepwrite('../line1/Mio1.H',zUMio2,[nx;1],[0;0],[25;1]);
	
    ages0=[2.58;5.33;10.7];
    
	mindepth=0;
    maxdepth=10000;
    dz=25;
    depth=transpose(mindepth:dz:maxdepth);
    nz=size(depth,1);

    dt=zeros(nx,nz,19);
    beta=zeros(nx,nz);
    smec_fr=beta;
    
    for i=1:nx
        historydepths0=[zPlio(i);zMio(i);zUMio2(i)];
        [dt(i,:,:),beta(i,:),smec_fr(i,:)]=oneRPT(ages0,historydepths0,depth);
    end
    dt=0.3048e6./dt;
    sepwrite('../line1/RPT1.H',dt,[nx;nz;19],[min(x);mindepth;1],[25;25;1]);
    sepwrite('../line1/beta1.H',beta,[nx;nz],[min(x);mindepth],[25;25]);
    sepwrite('../line1/smecfr1.H',smec_fr,[nx;nz],[min(x);mindepth],[25;25]);
end

