function porocorrect
    data=loadlas('177114129700_Orig+Edit+RckPhys.las');
    kb=96.3;
    wdepth=60;
    
	n=size(data.depth,1);
	depth=0.3048*(data.depth-kb-wdepth);
	neuporo=data.data____;
	vshale=data.vcl;
	poro=zeros(n,1);
	
	for i=1:n
		factor=1;
		if(depth<=1524)
			factor=0.12;
		elseif(depth<=3048)
			factor=0.1;
		else
			factor=0.06;
		end
		if(isfinite(vshale(i)) && isfinite(neuporo(i)))
			poro(i)=neuporo(i)-factor*vshale(i);
		end
	end

%	figure
%	plot(movmean(neuporo,1000),depth,movmean(poro,1000),depth)

	poro=poro(isfinite(poro));
	depth=depth(isfinite(poro));
	poro=movmean(poro,1000);
	o=min(depth);
	d=depth(2)-depth(1);
	b=round((1101-o)/d);
	e=round((3839-o)/d);
	poro0=poro(b:e);
	depth0=depth(b:e);
	plot(poro0,depth0)
 
	index=find(poro0);
	poro1=poro0(index);
	depth1=depth0(index);
	[phi0,c]=compaction(poro1,0.001*depth1)
	figure
	plot(poro1,0.001*depth1,phi0*exp(-c*(0.001*depth1)),0.001*depth1)
	
%	index1=find(depth1<2112);
%	depth2=depth1(index1);
%	poro2=poro1(index1);
%	[phi0,c]=compaction(poro2,0.001*depth2)
%	figure
%	plot(poro2,0.001*depth2,phi0*exp(-c*(0.001*depth2)),0.001*depth2)
	
%	index1=find(2112<depth1<3882);
%	depth2=depth1(index1);
%	poro2=poro1(index1);
%	[phi0,c]=compaction(poro2,0.001*depth2)
%	figure
%	plot(poro2,0.001*depth2,phi0*exp(-c*(0.001*depth2)),0.001*depth2)
%	
%	index1=find(depth1>3882);
%	depth2=depth1(index1);
%	poro2=poro1(index1);
%	[phi0,c]=compaction(poro2,0.001*depth2)
%	figure
%	plot(poro2,0.001*depth2,phi0*exp(-c*(0.001*depth2)),0.001*depth2)
   
%	figure
%	plot(neuporo(index1),depth2,'r',poro1(index1),depth2,'b')
%    set(gca,'Ydir','reverse')

%	figure
%    hold on
%    plot(data.gr,depth,'b')
%    plot(data.grr,depth,'g')
%    plot(data.grs,depth,'k')
%    set(gca,'Ydir','reverse')
%    hold off
%	
%	figure
%    plot(data.vcl,depth,'k')
%    set(gca,'Ydir','reverse')
%    
%	figure
%    plot(data.tnph,depth,'b')
%    hold on
%    plot(data.data____,depth,'g') % use this one not the other
%    hold off
%    set(gca,'Ydir','reverse')
%    xlabel('Neutron porosity')

%	figure
%	plot(neuporo,depth,'r')
%    set(gca,'Ydir','reverse')
%	figure 
%	plot(poro,depth,'b')
%    set(gca,'Ydir','reverse')
end
