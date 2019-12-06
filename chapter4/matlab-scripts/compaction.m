function [phi0,c]=compaction(phi,depth)
	%depth in km

	n=size(phi,1);
	y=log(phi);
	z=depth;
	A=ones(n,2);
	A(:,2)=-z;
	x=linsolve(transpose(A)*A,transpose(A)*y);
	phi0=exp(x(1));
	c=x(2);
	
%	figure
%	hold on
%	scatter(depth,y)
%	plot(depth,x(1)-c*depth)
%	hold off
%	xlabel('depth')
%	ylabel('log phi')
%
%	figure
%	plot(phi,depth,'r',phi0*exp(-c*depth),depth,'b')
%    set(gca,'Ydir','reverse')
	
%	minphi0=0.47;
%	maxphi0=0.65;
%	dphi0=maxphi0-minphi0;
%	minc=0.27;
%	maxc=0.51;
%	dc=maxc-minc;
%
%	n=100;
%	phi0=minphi0;
%	c=minc;
%	error=sum((phi-phi0*exp(-c*depth)).^2);
%	for i=1:n
%		phi0trial=minphi0+dphi0/n*i;
%		for j=1:n
%			ctrial=minc+dc/n*j;
%			errortrial=sum((phi-phi0trial*exp(-ctrial*depth)).^2);
%			if(errortrial<error)
%				phi0=phi0trial;
%				c=ctrial;
%			end
%		end
%	end

end
