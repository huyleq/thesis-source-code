function [MioDepth,PlioDepth]=getMioPlioDepth(x,y)
% x y are inline (east-west) and crossline (north-south) location in meters
    xft=x*3.28084;
    yft=y*3.28084;
    nx=size(x,1);
	MioDepth=zeros(nx,1);
	PlioDepth=MioDepth;

	sMio=textread('../E-Dragon_Top_Miocene'); % in feet
	sPlio=textread('../E-Dragon_Top_Pliocene'); % in feet
	[minvaly,k]=min(abs(sMio(:,2)-yft));
	for i=1:nx
		mini=k;
		minvalx=abs(sMio(mini,1)-xft(i));
		MioDepth(i)=sMio(mini,3);
		j=k;
		while j<=size(sMio,1) && abs(sMio(j,2)-yft)==minvaly
			if abs(sMio(j,1)-xft(i))<minvalx
				MioDepth(i)=sMio(j,3);
				minvalx=abs(sMio(j,1)-xft(i));
				mini=j;
			end
			j=j+1;
		end
		PlioDepth(i)=sPlio(mini,3);
	end
end

