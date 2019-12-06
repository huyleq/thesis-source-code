function z=extract2DHor(hor,x,y)
	[val,k]=min(abs(hor(:,2)-y));
	y=hor(k,2);
	x1=hor(hor(:,2)==y,1);
	z1=hor(hor(:,2)==y,3);
	z=interp1(x1,z1,x,'linear','extrap');
end
