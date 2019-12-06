function z=getHistoryDepth(x,y,s)
	sx=s(:,1)-x;
	sy=s(:,2)-y;
	[m,i]=min(sx.*sx+sy.*sy);
	z=s(i,3);
end
	
