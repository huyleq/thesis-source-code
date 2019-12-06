function ageandtemp
kb_ST168=130; wd_ST168=70; 
lat_ST168=28.57276479;
bht_depths=[17060;17552;19361];
bht_depths=[0;bht_depths-(wd_ST168+kb_ST168)];
Tswi_ST168=tswi_calc(lat_ST168,wd_ST168);

bht_ST168=[265;280;300];
temp_ST168=[Tswi_ST168;bht_ST168];
ages=[2.58;5.33];
historydepths=[5.9538e3,7.2277e3,7.4678e3,7.7944e3,6.9199e3,6.9781e3;...
			   8.2555e3,1.0304e4,1.0755e4,1.1877e4,9.5231e3,9.7039e3];	

depth=0:20:7000;
depth=transpose(depth);
depthft=depth*3.28084;
lat=[28.67544544;28.62160218;28.58814302;28.45741797;28.60183226;28.57276479];
wd=[50;60;72;131;75;70]; 
kb=[94.5;96.3;94;93;97;130];


Tswi=zeros(6,1);
for i=1:6
	Tswi(i)=tswi_calc(lat(i),wd(i));
end
ages1=[0;ages];
historydepths1=[zeros(1,6);historydepths];
depth2=transpose(linspace(0,10000*3.28084,1000));	
temp=interp1(bht_depths,temp_ST168,depth2,'linear','extrap');
time=zeros(1000,6);
for i=1:6
	time(:,i)=interp1(historydepths1(:,i),ages1,depth2,'linear','extrap');
end

wellname=['SS160','SS187','SS191','ST200','ST143','ST168'];
colors=[0 0 0;1 0 1;0 1 1;1 0 0;0 1 0;0 0 1];
figure('units','normalized','outerposition',[0 0 1 1])
hold on
for i=1:6
	plot(time(:,i),depth2*0.3048,'Color',colors(i,:))
%	text(time(1000,i)-0.5,depth2(1000)*0.3048,wellname((i-1)*5+1:i*5),'Color','black','Fontsize',10)
end
for i=1:6
	scatter(ages(1),historydepths(1,i)*0.3048,100,'s','MarkerEdgeColor',[0 .7 .7],'MarkerFaceColor',[0 .7 .7])
	scatter(ages(2),historydepths(2,i)*0.3048,100,'d','MarkerEdgeColor',[0 .7 .7],'MarkerFaceColor',[0 .7 .7])
end
set(gca,'Ydir','reverse')
legend('SS160','SS187','SS191','ST200','ST143','ST168','Top Pliocene','Top Miocene')
xlabel('Age (My.)')
ylabel('Depth (m)')
ylim([0 11000])
hold off
box on
export_fig('../Fig/ages.pdf')

figure('units','normalized','outerposition',[0 0 1 1])
hold on
for i=1:6
	plot((temp+Tswi(i)-Tswi_ST168-32)*5/9,depth2*0.3048,'Color',colors(i,:))
end
scatter((temp_ST168(2:end)-32)*5/9,bht_depths(2:end)*0.3048,100,'MarkerEdgeColor',[0 .7 .7],'MarkerFaceColor',[0 .7 .7])
set(gca,'Ydir','reverse')
legend('SS160','SS187','SS191','ST200','ST143','ST168','BHT data')
xlabel('Temperature (C)')
ylabel('Depth (m)')
ylim([0 11000])
hold off
box on
export_fig('../Fig/temps.pdf')

end
