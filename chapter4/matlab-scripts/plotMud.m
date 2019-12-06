
set(0,'defaultLineLineWidth',3)
set(0,'DefaultAxesFontSize',15);
mud=textread('../wells/MudWeightAverage.txt');
mud0=textread('../wells/MudWeightAverage0.txt');
mud1=textread('../wells/MudWeightAverage1.txt');
mud2=textread('../wells/MudWeightAverage2.txt');
mud3=textread('../wells/MudWeightAverage3.txt');
mud4=textread('../wells/MudWeightAverage4.txt');
mud5=textread('../wells/MudWeightAverage5.txt');
mud6=textread('../wells/MudWeightAverage6.txt');
mud7=textread('../wells/MudWeightAverage7.txt');
mud187=textread('../wells/MudWeightSS187.txt');
mud191=textread('../wells/MudWeightSS191.txt');
mud143=textread('../wells/MudWeightST143.txt');
mud168=textread('../wells/MudWeightST168.txt');
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,2,2)
plot(mud(:,2),mud(:,1)/3.28084*0.001,'m',mud0(:,2),mud0(:,1)/3.28084*0.001,mud1(:,2),mud1(:,1)/3.28084*0.001,mud2(:,2),mud2(:,1)/3.28084*0.001,mud3(:,2),mud3(:,1)/3.28084*0.001,mud4(:,2),mud4(:,1)/3.28084*0.001,mud5(:,2),mud5(:,1)/3.28084*0.001,mud6(:,2),mud6(:,1)/3.28084*0.001,mud7(:,2),mud7(:,1)/3.28084*0.001)
set(gca,'Ydir','reverse')
ylim([0 10])
xlabel('Mud Weight (ppg)')
ylabel('Depth (km)')
title('(b)')
legend('average','try 0','try 1','try 2','try 3','try 4','try 5','try 6','try 7')

subplot(1,2,1)
hold on
plot(mud(:,2),mud(:,1)/3.28084*0.001,'m')
plot(mud187(:,2),mud187(:,1)/3.28084*0.001,'k')
plot(mud191(:,2),mud191(:,1)/3.28084*0.001,'r')
plot(mud143(:,2),mud143(:,1)/3.28084*0.001,'c')
plot(mud168(:,2),mud168(:,1)/3.28084*0.001,'b')
scatter(mud187(:,2),mud187(:,1)/3.28084*0.001,80,'filled','k')
scatter(mud191(:,2),mud191(:,1)/3.28084*0.001,80,'filled','r')
scatter(mud143(:,2),mud143(:,1)/3.28084*0.001,80,'filled','c')
scatter(mud168(:,2),mud168(:,1)/3.28084*0.001,80,'filled','b')
set(gca,'Ydir','reverse')
ylim([0 10])
xlabel('Mud Weight (ppg)')
ylabel('Depth (km)')
title('(a)')
legend('average','SS187','SS191','ST143','ST168')
box on
%export_fig('../Fig/mudweights.pdf')

%figure('units','normalized','outerposition',[0 0 0.2 1])
%h=plot(mud(:,2),mud(:,1)/3.28084*0.001,'m',mud0(:,2),mud0(:,1)/3.28084*0.001,'--',mud1(:,2),mud1(:,1)/3.28084*0.001,'--',mud2(:,2),mud2(:,1)/3.28084*0.001,'--',mud3(:,2),mud3(:,1)/3.28084*0.001,'--',mud4(:,2),mud4(:,1)/3.28084*0.001,'--',mud5(:,2),mud5(:,1)/3.28084*0.001,'--',mud6(:,2),mud6(:,1)/3.28084*0.001,'--')
%set(h,{'LineWidth'},{3;1;1;1;1;1;1;1})
%set(gca,'Ydir','reverse')
%ylim([0 10])
%xlabel('Mud Weight (ppg)')
%ylabel('Depth (km)')
%legend('average','try 1','try 2','try 3','try 4','try 5','try 6','try 7')
%export_fig('../Fig/trymudweight.pdf')
%
%figure('units','normalized','outerposition',[0 0 0.2 1])
%h=plot(mud(:,2),mud(:,1)/3.28084*0.001,'m--',mud0(:,2),mud0(:,1)/3.28084*0.001,mud1(:,2),mud1(:,1)/3.28084*0.001,'--',mud2(:,2),mud2(:,1)/3.28084*0.001,'--',mud3(:,2),mud3(:,1)/3.28084*0.001,'--',mud4(:,2),mud4(:,1)/3.28084*0.001,'--',mud5(:,2),mud5(:,1)/3.28084*0.001,'--',mud6(:,2),mud6(:,1)/3.28084*0.001,'--')
%set(h,{'LineWidth'},{1;3;1;1;1;1;1;1})
%set(gca,'Ydir','reverse')
%ylim([0 10])
%xlabel('Mud Weight (ppg)')
%ylabel('Depth (km)')
%legend('average','try 1','try 2','try 3','try 4','try 5','try 6','try 7')
%export_fig('../Fig/trymudweight1.pdf')
%
%figure('units','normalized','outerposition',[0 0 0.2 1])
%h=plot(mud(:,2),mud(:,1)/3.28084*0.001,'m--',mud0(:,2),mud0(:,1)/3.28084*0.001,'--',mud1(:,2),mud1(:,1)/3.28084*0.001,mud2(:,2),mud2(:,1)/3.28084*0.001,'--',mud3(:,2),mud3(:,1)/3.28084*0.001,'--',mud4(:,2),mud4(:,1)/3.28084*0.001,'--',mud5(:,2),mud5(:,1)/3.28084*0.001,'--',mud6(:,2),mud6(:,1)/3.28084*0.001,'--')
%set(h,{'LineWidth'},{1;1;3;1;1;1;1;1})
%set(gca,'Ydir','reverse')
%ylim([0 10])
%xlabel('Mud Weight (ppg)')
%ylabel('Depth (km)')
%legend('average','try 1','try 2','try 3','try 4','try 5','try 6','try 7')
%export_fig('../Fig/trymudweight2.pdf')
%
%figure('units','normalized','outerposition',[0 0 0.2 1])
%h=plot(mud(:,2),mud(:,1)/3.28084*0.001,'m--',mud0(:,2),mud0(:,1)/3.28084*0.001,'--',mud1(:,2),mud1(:,1)/3.28084*0.001,'--',mud2(:,2),mud2(:,1)/3.28084*0.001,mud3(:,2),mud3(:,1)/3.28084*0.001,'--',mud4(:,2),mud4(:,1)/3.28084*0.001,'--',mud5(:,2),mud5(:,1)/3.28084*0.001,'--',mud6(:,2),mud6(:,1)/3.28084*0.001,'--')
%set(h,{'LineWidth'},{1;1;1;3;1;1;1;1})
%set(gca,'Ydir','reverse')
%ylim([0 10])
%xlabel('Mud Weight (ppg)')
%ylabel('Depth (km)')
%legend('average','try 1','try 2','try 3','try 4','try 5','try 6','try 7')
%export_fig('../Fig/trymudweight3.pdf')
%
%figure('units','normalized','outerposition',[0 0 0.2 1])
%h=plot(mud(:,2),mud(:,1)/3.28084*0.001,'m--',mud0(:,2),mud0(:,1)/3.28084*0.001,'--',mud1(:,2),mud1(:,1)/3.28084*0.001,'--',mud2(:,2),mud2(:,1)/3.28084*0.001,'--',mud3(:,2),mud3(:,1)/3.28084*0.001,mud4(:,2),mud4(:,1)/3.28084*0.001,'--',mud5(:,2),mud5(:,1)/3.28084*0.001,'--',mud6(:,2),mud6(:,1)/3.28084*0.001,'--')
%set(h,{'LineWidth'},{1;1;1;1;3;1;1;1})
%set(gca,'Ydir','reverse')
%ylim([0 10])
%xlabel('Mud Weight (ppg)')
%ylabel('Depth (km)')
%legend('average','try 1','try 2','try 3','try 4','try 5','try 6','try 7')
%export_fig('../Fig/trymudweight4.pdf')
%
%figure('units','normalized','outerposition',[0 0 0.2 1])
%h=plot(mud(:,2),mud(:,1)/3.28084*0.001,'m--',mud0(:,2),mud0(:,1)/3.28084*0.001,'--',mud1(:,2),mud1(:,1)/3.28084*0.001,'--',mud2(:,2),mud2(:,1)/3.28084*0.001,'--',mud3(:,2),mud3(:,1)/3.28084*0.001,'--',mud4(:,2),mud4(:,1)/3.28084*0.001,mud5(:,2),mud5(:,1)/3.28084*0.001,'--',mud6(:,2),mud6(:,1)/3.28084*0.001,'--')
%set(h,{'LineWidth'},{1;1;1;1;1;3;1;1})
%set(gca,'Ydir','reverse')
%ylim([0 10])
%xlabel('Mud Weight (ppg)')
%ylabel('Depth (km)')
%legend('average','try 1','try 2','try 3','try 4','try 5','try 6','try 7')
%export_fig('../Fig/trymudweight5.pdf')
%
%figure('units','normalized','outerposition',[0 0 0.2 1])
%h=plot(mud(:,2),mud(:,1)/3.28084*0.001,'m--',mud0(:,2),mud0(:,1)/3.28084*0.001,'--',mud1(:,2),mud1(:,1)/3.28084*0.001,'--',mud2(:,2),mud2(:,1)/3.28084*0.001,'--',mud3(:,2),mud3(:,1)/3.28084*0.001,'--',mud4(:,2),mud4(:,1)/3.28084*0.001,'--',mud5(:,2),mud5(:,1)/3.28084*0.001,mud6(:,2),mud6(:,1)/3.28084*0.001,'--')
%set(h,{'LineWidth'},{1;1;1;1;1;1;3;1})
%set(gca,'Ydir','reverse')
%ylim([0 10])
%xlabel('Mud Weight (ppg)')
%ylabel('Depth (km)')
%legend('average','try 1','try 2','try 3','try 4','try 5','try 6','try 7')
%export_fig('../Fig/trymudweight6.pdf')
%
%figure('units','normalized','outerposition',[0 0 0.2 1])
%h=plot(mud(:,2),mud(:,1)/3.28084*0.001,'m--',mud0(:,2),mud0(:,1)/3.28084*0.001,'--',mud1(:,2),mud1(:,1)/3.28084*0.001,'--',mud2(:,2),mud2(:,1)/3.28084*0.001,'--',mud3(:,2),mud3(:,1)/3.28084*0.001,'--',mud4(:,2),mud4(:,1)/3.28084*0.001,'--',mud5(:,2),mud5(:,1)/3.28084*0.001,'--',mud6(:,2),mud6(:,1)/3.28084*0.001)
%set(h,{'LineWidth'},{1;1;1;1;1;1;1;3})
%set(gca,'Ydir','reverse')
%ylim([0 10])
%xlabel('Mud Weight (ppg)')
%ylabel('Depth (km)')
%legend('average','try 1','try 2','try 3','try 4','try 5','try 6','try 7')
%export_fig('../Fig/trymudweight7.pdf')
%
