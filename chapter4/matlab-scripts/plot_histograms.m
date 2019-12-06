function plot_histograms(semlegacyfile,semmudfile)
	addpath('/data/cees/huyle/sep/sep-matlab-io');

	addpath('/data/cees/huyle/matlab/export-fig');
	
	set(0,'DefaultAxesFontSize',15)
	
	sem_legacy=sepread(semlegacyfile,201,1067);
	sem_mud=sepread(semmudfile,201,1067);
	
%	figure
%	histogram(sem_legacy(40:160,1:800),50)
%	xlim([0 1])
%	ylim([0 10000])
%	
%	figure
%	histogram(sem_mud(40:160,1:800),50)
%	xlim([0 1])
%	ylim([0 10000])
	
	figure('units','normalized','outerposition',[0 0 1 1])
	hold on
	histogram(sem_legacy(80:160,1:640),100,'FaceColor','m')
	histogram(sem_mud(80:160,1:640),100,'FaceColor','b','FaceAlpha',0.4)
	xlim([0.01 1])
	ylim([0 5000])
	hold off
	box on
	legend('Legacy velocity','Mud weight-converted velocity')
	legend boxoff
%	%export_fig('./Fig/histograms.pdf')
end
