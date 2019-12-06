function wellqc(wellname)
if strcmp(wellname,'SS160')
    minz=0;
    maxz=6000*3.28084;
    data=loadlas('177114095100_Orig+Edit+RckPhys.las');
    kb=94.5;
    wdepth=50;
    figure
    plot(0.3048e6./data.dt_ed7,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vp (m/s)')
    ylabel('Depth (m)')
    ylim([minz maxz])
    figure
    plot(0.3048e6./data.dtsm_ed1,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vs (m/s)')
    ylim([minz maxz])
    figure
    plot(data.rhogard,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Gardner density (g/cc)')
    ylim([minz maxz])
    figure
    plot(data.gr,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Gamma ray')
    ylim([minz maxz])
    figure
    plot(data.ild,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Deep induction resistivity')
    ylim([minz maxz])
    figure
    plot(data.dphi,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Neutron porosity')
    ylim([minz maxz])
    figure
    plot(data.phie,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Effective porosity')
    ylim([minz maxz])
    figure
    plot(data.poisratio,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Poison ratio')
    ylim([minz maxz])
    figure
    plot(data.vcl,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Volume clay')
    ylim([minz maxz])
end
if strcmp(wellname,'SS187')
    minz=0;
    maxz=6000*3.28084;
    data=loadlas('177114129700_Orig+Edit+RckPhys.las');
    kb=96.3;
    wdepth=60;
    figure
    plot(0.3048e6./data.dt_ed5,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vp (m/s)')
    ylabel('Depth (m)')
    ylim([minz maxz])
    figure
    plot(0.3048e6./data.dtsm_ed1,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vs (m/s)')
    ylim([minz maxz])
    figure
    plot(data.rhob,(data.depth-kb-wdepth),'b')
    hold on
    plot(data.rhoz,(data.depth-kb-wdepth),'g')
    hold off
    set(gca,'Ydir','reverse')
    xlabel('Bulk density (g/cc)')
    ylim([minz maxz])
    figure
    plot(data.rhogard,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Gardner density (g/cc)')
    ylim([minz maxz])
    figure
    hold on
    plot(data.gr,(data.depth-kb-wdepth),'b')
    plot(data.grr,(data.depth-kb-wdepth),'g')
    plot(data.grs,(data.depth-kb-wdepth),'k')
    hold off
    set(gca,'Ydir','reverse')
    xlabel('Gamma ray')
    ylim([minz maxz])
    figure
    plot(data.ao90,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Deep induction resistivity')
    ylim([minz maxz])
    figure
    plot(data.tnph,(data.depth-kb-wdepth),'b')
    hold on
    plot(data.data____,(data.depth-kb-wdepth),'g') % use this one not the other
    hold off
    set(gca,'Ydir','reverse')
    xlabel('Neutron porosity')
    ylim([minz maxz])
    figure
    plot(data.phie,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Effective porosity')
    ylim([minz maxz])
    figure
    plot(data.poisratio,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Poison ratio')
    ylim([minz maxz])
    figure
    plot(data.vcl,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Volume clay')
    ylim([minz maxz])
end
if strcmp(wellname,'SS191')
    minz=0;
    maxz=6000*3.28084;
    data=loadlas('177114136300_Orig+Edit+RckPhys.las');
    kb=94;
    wdepth=72;
    figure
    plot(0.3048e6./data.dtln_ed2,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vp (m/s)')
    ylabel('Depth (m)')
    ylim([minz maxz])
    figure
    plot(0.3048e6./data.dtsm_ed1,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vs (m/s)')
    ylim([minz maxz])
    figure
    plot(data.rhogard,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Gardner density (g/cc)')
    ylim([minz maxz])
    figure
    plot(data.gr,(data.depth-kb-wdepth),'b')
    hold on
    plot(data.gram,(data.depth-kb-wdepth),'g')
    hold off
    set(gca,'Ydir','reverse')
    xlabel('Gamma ray')
    ylim([minz maxz])
    figure
    plot(data.at90,(data.depth-kb-wdepth),'b') % this on is fine
    hold on
    plot(data.raclm,(data.depth-kb-wdepth),'g')
    plot(data.rachm,(data.depth-kb-wdepth),'k')
    hold off
    set(gca,'Ydir','reverse')
    xlabel('Deep induction resistivity')
    ylim([minz maxz])
    figure
    plot(data.npor,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Neutron porosity')
    ylim([minz maxz])
    figure
    plot(data.poisratio,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Poison ratio')
    ylim([minz maxz])
    figure
    plot(data.vcl,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Volume clay')
    ylim([minz maxz])
    figure
    plot(data.tcdm,(data.depth-kb-wdepth)) %filter out high freq
    set(gca,'Ydir','reverse')
    xlabel('Temp')
    ylim([minz maxz])
end
if strcmp(wellname,'ST200')
    minz=0;
    maxz=6000*3.28084;
    data=loadlas('177154042100_Orig+Edit+RckPhys.las');
    kb=93;
    wdepth=131;
    figure
    plot(0.3048e6./data.dt_ed4,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vp (m/s)')
    ylabel('Depth (m)')
    ylim([minz maxz])
    figure
    plot(0.3048e6./data.dtsm_ed1,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vs (m/s)')
    ylim([minz maxz])
    figure
    plot(data.rhogard,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Gardner density (g/cc)')
    ylim([minz maxz])
    figure
    plot(data.gr,(data.depth-kb-wdepth),'b') % use this one
    hold on 
    plot(data.grd,(data.depth-kb-wdepth),'g')
    hold off
    set(gca,'Ydir','reverse')
    xlabel('Gamma ray')
    ylim([minz maxz])
    figure 
    plot(data.ild,(data.depth-kb-wdepth))
    xlim([0 5])
    set(gca,'Ydir','reverse')
    xlabel('Deep induction resistivity')
    ylim([minz maxz])
    figure 
    plot(data.phi,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Neutron porosity')
    ylim([minz maxz])
    figure
    plot(data.poisratio,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Poison ratio')
    ylim([minz maxz])
    figure
    plot(data.vcl,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Volume clay')
    ylim([minz maxz])
end
if strcmp(wellname,'ST143')
    minz=0;
    maxz=6000*3.28084;
    data=loadlas('177154098500_Orig+Edit+RckPhys.las');
    kb=97;
    wdepth=75;
    figure
    plot(0.3048e6./data.dt_ed1,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vp (m/s)')
    ylabel('Depth (m)')
    ylim([minz maxz])
    figure
    plot(0.3048e6./data.dtsm_ed1,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vs (m/s)')
    ylim([minz maxz])
    figure
    plot(data.rhogard,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Gardner density (g/cc)')
    ylim([minz maxz])
    figure
    plot(data.gram,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Gamma ray')
    ylim([minz maxz])
    figure 
    plot(data.rachm,(data.depth-kb-wdepth))
    xlim([0 5])
    set(gca,'Ydir','reverse')
    xlabel('Deep induction resistivity')
    ylim([minz maxz])
    figure 
    plot(data.porz,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Neutron porosity')
    ylim([minz maxz])
    figure 
    plot(data.phie,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Effective porosity')
    ylim([minz maxz])
    figure
    plot(data.poisratio,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Poison ratio')
    ylim([minz maxz])
    figure
    plot(data.vcl,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Volume clay')
    ylim([minz maxz])
    figure
    plot(data.tcdm,(data.depth-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Temp')
    ylim([minz maxz])
end
if strcmp(wellname,'ST168')
    minz=0;
    maxz=6000*3.28084;
    data=loadlas('177154117301.las');
    kb=130;
    wdepth=70;
    figure
    plot(0.3048e6./data.dtco_us_f,(data.dept_f-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vp (m/s)')
    ylabel('Depth (m)')
    ylim([minz maxz])
    data.dtsm_us_f(data.dtsm_us_f<132)=nan;
    figure
    plot(0.3048e6./data.dtsm_us_f,(data.dept_f-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Vs (m/s)')
    ylabel('Depth (m)')
    ylim([minz maxz])
    figure
    plot(data.gr_arc_filt_gapi,(data.dept_f-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Gamma ray')
    ylabel('Depth (m)')
    ylim([minz maxz])
    figure
    plot(data.at90_ohmm,(data.dept_f-kb-wdepth))
    set(gca,'Ydir','reverse')
    xlabel('Resistivity')
    ylim([minz maxz])
end
end
