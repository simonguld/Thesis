%Example plotting script
clear
clc
% add libraries paths
addpath('./jsonlab/');
addpath('./archive/');
addpath('./plot/lyotropic');

% load the archive
ar = loadarchive('../../out/');
%%For no compresssion:
%ar = loadarchive('../../output/');
igap = 1;
Urms=[];
qc=[];
charge=[];


Clist_matrix=[]
for m=0:igap:getnframes(ar)
    m
    % load and reshape frame
    fr = loadframe(ar, m);
    fr = reshapeframe(fr);
    % get polarity and velocity (2d)
    %[ ux, uy ]    = getvelocity(fr);
    [S, dx, dy]     = getdirector(fr);
    phi = getphi(fr);
    %Urms=[Urms sum(sum(sqrt(ux.^2+uy.^2)))];
    %% Calculate velocity and vorticity correlation
%    [ c, comg, x, y ] = calcorrelationsfft2( ux,uy );
%     r = sqrt(x.^2+y.^2); rmax = round(ar.LX/2);
%     rlist = zeros(0,rmax+1); clist = zeros(1,rmax+1); comglist = zeros(1,rmax+1);
%     for k=0:rmax
%         ind = r>=k & r<(k+1);
%         rlist(k+1) = k;
%         clist(k+1) = mean(c(ind));
%         comglist(k+1) = mean(comg(ind));
%     end
%     clist = clist/clist(1);
%     %clist_matrix=[clist_matrix; clist];
%     comglist = comglist/comglist(1);

    % plot order, director, defects, velocity and vorticity
    clf; %sp1=1; sp2=2;
    %subplot(sp1,sp2,1); 
                        plotphi(phi - ones(size(phi)));hold on;
                        plotdirector(S, dx, dy, 1); hold on;
                        plotdefects(ar.LX, ar.LY, dx, dy); hold off;
                        axis equal tight; title('conc. and director')
    %subplot(sp1,sp2,2); plotvelmagnitude(ux,uy);
    %                    axis equal tight; title('velmagnitude')
%     subplot(sp1,sp2,2); %plotvelocity(px, py, ar.LX/25); hold on;
%                         plotdirector(S, px, py, 2); hold on;
%                         %plotcharge(px, py); hold off;
%                         axis equal tight; title('polarity director')
%     subplot(sp1,sp2,3); plotvelocity(ux,uy,ar.LX/20);
%                         axis equal tight; title('velocity')
%     %subplot(sp1,sp2,5); plotvorticity(ux,uy);
%     %                    axis equal tight; title('vorticity')
%     subplot(sp1,sp2,4); plot(rlist,clist,rlist,comglist)
%                         xlabel('r'); ylabel('C(r)'); title('correlation')
%                         legend('velocity','vorticity')

    drawnow;
    pause(2.0)

end
charge = qc;
nanmean(charge)
