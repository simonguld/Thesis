% Example plotting script

% add libraries paths
addpath('./jsonlab/');
addpath('./archive/');
addpath('./plot/phases');

% load the archive
% ar = loadarchive('../../output.zip');
% For no compresssion:
ar = loadarchive('../../output/');

% set to 1 to produce only final plot
plotfinal = 0;
% set to one to produce a movie instead of plotting
makemovie = 0;

% set figure size for movie
set(gcf, 'Visible', 'Off');
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [0 0 1000 600]);
if(makemovie==1)
    vidObj           = VideoWriter('movie.avi');
    vidObj.Quality   = 100;
    vidObj.FrameRate = 8;
    close(vidObj); open(vidObj);
else
    clf;
    set(gcf, 'Visible', 'On');
end

% region of interest using scaling factor
f = 0.15;
roi = [ int64(f*ar.LY) int64((1-f)*ar.LY) ...
        int64(f*ar.LX) int64((1-f)*ar.LX) ];
% region of interest using fixed size
%s = 20;
%roi = [ int64((ar.LY-s)/2) int64((ar.LY+s)/2) ...
%        int64((ar.LX-s)/2) int64((ar.LX+s)/2) ];

% init running variables
time   = [];
meanv  = [];
meanvx = [];
meanvy = [];
kymoxy = [];
kymoyy = [];

first = 1;
last  = getnframes(ar);
for m=first:last
    
    % print frame number
    fprintf('processing frame %d/%d\n', m, getnframes(ar));
    
    % load and reshape frame
    fr = loadframe(ar, m);
    fr = reshapeframe(fr);
    
    % get the total velocity
    [vx vy] = getvelocityfield(fr, 2*ar.R);
    % restrict to roi
    vxr = vx(roi(1):roi(2)-1, roi(3):roi(4)-1);
    vyr = vy(roi(1):roi(2)-1, roi(3):roi(4)-1);
    vr  = sqrt(vxr.^2 + vyr.^2);
    
    % get stats of the vel in the roi
    time   = [time, ar.nstart + ar.ninfo*(m-1) ];
    meanv  = [meanv,  mean(vr(:))];
    meanvx = [meanvx, mean(vxr(:))];
    meanvy = [meanvy, mean(vyr(:))];
    kymoxy = [kymoxy; mean(vxr)];
    kymoyy = [kymoyy; mean(vyr)];
    
    % plotfinal gives a way to plot only a the end
    if(plotfinal==1 && m<last) continue; end;
    
    % otherwise just plot
    subplot(2,2,1);
    for i=1:ar.nphases
        plotphi(fr.phi{i});
        hold on;
    end
    % velocity field
    plotvelocity(vx, vy, 2);
    hold on;
    % roi
    rectangle('Position', [roi(3) roi(1) roi(4)-roi(3) roi(2)-roi(1)], 'EdgeColor', 'r');
    hold off;
    xlim([0 ar.LX]); ylim([0 ar.LY]); axis equal;
        
    subplot(2,2,2);
    plot(time, meanv, 'k', time, meanvx, 'b', time, meanvy, 'r');
    xlabel('time'); title('Mean velocity (ROI)');
    %legend('<|v|>_{ROI}', '< v_x>_{ROI}', '< v_y>_{ROI}');
    
    % we can not plot the kymographs on the first round
    if(m<2) continue; end
    
    subplot(2,2,3);
    plotkymograph(kymoxy);
    ylabel('time'); xlabel('x');
    title('Kymograph for v_x averaged over y (ROI)');
    
    subplot(2,2,4);
    plotkymograph(kymoyy);
    ylabel('time'); xlabel('x');
    title('Kymograph for v_y averaged over y (ROI)');
    
    % draw or add frame to movie
    if(makemovie==0)
        drawnow;
    else
        writeVideo(vidObj, getframe(gcf));
    end
end

if(makemovie==1)
    close(vidObj);
end