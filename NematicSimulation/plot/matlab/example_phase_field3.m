% Example plotting script

% add libraries paths
addpath('./jsonlab/');
addpath('./archive/');
addpath('./plot/phases');

% load the archive
% ar = loadarchive('../../output.zip');
% For no compresssion:
ar = loadarchive('~/NoBackup/output/');

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
meanvo = [];

for m=1:getnframes(ar)
    
    % print frame number
    fprintf('processing frame %d/%d\n', m, getnframes(ar));
    
    % load and reshape frame
    fr = loadframe(ar, m);
    fr = reshapeframe(fr);
    
    % get the total velocity
    [vx vy] = getvelocityfield(fr, 2*ar.R);
    % get vorticity
    dyux = gradient(vx')';
    dxuy = gradient(vy);
    omgz = dxuy-dyux;
    
    % get stats of the vel in the roi
    x = (0:ar.LX-1) - (ar.LX-1)/2;
    y = (0:ar.LY-1) - (ar.LY-1)/2;
    [X, Y] = meshgrid(x, y);
    prout = vy.*X-vx.*Y;
    
    time   = [time, ar.nstart + ar.ninfo*(m-1) ];
    meanv  = [meanv,  mean(vr(:))];
    meanvx = [meanvx, mean(vxr(:))];
    meanvy = [meanvy, mean(vyr(:))];
    meanvo = [meanvo, mean(abs(prout(:)))];
    
    % plotfinal gives a way to plot only a the end
    if(plotfinal==1 && m<getnframes(ar)) continue; end;
    
    % otherwise just plot
    subplot(2,1,1);
    for i=1:ar.nphases
        plotphi(fr.phi{i});
        hold on;
    end
    % velocity field
    plotvelocity(vx, vy, 2);
    hold off; xlim([0 ar.LX]); ylim([0 ar.LY]); axis equal;
        
    subplot(2,1,2);
    plot(time, meanvo, 'r');
    xlabel('time'); title('Mean velocity (ROI)');
    %legend('<|v|>_{ROI}', '< v_x>_{ROI}', '< v_y>_{ROI}');
    
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