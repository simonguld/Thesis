% Example plotting script

% add libraries paths
addpath('./jsonlab/');
addpath('./archive/');
addpath('./plot/phases');

% load the archive
% ar = loadarchive('../../output.zip');
% For no compresssion:
ar = loadarchive('~/NoBackup/output');

% set to 1 to produce only final plot
plotfinal = 0;
% set to one to produce a movie instead of plotting
makemovie = 0;

% set figure size for movie
set(gcf, 'Visible', 'Off');
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [0 0 600 600]);
if(makemovie==1)
    vidObj           = VideoWriter('movie.avi');
    vidObj.Quality   = 100;
    vidObj.FrameRate = 8;
    close(vidObj); open(vidObj);
else
    clf;
    set(gcf, 'Visible', 'On');
end

for m=1:getnframes(ar)
    
    % print frame number
    fprintf('processing frame %d/%d\n', m, getnframes(ar));
    
    % load and reshape frame
    fr = loadframe(ar, m);
    fr = reshapeframe(fr);
    
    % get the total velocity
    %[vx vy] = getvelocity(fr, 2*ar.R);
    % get the director
    %[S dx dy] = getdirector(fr, 2*ar.R);
    
    % plot
    for i=1:ar.nphases
        plotphi(fr.phi{i});
        hold on;
        %plotvela(fr, i);
        %hold on;
        %plotvelf(fr, i);
        %hold on;
        %plotdirector(S, dx, dy, 3);
        %hold on;
        %imshow(fr.phi{i})
        %hold on;
        %plotcom(fr, i);
        %hold on;
    end
    % velocity field
    %plotvelocity(vx, vy, 2);
    hold off;
    xlim([0 ar.LX]); ylim([0 ar.LY]); axis equal;

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