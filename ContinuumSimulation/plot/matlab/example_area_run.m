% Example plotting script

% add libraries paths
addpath('./jsonlab/');
addpath('./archive/');
addpath('./plot/phases');

% load the archive
% ar = loadarchive('../../output.zip');
% For no compresssion:
ar = loadarchive('~/NoBackup/box_size75/');

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

time = [];
clearvars area;
area = cell(ar.nphases);

for m=1:getnframes(ar)
    
    % print frame number
    fprintf('processing frame %d/%d\n', m, getnframes(ar));
    
    % load and reshape frame
    fr = loadframe(ar, m);
    fr = reshapeframe(fr);
    
    % update
    time = [time, ar.nstart + ar.ninfo*(m-1) ];
    for i=1:ar.nphases
        area{i} = [area{i}, fr.area(i) ];
    end
    
    % plotfinal gives a way to plot only a the end
    if(plotfinal==1 && m<getnframes(ar)) continue; end;
    
    % plot
    subplot(2,1,1);
    for i=1:ar.nphases
        plotphi(fr.phi{i});
        hold on;
    end
    hold off;
    axis([0 ar.LX 0 ar.LY]); axis equal;
        
    subplot(2,1,2);
    for i=1:ar.nphases
        plot(time, area{i}/(pi*ar.R^2), 'b');
        hold on;
    end
    hold off;
    xlabel('time'); title('Area');
    
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