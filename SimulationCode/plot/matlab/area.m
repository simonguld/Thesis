% Example plotting script

% add libraries paths
addpath('./jsonlab/');
addpath('./archive/');
addpath('./plot/phases');

% load the archive
% ar = loadarchive('../../output.zip');
% For no compresssion:
ar = loadarchive('../../output/');

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
area = [];

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
        pplot(time, area{i}/(pi*ar.R^2));
        hold on;
    end
    xlabel('time'); title('Area');
    
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