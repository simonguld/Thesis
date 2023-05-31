%Example plotting script

% add libraries paths
addpath('./jsonlab/');
addpath('./archive/');
addpath('./tools/');
addpath('./plot/lyotropic');

% load the archive
ar = loadarchive('~/NoBackup/test/');
%%For no compresssion:
%ar = loadarchive('../../output/');

for m=1:getnframes(ar)
    % load and reshape frame
    fr = loadframe(ar, m);
    fr = reshapeframe(fr);
    
    % get order, director and velocity (2d)
    [ S, qx, qy ] = getdirector(fr);
    [ ux, uy ]    = getvelocity(fr);

    %Calculate velocity and vorticity correlation
    [ c, comg, x, y ] = calcorrelationsfft2( ux,uy );
    c = normxcorr2(ux)+normxcorr2(uy);
    r = sqrt(x.^2+y.^2);
    r(14, 3) - sqrt((14-1)^2 + (3-1)^2);
    rmax = floor(ar.LX/2);
    rlist = zeros(0,rmax+1); clist = zeros(1,rmax+1); comglist = zeros(1,rmax+1);
    for k=0:rmax
        ind = r>=k & r<(k+1);
        rlist(k+1) = k;
        clist(k+1) = mean(c(ind));
        comglist(k+1) = mean(comg(ind));
    end
    clist = clist/clist(1);
    comglist = comglist/comglist(1);

    % plot order, director, defects, velocity and vorticity
    clf;
    plot(rlist,clist,rlist, mat2polar(normxcorr2(ux)+normxcorr2(uy)))
    xlabel('r'); ylabel('C(r)'); title('correlation')
    legend('good','new')

    drawnow;
end
