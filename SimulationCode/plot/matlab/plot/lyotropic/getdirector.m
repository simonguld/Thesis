% Get director from frame (in two dimensions) and reshape. Note, dx>0
% always
function [S, dx, dy] = getdirector(fr)
    S  = sqrt(fr.QQxx.^2 + fr.QQyx.^2);              %Eigenvalues          
    dx = sqrt((ones(size(S)) + fr.QQxx./S)/2);       %Normalized eigenvector
    dy = sign(fr.QQyx).*sqrt((ones(size(S)) - fr.QQxx./S)/2);