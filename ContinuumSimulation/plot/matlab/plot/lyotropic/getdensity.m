% Compute density for frame
function [d] = getdensity(fr)
    d=sum(fr.ff,3);