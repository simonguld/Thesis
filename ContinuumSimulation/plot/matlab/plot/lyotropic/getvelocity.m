% Compute velocity for frame
function [ux,uy] = getvelocity(fr)
    d=getdensity(fr);
    ux = sum(fr.ff(:,:,[2 6 9])-fr.ff(:,:,[3 7 8]),3)./d;
    uy = sum(fr.ff(:,:,[4 6 8])-fr.ff(:,:,[5 7 9]),3)./d;