function plotvelocity(ux, uy, nbin)
    [ ny, nx ] = size(ux);
    [ x, y ] = meshgrid(1:nbin:nx, 1:nbin:ny);
    ux4 = ux(1:nbin:ny, 1:nbin:nx);
    uy4 = uy(1:nbin:ny, 1:nbin:nx);
    vh  = quiver(x, y, ux4, uy4);
    set(vh, 'Color', 'k');
