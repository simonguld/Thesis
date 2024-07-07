function plotdivu(ux, uy)
    dxux = gradient(ux);
    dyuy = gradient(uy')';
    p = dxux+dyuy;
    contourf(p);
    colormap(jet);
    m = max(abs(p(:)));
    caxis([-m m]);
    shading flat;
    colorbar;
