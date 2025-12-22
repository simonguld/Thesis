function [C qh] = plotkymograph(kymo)
    [C qh] = contourf(kymo);
    set(qh,'EdgeColor','none');
    colormap(jet);
    m = max(abs(kymo(:)));
    caxis([-m m]);
    shading flat;
    colorbar;
