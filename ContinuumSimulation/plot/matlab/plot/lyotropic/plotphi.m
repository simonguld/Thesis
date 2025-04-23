function answ = plotphi(phi)
    [C, qh] = contourf(phi, 88);
    set(qh,'EdgeColor', 'none');
    colormap(parula);
    shading flat;
    caxis([-0.05 0.05]);
    colorbar;