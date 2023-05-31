function answ = plotphi(phi)
    [C, qh] = contourf(phi, 88);
    set(qh,'EdgeColor', 'none');
    colormap(parula);
    shading flat;
    colorbar;