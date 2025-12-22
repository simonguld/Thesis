function plotorder(S)
    [C, qh] = contourf(S, 88);
    set(qh,'EdgeColor','none');
    colormap(summer);
    shading flat;
    colorbar;
