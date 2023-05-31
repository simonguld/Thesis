function plotvelmagnitude(ux, uy)
    mg=sqrt(ux.^2+uy.^2);
    [C, qh] = contourf(mg, 88);
    set(qh,'EdgeColor','none');
    colormap(parula);
    shading flat;
    colorbar;
