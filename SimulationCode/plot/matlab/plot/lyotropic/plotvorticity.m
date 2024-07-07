function plotvorticity(ux, uy)
    dyux = gradient(ux')';
    dxuy = gradient(uy);
    omgz = dxuy-dyux;
    [C, qh] = contourf(omgz,88);
    set(qh,'EdgeColor','none');
    colormap(jet);
    m = max(abs(omgz(:)));
    caxis([-m m]);
    shading flat;
    colorbar;
