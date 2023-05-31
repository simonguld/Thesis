% Plot kymograph (density plot with time as the t-axis).
% Inputs:
%   dat -- the array to be plotted
function plotkymograph(dat)

    % plot
    [C qh]=contourf(dat);
    set(qh,'EdgeColor','none');
    colormap(jet);
    
    % set colorbar
    m = max(abs(dat(:)));
    caxis([-m m]);
    colorbar;
