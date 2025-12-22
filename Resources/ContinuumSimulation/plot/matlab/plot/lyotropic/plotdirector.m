function plotdirector(S, dx, dy, nb)
    [x, y] = meshgrid(1:nb:size(S,1), 1:nb:size(S,2));
    dx = dx(1:nb:end,1:nb:end); dy = dy(1:nb:end,1:nb:end); S = S(1:nb:end,1:nb:end); 
    qh = quiver(x-dx/2, y-dy/2, S.*dx, S.*dy, 'ShowArrowHead', 'Off');
    set(qh, 'Color', 'k');