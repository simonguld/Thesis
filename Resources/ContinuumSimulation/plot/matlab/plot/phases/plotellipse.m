% Plot the ellipse obtained from structure tensor
% Inputs:
%   fr -- the frame
%   n -- the index of the cell to be plotted
function plotellipse(fr, n)
    l = fr.Q_order(n)
    R = sqrt(fr.area(n)/pi);
    s = 2*l/R/pi;
    w = fr.Q_angle(n) + pi/2;
    t = linspace(0, 2*pi);
    r = R*sqrt(1-s^2)./sqrt(1+s^2-2*s*cos(2*(t-w)));
    x = fr.com{n}(1) + 1 + r.*cos(t);
    y = fr.com{n}(2) + 1 + r.*sin(t);
    plot(x, y, 'g-');