% Plot the contour of a phase-field
% Inputs:
%   phi -- the phase-field to be plotted
%   color -- color of the contour
function plotphi(phi, color)
    % default arguments (i hate matlab)
    switch nargin
        case 1
            color = 'b';
    end
    % the actual plot
    [C, qh] = contour(phi, [.5 .5]);
    set(qh,'EdgeColor', color);