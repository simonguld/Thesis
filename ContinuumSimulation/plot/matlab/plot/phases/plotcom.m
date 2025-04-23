% Plot the com of a cell
% Inputs:
%   fr -- the frame
%   n -- the index of the cell to be plotted
function plotcom(fr, n)
    plot(fr.com{n}(1) + 1, fr.com{n}(2) + 1, 'r.', 'MarkerSize', 10);