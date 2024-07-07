% Plot the vela of a cell
% Inputs:
%   fr -- the frame
%   n -- the index of the cell to be plotted
function plotvela(fr, n)
    rescale = 1;%fr.parameters.alpha/fr.parameters.xi;%fr.parameters.nsubsteps*fr.parameters.ninfo;
    quiver(fr.com{n}(1) + 1, fr.com{n}(2) + 1,  ...
           fr.pol{n}(1)*rescale, fr.pol{n}(2)*rescale, ...
           'Color', 'g');