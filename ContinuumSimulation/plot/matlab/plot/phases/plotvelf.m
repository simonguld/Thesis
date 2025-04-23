% Plot the velf of a cell
% Inputs:
%   fr -- the frame
%   n -- the index of the cell to be plotted
function plotvelf(fr, n)
    rescale = fr.parameters.nsubsteps*fr.parameters.ninfo;
    quiver(fr.com{n}(1) + 1, fr.com{n}(2) + 1,  ...
           fr.velc{n}(1)*rescale, fr.velc{n}(2)*rescale, ...
           'Color', 'r');