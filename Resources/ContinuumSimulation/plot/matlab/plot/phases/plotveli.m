% Plot the veli of a cell
% Inputs:
%   fr -- the frame
%   n -- the index of the cell to be plotted
function plotveli(fr, n)
    rescale = fr.parameters.nsubsteps*fr.parameters.ninfo;
    quiver(fr.com{n}(1) + 1, fr.com{n}(2) + 1,  ...
           fr.veli{n}(1)*rescale, fr.veli{n}(2)*rescale, ...
           'Color', 'b');