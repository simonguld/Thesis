% Plot the vel = veli + vela + velf of a cell
% Inputs:
%   fr -- the frame
%   n -- the index of the cell to be plotted
function plotvel(fr, n)
    rescale = fr.parameters.nsubsteps*fr.parameters.ninfo;
    v1 = fr.veli{n}(1) + fr.vela{n}(1) + fr.velf{n}(1);
    v2 = fr.veli{n}(2) + fr.vela{n}(2) + fr.velf{n}(2);
    quiver(fr.com{n}(1) + 1, fr.com{n}(2) + 1,  ...
           v1*rescale, v2*rescale, 'Color', 'k');