% Returns the average velocity field produced by the cells
% Inputs:
%   fr -- the current frame
%   l -- the size of the averaging window
function [vx vy] = getvelocityfield(fr, l)
    
    % get total velocity
    vx = zeros([fr.parameters.LY fr.parameters.LX]);
    vy = zeros([fr.parameters.LY fr.parameters.LX]);
    
    % conversion factor
    c = fr.parameters.alpha/fr.parameters.xi;
    for i=1:fr.parameters.nphases
        vx = vx + fr.phi{i}*(fr.velp{i}(1)+fr.velc{i}(1)+c*fr.pol{i}(1));
        vy = vy + fr.phi{i}*(fr.velp{i}(2)+fr.velc{i}(1)+c*fr.pol{i}(2));
    end
    
    % average
    stencil = ones([l l])/l^2;
    vx = conv2(vx, stencil, 'same');
    vy = conv2(vy, stencil, 'same');