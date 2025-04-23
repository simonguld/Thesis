% Returns the directory field associated with cells.
% Inputs:
%   fr -- the current frame
%   l -- the size of the averaging window
function [S dx dy] = getdirector(fr, l)
    
    % get total velocity
    Qxx = zeros([fr.parameters.LY fr.parameters.LX]);
    Qyx = zeros([fr.parameters.LY fr.parameters.LX]);
    
    for i=1:fr.parameters.nphases
        Qxx = Qxx + fr.phi{i}*fr.Q_order(i)*cos(2*fr.Q_angle(i)+pi);
        Qyx = Qyx + fr.phi{i}*fr.Q_order(i)*sin(2*fr.Q_angle(i)+pi);
    end
    
    % average
    stencil = ones([l l])/l^2;
    Qxx = conv2(Qxx, stencil, 'same');
    Qyx = conv2(Qyx, stencil, 'same');
    
    % get director and order
    S  = sqrt(Qxx.^2 + Qyx.^2);
    dx = sqrt((ones(size(S)) + Qxx./S)/2);
    dy = sign(Qyx).*sqrt((ones(size(S)) - Qxx./S)/2);