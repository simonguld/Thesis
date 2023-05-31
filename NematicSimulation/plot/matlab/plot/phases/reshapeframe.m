% Reshape all fields that need to be reshaped in a frame
% Inputs:
%   fr -- the frame to be reshaped
function answ = reshapeframe(fr)

    % the size of the domain
    siz = [fr.parameters.LY fr.parameters.LX];
    
    % reshape
    fr.phi  = reshape_to_cell_array(fr.phi , siz);
    fr.pol  = reshape_to_cell_array(fr.pol, [1 2]);
    fr.velp = reshape_to_cell_array(fr.velp, [1 2]);
    fr.velc = reshape_to_cell_array(fr.velc, [1 2]);
    fr.velf = reshape_to_cell_array(fr.velf, [1 2]);
    fr.vel = reshape_to_cell_array(fr.vel, [1 2]);
    if isfield(fr, 'com')==1
        fr.com  = reshape_to_cell_array(fr.com , [1 2]);
    end

    answ = fr;