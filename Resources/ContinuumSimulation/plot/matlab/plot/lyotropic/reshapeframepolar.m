% Reshape all fields that need to be reshaped in a frame
% Inputs:
%   fr -- the frame to be reshaped
function answ = reshapeframepolar(fr)

    % size of domain
    siz = [fr.parameters.LY fr.parameters.LX];
    % reshape all fields
    %fr.ff = reshape(fr.ff, fr.parameters.LY, fr.parameters.LX, []);
    fr.Px = reshape(fr.Px, siz);
    fr.Py = reshape(fr.Py, siz);
    fr.phi= reshape(fr.phi, siz);
    
    answ = fr;