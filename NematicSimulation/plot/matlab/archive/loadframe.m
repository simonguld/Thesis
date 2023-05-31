% Load specific frame from archive.
% Inputs:
%   ar -- archive (loadarchive function must have been called before)
%   n -- the frame number (from 0 to ar._nframes)
function answ = loadframe(ar, n)
    % get frame name
    name = strcat('frame', num2str(ar.nstart+n*ar.ninfo));
    % read
    frame = loadfile(ar, name);
    fields = fieldnames(frame.data);
    % convert to correct type and add to archive object
    for i =1:numel(fields)
        fr(1).(fields{i}) = get_value(frame.data.(fields{i}));
    end
    % forward parameters
    fr(1).parameters = ar;

    answ = fr;