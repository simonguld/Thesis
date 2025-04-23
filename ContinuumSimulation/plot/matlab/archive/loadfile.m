% Extract and parse json file from archive.
% Inputs:
%   ar -- archive (loadarchive function must have been called before)
%   fname -- name of the json file to be extracted
function answ = loadfile(ar, fname)
    if ar.compress_full_ == 1
        % extract file from archive archive
        [status fcontent] = system(['unzip -pj ' ar.path_ ' ' strcat(fname, ar.ext_)]);
        if status ~= 0
            error(strcat('Error: zip returned non zero value while opening file:', fname, ar.ext_));
        end
        % return parsed json file
        answ = loadjson(fcontent);
    elseif ar.compress_ == 1
        % extract single file
        [status fcontent] = system(['unzip -pj ' strcat(ar.path_, filesep, fname, ar.ext_) ' ' strcat(fname, '.json')]);
        if status ~= 0
            error(strcat('Error: zip returned non zero value while opening file:', fname, ar.ext_));
        end
        % return parsed json file
        answ = loadjson(fcontent);
    else
        % directly parse file
        answ = loadjson(strcat(ar.path_, filesep, fname, ar.ext_));
    end
    
    