% Load archive from file.
% Inputs:
%   path -- path of archive to load
function answ = loadarchive(path)
    % get file name and decide if we have a directory or a zip file
    ar(1).path_ = path;
    [ dirname, ar(1).name_, ext ] = fileparts(path);
    
    % 'default' values
    ar(1).compress_ = 0;
    ar(1).compress_full_ = 0;
    ar(1).ext_ = '.json';
    
    % detect full archive compression
    if strcmp(ext, '')
        % store name of directory as ar.name_
        [dirname, ar(1).name_] = fileparts([dirname '.']);
    elseif strcmp(ext, '.zip')
        ar(1).compress_full_ = 1;
    else
        error(strcat('Can not open file with extension ', ext));
    end
    
    % detect single file compression
    if ar.compress_full_ == 0 && ...
       exist(strcat(ar.path_, filesep, 'parameters.json.zip'), 'file') == 2
        ar(1).compress_ = 1;
        ar(1).ext_ = '.json.zip';
    end
    
    % read parameters
    parameters = loadfile(ar, 'parameters');
    fields = fieldnames(parameters.data);
    % convert to correct type and add to archive object
    for i=1:numel(fields)
        ar(1).(fields{i}) = get_value(parameters.data.(fields{i}));
    end

    % compute some private variables
    ar(1).nframes_ = (ar.nsteps-ar.nstart)/ar.ninfo;

    answ = ar;