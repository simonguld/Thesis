% Reshape a matrix into a cell array.
% This is used to reshape multidimensional arrays that are actually arrays
% of arrays. We need this function because Matlab does not support arrays
% of arrays and they get imported as a single mutlidimensional matrix,
% which can be counter intuitive.
% Inputs:
%   arr -- the array to be reshaped
%   siz -- the size of each reshaped array

function answ = reshape_to_cell_array(arr, siz)

    % the number of cells to be
    n = size(arr, 1);
    % reshape
    cells = cell([n 1]);
    for i=1:n
        cells{i} = reshape(arr(i,:), siz);
    end
    % return
    answ = cells;