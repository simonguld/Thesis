% Convert 
%
% Input:
%   M     well, the above mentioned matrix
%   R     the resulting vector
%
function R = mat2polar(M, i0, j0)
    % default arguments for the starting nodes
    switch nargin
        case 1
            i0 = 1;
            j0 = 1;
        case 2
            j0 = 1;
    end
    % shift circularly the matrix to set correct zero point
    M = circshift(M, 1-i0, 1);
    M = circshift(M, 1-j0, 2);
    % get dimensions
    [sx, sy] = size(M);
    % the length of the correlation vector is set by the smallest dimension
    l = floor(min(sx/2, sy/2)) + 1;
    % we go through all the nodes and bin in the right entry of R
    R = zeros(1, l); Rcount = zeros(1, l);
    for i=1:sx
        for j=1:sy
            % the index to be binned (wrap for PBC)
            k = 0;
            if i<sx/2 k = k + (i-1)^2;
            else      k = k + (sx-i+1)^2; end
            if j<sy/2 k = k + (j-1)^2;
            else      k = k + (sy-j+1)^2; end
            k = floor(sqrt(k)) + 1;
            
            if k>l continue; end
            
            R(k)      = R(k) + M(i,j);
            Rcount(k) = Rcount(k) + 1;
        end
    end
    % normalize
    R = R./Rcount;
    R = R/R(1);