% Compute the cross correlation of two matrices using fft.
%
% Input:
%   A, B     well, the above mentioned two matrices
%   C        the resulting correlation matrix
%
function C = normxcorr2(A, B)
    % if only one arg compute autocorrelation
    switch nargin
        case 1
            B = A;
    end
    C = ifft2(fft2(A).*conj(fft2(B)))/(norm(A)*norm(B));