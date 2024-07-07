function [ c, comg, x, y ] = calcorrelationsfft2( ux,uy )
%CALCORRELATIONSFFT2 Calculales the matrix of correlations c, size(ux)
%using the fast fourier transform
%   Input:
%       ux, uy      Velocity field components, size(ux)=size(uy)
% 
%   Output:
%       c           Velocity correlation, size(ux)
%       comg        Vorticity correlation, size(ux)
%       x, y        Coordinate matrices for c and comg, size(ux)

    %Make grid
    nx = size(ux,1); ny = size(uy,1);
    xlin = [linspace(0,nx/2,nx/2) linspace(-nx/2,0,nx/2)];
    ylin = [linspace(0,ny/2,ny/2) linspace(-ny/2,0,ny/2)];
    [x,y] = meshgrid(xlin,ylin);

    %Calculate vorticity
    dyux=gradient(ux')';
    dxuy=gradient(uy);
    ov=dxuy-dyux;

    %Calculate speed correlation
    uxfft = fft2(ux);
    uyfft = fft2(uy);
    cux = ifft2(uxfft.*conj(uxfft));
    cuy = ifft2(uyfft.*conj(uyfft));
    c=cux+cuy;
    
    %Calculate vorticity correlation
    ovfft = fft2(ov);
    comg = ifft2(ovfft.*conj(ovfft));
    
end