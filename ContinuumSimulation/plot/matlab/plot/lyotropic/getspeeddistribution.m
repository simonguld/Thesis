function [vlist, pdfv, vmean]=getspeeddistribution(v,nbins)
%GETSPEEDDISTRIBUTION Calculates the pdf of the input array v (speed)
%normalized with the mean speed value mean(v(:)).
% Inputs:
%    v          a matrix of speed values (or some quantity in general)
%    nbins      number of bins to use in the histogram
% 
% Outputs:
%    vlist      a vector with the centervalues of the binned array
%    pdfv       a vector with the pdf(v). Same size as vlist.
%    vmean      the scalar mean speed.

%Calculate speed and normalize
vmean = mean(v(:));
v=v/vmean;

%Convert histogram to continous line
[pdfv,edges]=histcounts(v(:),nbins,'Normalization','pdf');
vlist=edges+abs(edges(2)-edges(1))/2; vlist(end)=[];



