function [wnrp, wnrn] = calcnoofdefects(dx,dy,qc,tm,tmax)
%CALCNOOFDEFECTS Calculate the number of +/- 1/2 defects
%   Input parameters:
%       dx,dy       director components, matlab size [ny,nx]
%       qc          defect charge, matrix size [ny,nx]
%       tm          current time
%       tmax        do not calculate correlations if tm<tmax
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    if(tmax>tm) return; end;
    
    %Check if the defect charge qc is calculated
    if(isempty(qc))
        wn=calcs(dx,dy);
    else
        wn=qc;
    end
    
    wnr=round(2*wn);
    wnrp=sum(sum((wnr+abs(wnr))/2))/4;  %Positive 1/2 defect charge
    wnrn=sum(sum((abs(wnr)-wnr)/2))/4;  %Negative 1/2 defect charge
