function [avrms,aqc]=calcrmsvelandcharge(ux,uy,qc)
%CALCRMSVELANDCHARGE Calculate mean rms velocity and total defect charge

    %Total defect charge
    aqc=[]; 
    if(~isempty(qc)) 
        aqc=sum(sum(abs(qc)));
        %if(aqc<0.1); qc(1,1)=0.1; end;
    end;

    %Rms velocity
    avrms = mean(mean(sqrt(ux.*ux+uy.*uy)));
