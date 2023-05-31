function plotdefects(nx, ny, dx, dy)
%PLOTDEFECTS Calculate defect charges and plot
    qc=calcs(dx, dy);           %Calculate defect charge
    chargearray(nx ,ny, qc);    %Plot
    
function chargearray(nx, ny, qc)
%CHARGEARRAY Identify +/- 1/2 defects from defect charges and plot
    yypq = [];
    xxpq = [];
    yynq = [];
    xxnq = [];
    %Loop over y
    for i=2:ny-1; 
        %Loop over x
        for j=2:nx-1; 
            yy=0; xx=0; n1=0;
            if(abs(qc(i,j))>0.4)
                ql = sign(qc(i,j)); yy = i; xx = j; n1 = 1; qc(i,j) = 0;
                for ii=-1:1
                    for jj=-1:1
                        if(ql*qc(i+ii,j+jj)>0.4)
                            yy = yy+i+ii; xx = xx+j+jj; n1 = n1+1;
                            qc(i+ii,j+jj) = 0;
                        end
                    end
                end
                if(ql>0)
                    yypq = [yypq; yy/n1]; xxpq = [xxpq; xx/n1];
                else
                    yynq = [yynq; yy/n1]; xxnq = [xxnq; xx/n1];
                end
            end
        end
    end
    plot(xxpq, yypq, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    plot(xxnq, yynq, 'g^', 'MarkerSize', 10, 'MarkerFaceColor', 'g');