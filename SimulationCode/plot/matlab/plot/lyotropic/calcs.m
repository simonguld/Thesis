function qc=calcs(dx, dy)
%CALCS Calculate defect charges in the frame
%   Input parameters:
%       dx, dy       director component matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    [m, n]=size(dx);
    qc=zeros(size(dx));
    
    %Loop over the x-direction and then the y-direction
    for j=2:n-1
        for i=2:m-1
            %Access directors of surrounding nodes of entry (i,j)
            ax1=[dx(i+1,j) dy(i+1,j)];
            ax2=[dx(i-1,j) dy(i-1,j)];
            ax3=[dx(i,j-1) dy(i,j-1)];
            ax4=[dx(i,j+1) dy(i,j+1)];
            ax5=[dx(i+1,j-1) dy(i+1,j-1)];
            ax6=[dx(i-1,j-1) dy(i-1,j-1)];
            ax7=[dx(i+1,j+1) dy(i+1,j+1)];
            ax8=[dx(i-1,j+1) dy(i-1,j+1)];

            %Sum the rotations of the directors in the counter clockwise direction
            dang=wang(ax1,ax5);
            dang=dang+wang(ax5,ax3);
            dang=dang+wang(ax3,ax6);
            dang=dang+wang(ax6,ax2);
            dang=dang+wang(ax2,ax8);
            dang=dang+wang(ax8,ax4);
            dang=dang+wang(ax4,ax7);
            dang=dang+wang(ax7,ax1);

            qc(i,j)=dang/2/pi;
        end
    end