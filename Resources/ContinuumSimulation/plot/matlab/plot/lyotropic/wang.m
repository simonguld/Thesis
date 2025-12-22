function answ=wang(ax1, ax2)
%WANG Calculate angle between axes ax1 and ax2
    ang = atan2(abs(det([ax1;ax2])), dot(ax1,ax2));
    if(ang>pi/2)
        ax2=-ax2;
    end;
    m = det([ax1;ax2]);
    ang = sign(m)*atan2(abs(m),dot(ax1,ax2));
    answ = ang;