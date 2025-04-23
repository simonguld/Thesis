function [Y,corrm,oorrm,dorrm] = calcorrelations(nx,ny,ux,uy,dx,dy,tm,nb,tmax)
%CALCORRELATIONS Calculate spatial correlations for one frame
%   Input parameters:
%       nx,ny       size of the x and y dimension
%       ux,uy       velocity components, matlab size [ny,nx]
%       dx,dy       director components, matlab size [ny,nx]
%       nb          resolution to use of the input matrices
%       tm          current time 
%       tmax        do not calculate correlations if tm<tmax
% Output:
%       Y                       distance
%       corrm,oorrm,dorrm       normalized correlation of velocity,
%                               vorticity and the director respectively
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    if(tmax>tm) return; end;
    
    %Vorticity
    dyux=gradient(ux')';
    dxuy=gradient(uy);
    ov=dxuy-dyux;
    %Director combinations
    xx=dx.*dx;
    yy=dy.*dy;
    xy=dx.*dy;
    
    %Reshape if only a subset of the matrices are used
    if(nb>1)
        v=reshape(sum(sum(reshape(ux,[nb ny/nb nb nx/nb]),1),3),[ny/nb nx/nb])/(nb*nb);
        w=reshape(sum(sum(reshape(uy,[nb ny/nb nb nx/nb]),1),3),[ny/nb nx/nb])/(nb*nb);
        omg=reshape(sum(sum(reshape(ov,[nb ny/nb nb nx/nb]),1),3),[ny/nb nx/nb])/(nb*nb);
        dxx=reshape(sum(sum(reshape(xx,[nb ny/nb nb nx/nb]),1),3),[ny/nb nx/nb])/(nb*nb);
        dxy=reshape(sum(sum(reshape(xy,[nb ny/nb nb nx/nb]),1),3),[ny/nb nx/nb])/(nb*nb);
        dyy=reshape(sum(sum(reshape(yy,[nb ny/nb nb nx/nb]),1),3),[ny/nb nx/nb])/(nb*nb);
        p=sqrt(dxx).*sign(dxy); q = sqrt(dyy);
    else
        v=ux; w=uy; p=dx; q=dy; omg=ov;
    end
    
    %Initialize
    [mm, nn]=size(v);
    m=floor(mm/2); n=floor(nn/2);   %The grid is periodic, so we only look at half the domain size
    [x, y]=meshgrid(linspace(1,nn,nn),linspace(1,mm,mm));
    radn=(1:m)-1;                   %Vector of distances to consider
    drad=1;                         %Distance resolution
    corrm=zeros(size(radn)); oorrm=zeros(size(radn)); dorrm=zeros(size(radn));
    v0norm=0; omnorm=0;
    
    %Loop over x-direction
    for i=1:nn
        %Loop over y-direction
        for j=1:mm
            rdist=sqrt((y-j).^2+(x-i).^2);
            v0norm=v0norm+v(j,i)*v(j,i)+w(j,i)*w(j,i);
            omnorm=omnorm+omg(j,i)*omg(j,i);
            %Loop ove possible radial distances from entry (m,n)
            for k=1:m
                %Find linear indices of matrix entries a distance radn(k) to
                %radn(k)+1 from entry (m,n)
                ndist=find((rdist>=radn(k))&(rdist<radn(k)+drad));
                
                %vecta: value at entry (m,n)
                %vectb: values distance radn(k) from entry (m,n)
                if(~isempty(ndist))
                    %Calculate velocity correlation
                    vectb = [v(ndist) w(ndist)];
                    vecta = repmat([v(j,i) w(j,i)],size(ndist));
                    corrm(k)=corrm(k)+mean(sum(vecta.*vectb,2));
                    %Calculate vorticity correlation
                    vectb=omg(ndist);
                    vecta=omg(j,i);
                    oorrm(k)=oorrm(k)+mean(vecta*vectb);
                    %Calculate director correlation
                    vectb = [p(ndist) q(ndist)];
                    vecta = repmat([p(j,i) q(j,i)],size(ndist));
                    u1v1=vecta(:,1).*vectb(:,1);
                    u2v2=vecta(:,2).*vectb(:,2);
                    dotv=u1v1+u2v2;
                    u1v2=vecta(:,1).*vectb(:,2);
                    u2v1=vecta(:,2).*vectb(:,1);
                    cros=u1v2-u2v1;
                    ang=atan2(abs(cros),dotv);
                    cx=repmat(ang>(pi/2),1,2);
                    vectb=(1-2*cx).*vectb;
                    dorrm(k)=dorrm(k)+mean(sum(vecta.*vectb,2));
                end
            end
        end
    end
    %Normalize
    corrm=corrm/v0norm; oorrm=oorrm/omnorm; 
    dorrm=(dorrm/(mm*nn)-2/pi)/(1-2/pi); %Subtract <n(0).n(infty)> = 2/pi to remove tail for r -> infty
    Y=linspace(0,ny/2,size(radn,2));