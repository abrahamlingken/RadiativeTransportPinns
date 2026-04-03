function [xyzb,distL] = IntersectBoundary(x0,y0,z0,xb,yb,zb,ux,uy,uz,dx,dy,dz)
% 多条射线从介质内部穿过最近的网格边界(计算射线与边界交点)
% x0,y0,z0: 射线起点
% xb,yb,zb: 计算域边界
% dx,dy,dz: 网格间距
% ux,uy,uz: 射线方向
% xyzb: 射线穿过临界网格边界坐标
% dist: 射线起点与临界网格边界坐标距离

np = length(x0);
bx = zeros(np,1);
by = zeros(np,1);
bz = zeros(np,1);
xi = (round(x0/dx*1e12))/1e12; % 取12位有效数字
yi = (round(y0/dy*1e12))/1e12;
zi = (round(z0/dz*1e12))/1e12;
ix = ceil(xi); 
iy = ceil(yi);
iz = ceil(zi);

for i = 0:length(xb)-1
    if sum(xi==i)>0
        ix(xi==i & ux>0) = ix(xi==i & ux>0)+1;
    end
    if sum(yi==i)>0
        iy(yi==i & uy>0) = iy(yi==i & uy>0)+1;
    end
    if sum(zi==i)>0
        iz(zi==i & uz>0) = iz(zi==i & uz>0)+1;
    end
end

bx(ux>0) = xb(ix(ux>0)+1);
bx(ux<0) = xb(ix(ux<0));
by(uy>0) = yb(iy(uy>0)+1);
by(uy<0) = yb(iy(uy<0));
bz(uz>0) = zb(iz(uz>0)+1);
bz(uz<0) = zb(iz(uz<0));

dlxyz = [(bx-x0)./ux (by-y0)./uy (bz-z0)./uz];
dl = min(abs(dlxyz),[],2);
xyzb = [x0+ux.*dl y0+uy.*dl z0+uz.*dl];
xyzb(abs(xyzb)<1e-10) = 0;
distL = sqrt(sum((xyzb-[x0,y0,z0]).^2,2));

end
