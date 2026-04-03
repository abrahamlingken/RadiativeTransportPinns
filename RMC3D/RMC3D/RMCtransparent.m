% 基于RDF的反向蒙特卡洛法-三维直角坐标系
clear
clc
tic

sigma = 5.67e-8; % 黑体常数
scatterType = 1; % 散射相函数Φ = 1+Acos(Θ)：1-各向同性散射; 2-前向散射(A = 1); 3-后向散射(A = -1)
nray = 100000; % RMC射线数（建议大于100000）

% 几何模型
xlen = 1;
ylen = 1;
zlen = 1;

% 网格划分
nx = 51;
ny = 51;
nz = 51;
nxyz = nx*ny*nz;
deltx = xlen/nx; 
delty = ylen/ny;
deltz = zlen/nz;

% 单元网格边界
xbound = linspace(0,xlen,nx+1);
ybound = linspace(0,ylen,ny+1);
zbound = linspace(0,zlen,nz+1);

% 单元网格体积中心
xcenter = 0.5*deltx:deltx:xlen-0.5*deltx;
ycenter = 0.5*delty:delty:ylen-0.5*delty;
zcenter = 0.5*deltz:deltz:zlen-0.5*deltz;
[ygrid,xgrid,zgrid] = meshgrid(ycenter,xcenter,zcenter);
xyzcenter = [reshape(xgrid,nxyz,1) reshape(ygrid,nxyz,1) reshape(zgrid,nxyz,1)]; % 变化规则x-y-z

% 衰减系数-散射反照率-温度分布
ke = -10*((xyzcenter(:,1)-0.5).^2+(xyzcenter(:,2)-0.5).^2+(xyzcenter(:,3)-0.5).^2)+10; % 衰减系数分布
albedo = -0.8*((xyzcenter(:,1)-0.5).^2+(xyzcenter(:,2)-0.5).^2+(xyzcenter(:,3)-0.5).^2)+0.8; % 散射反照率
Temp = -800*((xyzcenter(:,1)-0.5).^2+(xyzcenter(:,2)-0.5).^2+(xyzcenter(:,3)-0.5).^2)+2000; % 温度分布

% 射线起点位置与方向
xyz0 = [0.5 0.5 0.5];
theta = 45; phi = 45;
uxyz = [sind(theta).*cosd(phi),sind(theta).*sind(phi),cosd(theta)]; % 追踪射线方向分量
% uxyz = [0 -1 0];

x0 = repmat(xyz0(1),nray,1); % 射线初始位置
y0 = repmat(xyz0(2),nray,1);
z0 = repmat(xyz0(3),nray,1);
ux = repmat(uxyz(1),nray,1); % 追踪射线方向分量
uy = repmat(uxyz(2),nray,1);
uz = repmat(uxyz(3),nray,1);
RDFs = zeros((nx+2)*(ny+2)*(nz+2),1); % 辐射分配因子
while sum(RDFs(:))<nray
    aRand = rand(length(x0),1);
    RandOptThi = -log(1-aRand); % 随机光学厚度
    
    % 散射分量
    uxsca = []; uysca = []; uzsca = [];
    xsca = []; ysca = []; zsca = [];

    % 追踪分量
    uxin = []; uyin = []; uzin = [];
    xin = []; yin = []; zin = [];
    
    while ~isempty(RandOptThi)
        
        [xyzInterBound,dis] = IntersectBoundary(x0,y0,z0,xbound,ybound,zbound,ux,uy,uz,deltx,delty,deltz);
        xyzmid = (xyzInterBound+[x0,y0,z0])/2;
        
        ix = ceil(xyzmid(:,1)/deltx);
        iy = ceil(xyzmid(:,2)/delty);
        iz = ceil(xyzmid(:,3)/deltz);
        node = (iz-1)*ny*nx+(iy-1)*nx+ix; % 射线中点位置编号（x-y-z）

        OptThi = ke(node).*dis; % 光学厚度      
        ijk = RandOptThi>OptThi; % 未走完随机光学厚度
        xyzUndo = xyzInterBound(ijk,:);
        ijkHitBound = xyzUndo(:,1) == xbound(1) | xyzUndo(:,1) == xbound(end) |...
            xyzUndo(:,2) == xbound(1) | xyzUndo(:,2) == xbound(end) |...
            xyzUndo(:,3) == xbound(1) | xyzUndo(:,3) == xbound(end); % 到达壁面
        if sum(ijkHitBound)>0 %到达壁面
            ix = ceil(xyzUndo(ijkHitBound,1)/deltx);
            iy = ceil(xyzUndo(ijkHitBound,2)/delty);
            iz = ceil(xyzUndo(ijkHitBound,3)/deltz);
            ix(ix~=nx) = ix(ix ~= nx)+1;
            ix(ix == nx) = ix(ix == nx)+2;
            iy(iy ~= ny) = iy(iy ~= ny)+1;
            iy(iy == ny) = iy(iy == ny)+2;
            iz(iz ~= nz) = iz(iz ~= nz)+1;
            iz(iz == nz) = iz(iz == nz)+2;
            nodeBound = (iz-1)*(ny+2)*(nx+2)+(iy-1)*(nx+2)+ix; % 吸收射线统计
            if length(nodeBound)>1
                RDFs = RDFs+histc(nodeBound,1:(nx+2)*(ny+2)*(nz+2)); 
            else
                RDFs = RDFs+histc(nodeBound,1:(nx+2)*(ny+2)*(nz+2))';
            end
        end
        
        % 判断吸收散射
        ijkDone = logical(1-ijk); % 走完随机光学厚度
        uxDone = ux(ijkDone);
        uyDone = uy(ijkDone);
        uzDone = uz(ijkDone);
        xDone = x0(ijkDone)+RandOptThi(ijkDone)./ke(node(ijkDone)).*uxDone; % 吸收散射位置坐标
        yDone = y0(ijkDone)+RandOptThi(ijkDone)./ke(node(ijkDone)).*uyDone;
        zDone = z0(ijkDone)+RandOptThi(ijkDone)./ke(node(ijkDone)).*uzDone;
        xDone(xDone<0) = 1e-10; xDone(xDone>xlen) = xlen-1e-10;
        yDone(yDone<0) = 1e-10; yDone(yDone>ylen) = ylen-1e-10;
        zDone(zDone<0) = 1e-10; zDone(zDone>zlen) = zlen-1e-10;
        
        cRand = rand(length(xDone),1);
        ijkabs = cRand>albedo(node(ijkDone)); % 吸收
        if sum(ijkabs)>0
            ix = ceil(xDone(ijkabs)/deltx);
            iy = ceil(yDone(ijkabs)/delty);
            iz = ceil(zDone(ijkabs)/deltz);
            nodeabsorb = iz*(ny+2)*(nx+2)+iy*(nx+2)+ix+1;
            if length(nodeabsorb)>1
                RDFs = RDFs+histc(nodeabsorb,1:(nx+2)*(ny+2)*(nz+2)); %吸收射线统计
            else
                RDFs = RDFs+histc(nodeabsorb,1:(nx+2)*(ny+2)*(nz+2))'; %吸收射线统计
            end
        end
        ijksca = logical(1-ijkabs); % 散射
        xsca = xDone(ijksca);
        ysca = yDone(ijksca);
        zsca = zDone(ijksca);
        [uxsca,uysca,uzsca] = scatter(uxDone(ijksca),uyDone(ijksca),uzDone(ijksca),scatterType);

        RandOptThi = RandOptThi(ijk);  % 未走完随机光学厚度
        OptThi = OptThi(ijk);
        ijkUndo = logical(1-ijkHitBound);
        RandOptThi = RandOptThi(ijkUndo)-OptThi(ijkUndo); % 未走完随机光学厚度且未到达边界
        
        uxDone = ux(ijk);
        uyDone = uy(ijk);
        uzDone = uz(ijk);
        ux = uxDone(ijkUndo);
        uy = uyDone(ijkUndo);
        uz = uzDone(ijkUndo);
        x0 = xyzUndo(ijkUndo,1); % 到达网格线坐标
        y0 = xyzUndo(ijkUndo,2);
        z0 = xyzUndo(ijkUndo,3);
        xin = [xin;xsca]; % 未吸收
        yin = [yin;ysca];
        zin = [zin;zsca];
        uxin = [uxin;uxsca];
        uyin = [uyin;uysca];
        uzin = [uzin;uzsca];
    end
    
    ux = uxin;
    uy = uyin;
    uz = uzin;
    x0 = xin; % 到达壁面坐标
    y0 = yin;
    z0 = zin;
    if sum(RDFs(:))+length(ux) ~= nray
        disp(sum(RDFs(:))+length(ux))
        break
    end
end

RDF = reshape(RDFs,nx+2,ny+2,nz+2);% 辐射分配因子
RDF = RDF(2:nx+1,2:ny+1,2:nz+1)/nray;
RDF = reshape(RDF,nx*ny*nz,1); 
RI = sigma*RDF'*Temp.^4/pi; % 辐射强度

toc

