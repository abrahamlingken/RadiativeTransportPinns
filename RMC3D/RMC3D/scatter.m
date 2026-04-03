function [ux,uy,uz]=scatter(ux,uy,uz,scatter_type)
% 散射子程序-线性散射相函数
n=length(ux);
if scatter_type==1 % 各向同性散射
    a=rand(n,1);
    theta_rand=acosd(1-2*a);
    b=rand(n,1);
    phi_rand=b*(2*180);
end
if scatter_type==2 % 向前各向异性散射
    pb=1;
    a=rand(n,1);
    theta_rand=acosd((sqrt(1-pb*(4*a-2-pb))-1)/pb);
    b=rand(n,1);
    phi_rand=b*(2*180);
end
if scatter_type==3 % 向后各向异性散射
    pb=-1;
    a=rand(n,1);
    theta_rand=acosd((sqrt(1-pb*(4*a-2-pb))-1)/pb);
    b=rand(n,1);
    phi_rand=b*(2*180);
end
% Radiative Heat Transfer, 3rd-edition Chapter 21. The Monte Carlo Method for Participating Media
ex1=(uz-uy)./sqrt((uz-uy).^2+(ux-uz).^2+(uy-ux).^2);
ey1=(ux-uz)./sqrt((uz-uy).^2+(ux-uz).^2+(uy-ux).^2);
ez1=(uy-ux)./sqrt((uz-uy).^2+(ux-uz).^2+(uy-ux).^2);
ex2=uy.*ez1-uz.*ey1;
ey2=uz.*ex1-ux.*ez1;
ez2=ux.*ey1-uy.*ex1;
esx=sind(theta_rand).*(cosd(phi_rand).*ex1+sind(phi_rand).*ex2)+cosd(theta_rand).*ux;
esy=sind(theta_rand).*(cosd(phi_rand).*ey1+sind(phi_rand).*ey2)+cosd(theta_rand).*uy;
esz=sind(theta_rand).*(cosd(phi_rand).*ez1+sind(phi_rand).*ez2)+cosd(theta_rand).*uz;
ux=esx./sqrt(esx.^2+esy.^2+esz.^2);
uy=esy./sqrt(esx.^2+esy.^2+esz.^2);
uz=esz./sqrt(esx.^2+esy.^2+esz.^2);

end