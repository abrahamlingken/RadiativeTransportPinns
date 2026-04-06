% MATLAB 结果导出脚本
% 将此代码添加到你的 MC 代码末尾，用于导出结果到 Python

% ============================================
% 配置
% ============================================
case_name = 'A';  % 改为 'B' 或 'C'
kappa = 5.0;      % 根据案例调整
sigma_s = 0.0;    % 根据案例调整  
g = 0.0;          % 根据案例调整
n_photons = 5000000;  % 使用的光子数

% ============================================
% 提取关键数据（假设 G 是你的 3D 结果数组）
% ============================================

% 网格尺寸
[nx, ny, nz] = size(G);

% 坐标（假设均匀网格 [0,1]）
x = linspace(0.5/nx, 1-0.5/nx, nx);
y = linspace(0.5/ny, 1-0.5/ny, ny);
z = linspace(0.5/nz, 1-0.5/nz, nz);

% 关键位置的值
ix_center = ceil(nx/2);
iy_center = ceil(ny/2);
iz_center = ceil(nz/2);

G_center = G(ix_center, iy_center, iz_center);
G_face_center = G(1, iy_center, iz_center);
G_max = max(G(:));
G_min = min(G(:));

% 中心线数据（沿 x 轴）
G_centerline = G(:, iy_center, iz_center);

% ============================================
% 显示结果（可复制粘贴）
% ============================================
fprintf('\n');
fprintf('========================================\n');
fprintf('MATLAB MC Results - Case %s\n', case_name);
fprintf('========================================\n');
fprintf('Physics: kappa=%.1f, sigma_s=%.1f, g=%.1f\n', kappa, sigma_s, g);
fprintf('Grid: %d x %d x %d, Photons: %d\n', nx, ny, nz, n_photons);
fprintf('\n');
fprintf('Results:\n');
fprintf('  G_center = %.6f\n', G_center);
fprintf('  G_face_center = %.6f\n', G_face_center);
fprintf('  G_max = %.6f\n', G_max);
fprintf('  G_min = %.6f\n', G_min);
fprintf('\n');
fprintf('Centerline (x vs G):\n');
fprintf('x = ['); fprintf('%.4f ', x); fprintf(']\n');
fprintf('G = ['); fprintf('%.4f ', G_centerline); fprintf(']\n');
fprintf('========================================\n');

% ============================================
% 保存为 .mat 文件
% ============================================
output_filename = sprintf('MATLAB_Case%s_Results.mat', case_name);
save(output_filename, 'G', 'x', 'y', 'z', 'kappa', 'sigma_s', 'g', ...
     'n_photons', 'G_center', 'G_face_center', 'G_centerline');
fprintf('\nSaved to: %s\n', output_filename);

% ============================================
% 可选：保存为文本格式（方便查看）
% ============================================
txt_filename = sprintf('MATLAB_Case%s_Summary.txt', case_name);
fid = fopen(txt_filename, 'w');
fprintf(fid, 'Case %s Results\n', case_name);
fprintf(fid, 'G_center = %.6f\n', G_center);
fprintf(fid, 'G_face_center = %.6f\n', G_face_center);
fprintf(fid, 'G_max = %.6f\n', G_max);
fprintf(fid, 'G_min = %.6f\n', G_min);
fclose(fid);
fprintf('Summary saved to: %s\n', txt_filename);

% ============================================
% 可选：绘制并保存图片
% ============================================
figure('Visible', 'off');

% 子图 1: 中心切片
subplot(1, 2, 1);
contourf(squeeze(G(:, :, iz_center)), 20, 'LineColor', 'none');
colorbar;
title(sprintf('Case %s: G(x,y,0.5)', case_name));
xlabel('y'); ylabel('x');
axis equal tight;

% 子图 2: 中心线
subplot(1, 2, 2);
plot(x, G_centerline, 'b-', 'LineWidth', 2);
xlabel('x');
ylabel('G(x, 0.5, 0.5)');
title('Centerline');
grid on;

% 保存
img_filename = sprintf('MATLAB_Case%s_Plot.png', case_name);
saveas(gcf, img_filename);
close(gcf);
fprintf('Plot saved to: %s\n', img_filename);
