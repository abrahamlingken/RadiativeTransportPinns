# MATLAB 结果导出与 Python 分析指南

## 方案一：在云端安装 MATLAB（推荐）

### 1. 检查是否已安装 MATLAB
```bash
which matlab
# 或
matlab -batch "disp('Hello')"
```

### 2. 如果没有安装，使用 Octave（MATLAB 开源替代品）
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y octave

# 运行
octave --eval "your_script.m"
```

### 3. 运行你的 MATLAB 代码
```bash
# 如果有 MATLAB
matlab -batch "your_mc_code"

# 如果用 Octave
octave your_mc_code.m
```

---

## 方案二：导出数据给我分析

### 方法 A：保存为 .mat 文件（推荐）

在 MATLAB 代码末尾添加：
```matlab
% 保存结果
save('MC_Results_CaseA.mat', 'G', 'x', 'y', 'z', 'kappa', 'sigma_s', 'g');

% 或者保存多个变量
results.G = G;
results.x = x;
results.y = y;
results.z = z;
results.kappa = kappa;
results.sigma_s = sigma_s;
results.g = g;
save('MC_Results.mat', '-struct', 'results');
```

然后下载并发送给我。

---

### 方法 B：保存为 NumPy .npz 文件（Python 可直接读取）

在 MATLAB 中：
```matlab
% 保存为文本格式，然后用 Python 转换
G_center = G(:, 26, 26);  % 中心线数据
dlmwrite('G_centerline.txt', G_center);
dlmwrite('G_center_slice.txt', G(:, :, 26));
```

在 Python 中转换：
```python
import numpy as np

G_center = np.loadtxt('G_centerline.txt')
G_slice = np.loadtxt('G_center_slice.txt')

# 保存为 npz
np.savez('MC_Results.mat', 
         G_center=G_center, 
         G_slice=G_slice,
         kappa=5.0, sigma_s=0.0, g=0.0)
```

---

### 方法 C：直接复制关键数值（最快）

在 MATLAB 命令行运行：
```matlab
% 获取关键数值
G_center = G(26, 26, 26);  % 中心点
G_max = max(G(:));
G_min = min(G(:));
x_center = x(26);

% 显示结果（可复制粘贴）
fprintf('G_center = %.6f\n', G_center);
fprintf('G_max = %.6f\n', G_max);
fprintf('G_min = %.6f\n', G_min);
fprintf('Center slice shape: %d x %d\n', size(G, 1), size(G, 2));

% 显示中心线数据
fprintf('\nCenterline G values:\n');
fprintf('x = '); fprintf('%.4f ', x); fprintf('\n');
fprintf('G = '); fprintf('%.4f ', G(:, 26, 26)); fprintf('\n');
```

---

## 方案三：使用 SCP 传输文件

### 从云端下载到本地
```bash
# 在本地终端运行
scp root@your-autodl-ip:~/code/RadiativeTransportPinns/MC_Results.mat ./
```

### 上传到文件共享服务
如果文件较大，可以上传到：
- Google Drive
- Dropbox
- 百度网盘
- 或者直接在聊天中发送（如果<100MB）

---

## 我需要的具体数据格式

### 必备信息：
```matlab
% 请提供这些值
Case = 'A';  % 或 'B' 或 'C'
G_center = G(26, 26, 26);  % 网格中心（假设51x51x51）
G_face_center = G(1, 26, 26);  % 面心
G_max = max(G(:));
G_min = min(G(:));

% 物理参数
kappa = 5.0;      % 或 0.5
sigma_s = 0.0;    % 或 4.5
g = 0.0;          % 或 0.0/0.8

% 可选：中心线数据用于绘图
G_centerline = G(:, 26, 26);  % 沿x轴中心线
x_coords = linspace(0, 1, 51);
```

### 完整数据文件结构：
```matlab
% 保存所有这些变量
save('MATLAB_Results_CaseA.mat', 
     'G',           % 3D数组 [nx, ny, nz]
     'x', 'y', 'z', % 1D坐标数组
     'kappa', 
     'sigma_s', 
     'g',
     'n_photons');  % 光子数
```

---

## 快速检查清单

运行 MATLAB 代码后，请确认：

- [ ] 代码成功完成，无错误
- [ ] 生成了 `G` 数组（3D）
- [ ] 记录了物理参数（κ, σs, g）
- [ ] 获取了 G_center 数值
- [ ] 可选：保存了中心线数据用于对比

---

## 发送给我

### 方式 1：直接粘贴数值（最简单）
在 MATLAB 中运行：
```matlab
fprintf('=== Case A Results ===\n');
fprintf('G_center = %.6f\n', G(26,26,26));
fprintf('G_face = %.6f\n', G(1,26,26));
fprintf('G_max = %.6f\n', max(G(:)));
fprintf('kappa = %.1f, sigma_s = %.1f, g = %.1f\n', kappa, sigma_s, g);
```
然后把输出复制粘贴给我。

### 方式 2：上传文件
1. 在 MATLAB 中：`save('results.mat', 'G', 'x', 'y', 'z', 'kappa', 'sigma_s', 'g');`
2. 下载文件到本地
3. 直接拖拽上传到对话中

### 方式 3：使用 GitHub/GitLab
```bash
# 添加到 git
git add results.mat
git commit -m "Add MATLAB results for Case A"
git push
```

---

## Python 读取 MATLAB .mat 文件

如果你给我 .mat 文件，我可以用 Python 读取：

```python
import scipy.io
import numpy as np

# 读取 MATLAB 文件
data = scipy.io.loadmat('MATLAB_Results.mat')

# 提取变量
G = data['G']
x = data['x'].flatten()
y = data['y'].flatten()
z = data['z'].flatten()
kappa = float(data['kappa'])
sigma_s = float(data['sigma_s'])
g = float(data['g'])

# 分析
print(f"G_center = {G[25, 25, 25]:.4f}")  % MATLAB 是 1-indexed
print(f"Shape: {G.shape}")
```

---

## 常见问题

### Q: MATLAB 索引从 1 开始，Python 从 0 开始
**A:** 注意转换！MATLAB 的 `G(26,26,26)` 对应 Python 的 `G[25,25,25]`

### Q: 文件太大无法发送
**A:** 只发送中心线数据：
```matlab
% MATLAB: 只保存关键切片
save('results_small.mat', 'G_centerline', 'kappa', 'sigma_s', 'g');
```

### Q: 如何在云端查看 MATLAB 结果图
**A:** 保存为图片：
```matlab
% MATLAB
contourf(squeeze(G(:,:,26)));
colorbar;
saveas(gcf, 'G_slice.png');
```
然后下载图片查看。

---

**准备好后，直接把 MATLAB 输出或文件发给我！**
