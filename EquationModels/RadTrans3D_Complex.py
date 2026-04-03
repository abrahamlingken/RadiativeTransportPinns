"""
RadTrans3D_Complex.py - 3D稳态辐射传输方程物理引擎 (OOP优雅版)

使用面向对象设计，彻底消除全局状态和猴子补丁。
每个物理工况通过实例化不同的RadTrans3D_Physics对象来实现。
"""

import torch
import numpy as np
from scipy.special import roots_legendre
import math

pi = math.pi


class RadTrans3D_Physics:
    """
    3D辐射传输物理引擎类
    
    封装所有物理参数和计算方法，支持：
    - 吸收系数 kappa
    - 散射系数 sigma_s
    - HG不对称因子 g
    - 立体角求积配置
    
    Args:
        kappa_val: 吸收系数（常数或函数）
        sigma_s_val: 散射系数（常数或函数）
        g_val: HG不对称因子
        n_theta: 极角方向求积点数
        n_phi: 方位角方向求积点数
        dev: 计算设备（torch.device）
    
    NOTE: Source term decoupled from kappa for rigorous scattering benchmark
    Equation: s·∇I + (κ+σs)I = Ib + (σs/4π)∫ΦI dΩ'  [Source: Ib, NOT κ·Ib]
    This ensures sufficient energy in high-scattering cases (e.g., Case C).
    """
    
    def __init__(self, kappa_val=0.5, sigma_s_val=0.5, g_val=0.0, 
                 n_theta=8, n_phi=16, dev=None):
        """
        初始化物理引擎
        
        所有物理参数在此绑定，创建独立的计算环境
        """
        # 保存设备
        self.dev = dev if dev is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 保存物理参数（可以是常数或可调用的空间函数）
        self.kappa_val = kappa_val
        self.sigma_s_val = sigma_s_val
        self.g_val = g_val
        
        # 求积配置
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.n_dir = n_theta * n_phi  # 总方向数
        
        # 预计算求积点（在初始化时一次性计算，避免重复计算）
        self._setup_quadrature()
        
        print(f"[RadTrans3D_Physics] Initialized:")
        print(f"  Device: {self.dev}")
        print(f"  kappa = {kappa_val}, sigma_s = {sigma_s_val}, g = {g_val}")
        print(f"  Quadrature: n_theta={n_theta}, n_phi={n_phi}, total_dirs={self.n_dir}")
    
    def _setup_quadrature(self):
        """
        预计算3D立体角求积点和权重
        
        使用Gauss-Legendre求积(theta) + 均匀分布(phi)
        所有张量在初始化时创建并移至目标设备
        """
        # Gauss-Legendre求积点 (theta ∈ [0, π])
        xi, w_xi = roots_legendre(self.n_theta)
        theta_q = torch.tensor(np.arccos(-xi), dtype=torch.float32, device=self.dev)
        w_theta = torch.tensor(w_xi, dtype=torch.float32, device=self.dev)
        
        # 均匀分布 (phi ∈ [0, 2π])
        phi_q = torch.linspace(0, 2*pi, self.n_phi+1, device=self.dev)[:-1]
        phi_q = phi_q + pi / self.n_phi
        w_phi = torch.ones(self.n_phi, device=self.dev) * (2 * pi / self.n_phi)
        
        # 创建2D网格
        theta_grid, phi_grid = torch.meshgrid(theta_q, phi_q, indexing='ij')
        
        # 保存网格（用于散射积分）
        self.theta_flat = theta_grid.reshape(-1)  # [n_dir]
        self.phi_flat = phi_grid.reshape(-1)      # [n_dir]
        
        # 计算方向向量
        sin_theta = torch.sin(theta_grid)
        cos_theta = torch.cos(theta_grid)
        cos_phi = torch.cos(phi_grid)
        sin_phi = torch.sin(phi_grid)
        
        # 方向向量 [n_dir, 3]
        self.dir_vectors = torch.stack([
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        ], dim=-1).reshape(-1, 3)
        
        # 立体角权重 dOmega = sin(theta) * dtheta * dphi
        # 关键修正：w_theta 是 Gauss-Legendre 在 μ=cos(θ) 上的权重
        # 由于 dμ = -sin(θ)dθ，w_theta 已经包含了 sin(θ)dθ 的贡献
        # 因此不需要再乘 sin_theta！
        theta_weights = w_theta.reshape(-1, 1)  # 对应 sin(θ)dθ
        phi_weights = w_phi.reshape(1, -1)      # 对应 dφ
        self.solid_angle_weights = (theta_weights * phi_weights).reshape(-1)  # 修正：移除 * sin_theta
    
    # ========================================================================
    # 物理参数函数（支持常数或空间变化）
    # ========================================================================
    
    def kappa(self, x, y, z):
        """吸收系数，可以是常数或空间函数"""
        if callable(self.kappa_val):
            return self.kappa_val(x, y, z)
        return torch.ones_like(x) * self.kappa_val
    
    def sigma_s(self, x, y, z):
        """散射系数，可以是常数或空间函数"""
        if callable(self.sigma_s_val):
            return self.sigma_s_val(x, y, z)
        return torch.ones_like(x) * self.sigma_s_val
    
    def I_b(self, x, y, z):
        """
        黑体辐射源项 - 高度局部化用于散射测试
        
        球形分布: max(0, 1 - 5*r)，仅在r<0.2区域有源
        创建厚冷区(0.2<r<0.5)用于显著测试散射效应
        """
        x0, y0, z0 = 0.5, 0.5, 0.5
        r = torch.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        return torch.clamp(1.0 - 5.0 * r, min=0.0)
    
    def kernel_HG_3d(self, s, s_prime):
        """
        3D Henyey-Greenstein相函数
        
        Phi_HG(s, s', g) = (1 - g^2) / (1 + g^2 - 2*g*cos(Theta))^{1.5}
        """
        cos_theta = torch.sum(s * s_prime, dim=-1)
        g_sq = self.g_val ** 2
        numerator = 1.0 - g_sq
        denominator = torch.pow(1.0 + g_sq - 2.0 * self.g_val * cos_theta, 1.5)
        return numerator / denominator
    
    # ========================================================================
    # 核心计算方法
    # ========================================================================
    
    def compute_scattering_3d(self, x, y, z, theta, phi, model):
        """
        计算3D散射积分（完全向量化，无for循环）
        
        内存管理策略：
        - 使用张量膨胀将空间点和方向点组合
        - 如果总点数过大，在内部分批处理网络前向传播
        - 但保持梯度追踪，因为这在closure内部调用
        
        Args:
            x, y, z: 空间坐标 [N]
            theta, phi: 当前方向 [N]
            model: PINN网络
            
        Returns:
            散射积分值 [N]
        """
        N = x.shape[0]
        
        # 当前方向向量 [N, 3]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        
        s_curr = torch.stack([
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        ], dim=-1)
        
        # 张量膨胀：构造 [N*n_dir, 5] 输入
        xyz = torch.stack([x, y, z], dim=-1)  # [N, 3]
        xyz_repeated = torch.repeat_interleave(xyz, self.n_dir, dim=0)  # [N*n_dir, 3]
        
        theta_expanded = self.theta_flat.repeat(N)  # [N*n_dir]
        phi_expanded = self.phi_flat.repeat(N)      # [N*n_dir]
        
        inputs = torch.cat([
            xyz_repeated,
            theta_expanded.unsqueeze(1),
            phi_expanded.unsqueeze(1)
        ], dim=-1)  # [N*n_dir, 5]
        
        # 网络预测（如果点数过大，内部分批）
        max_batch = 50000
        if inputs.shape[0] > max_batch:
            u_list = []
            for i in range(0, inputs.shape[0], max_batch):
                u_batch = model(inputs[i:i+max_batch]).reshape(-1)
                u_list.append(u_batch)
            u_pred = torch.cat(u_list, dim=0)
        else:
            u_pred = model(inputs).reshape(-1)  # [N*n_dir]
        
        # reshape为 [N, n_dir]
        u_matrix = u_pred.reshape(N, self.n_dir)
        
        # 相函数矩阵 [N, n_dir]
        s_expanded = s_curr.unsqueeze(1)           # [N, 1, 3]
        dir_expanded = self.dir_vectors.unsqueeze(0)  # [1, n_dir, 3]
        phi_matrix = self.kernel_HG_3d(s_expanded, dir_expanded)
        
        # 加权积分
        weights = self.solid_angle_weights.unsqueeze(0)  # [1, n_dir]
        integrand = phi_matrix * u_matrix * weights      # [N, n_dir]
        scatter = torch.sum(integrand, dim=1)            # [N]
        
        return scatter
    
    def compute_res(self, network, x_f_train, computing_error=False):
        """
        计算3D RTE残差
        
        PDE: s·∇u + (kappa + sigma_s)*u = (sigma_s/4π)*scatter + kappa*I_b
        
        Args:
            network: PINN网络
            x_f_train: 配点 [N, 5] (x, y, z, theta, phi)
            computing_error: 是否计算误差（兼容接口）
            
        Returns:
            残差 [N]
        """
        x_f_train = x_f_train.requires_grad_(True)
        
        # 解包
        x = x_f_train[:, 0]
        y = x_f_train[:, 1]
        z = x_f_train[:, 2]
        theta = x_f_train[:, 3]
        phi = x_f_train[:, 4]
        
        # 方向余弦
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        
        xi = sin_theta * cos_phi
        eta = sin_theta * sin_phi
        mu = cos_theta
        
        # 网络预测
        u = network(x_f_train).reshape(-1)
        
        # 空间梯度
        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(
            u, x_f_train,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        grad_x = grad_u[:, 0]
        grad_y = grad_u[:, 1]
        grad_z = grad_u[:, 2]
        
        # 方向导数
        s_dot_grad_u = xi * grad_x + eta * grad_y + mu * grad_z
        
        # 物理系数
        kappa_val = self.kappa(x, y, z)
        sigma_val = self.sigma_s(x, y, z)
        I_b_val = self.I_b(x, y, z)
        
        # 散射积分
        scatter_val = self.compute_scattering_3d(x, y, z, theta, phi, network)
        
        # 组装残差
        # NOTE: Source term decoupled from kappa for rigorous scattering benchmark
        # New equation: s·∇I + (κ+σs)I = Ib + (σs/4π)∫ΦI dΩ'
        lhs = s_dot_grad_u + (kappa_val + sigma_val) * u
        rhs = (sigma_val / (4.0 * pi)) * scatter_val + I_b_val  # Changed: kappa_val * I_b_val -> I_b_val
        residual = lhs - rhs
        
        return residual
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def compute_incident_radiation(self, x, y, z, model):
        """
        计算入射辐射 G(x) = ∫∫ u(x, s) dOmega
        
        用于后处理和可视化
        """
        N = x.shape[0]
        
        xyz = torch.stack([x, y, z], dim=-1)
        xyz_repeated = torch.repeat_interleave(xyz, self.n_dir, dim=0)
        
        theta_expanded = self.theta_flat.repeat(N)
        phi_expanded = self.phi_flat.repeat(N)
        
        inputs = torch.cat([
            xyz_repeated,
            theta_expanded.unsqueeze(1),
            phi_expanded.unsqueeze(1)
        ], dim=-1)
        
        # 分批预测
        max_batch = 50000
        if inputs.shape[0] > max_batch:
            u_list = []
            for i in range(0, inputs.shape[0], max_batch):
                u_batch = model(inputs[i:i+max_batch]).reshape(-1)
                u_list.append(u_batch)
            u_pred = torch.cat(u_list, dim=0)
        else:
            u_pred = model(inputs).reshape(-1)
        
        u_matrix = u_pred.reshape(N, self.n_dir)
        weights = self.solid_angle_weights.unsqueeze(0)
        G = torch.sum(u_matrix * weights, dim=1)
        
        return G
    
    def apply_bc(self, x_boundary, u_boundary, model):
        """
        应用边界条件
        
        只选择流入边界上的点 (n·s < 0)
        """
        x = x_boundary[:, 0]
        y = x_boundary[:, 1]
        z = x_boundary[:, 2]
        theta = x_boundary[:, 3]
        phi = x_boundary[:, 4]
        
        # 外法向
        n = torch.zeros_like(x_boundary[:, :3])
        n[:, 0] = torch.where(x < 0.01, -1.0, torch.where(x > 0.99, 1.0, 0.0))
        n[:, 1] = torch.where(y < 0.01, -1.0, torch.where(y > 0.99, 1.0, 0.0))
        n[:, 2] = torch.where(z < 0.01, -1.0, torch.where(z > 0.99, 1.0, 0.0))
        
        # 方向向量
        s = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=-1)
        
        # 流入边界判断
        n_dot_s = torch.sum(n * s, dim=1)
        inflow_mask = n_dot_s < 0
        
        if inflow_mask.sum() == 0:
            return torch.tensor([], device=self.dev), torch.tensor([], device=self.dev)
        
        x_inflow = x_boundary[inflow_mask]
        u_inflow = u_boundary[inflow_mask]
        
        u_pred = model(x_inflow).reshape(-1)
        
        return u_pred, u_inflow.reshape(-1)
    
    def generate_collocation_points(self, n_collocation, sampling_seed=0):
        """
        生成内部配点 - 静态物理优先采样 (Static Importance Sampling)
        
        采样策略：50/50分割，兼顾全局覆盖与热源区域加密
        - 问题背景：热源 S = max(0, 1-5r) 仅在 r < 0.2 内非零，体积占比仅约0.8%
        - 均匀采样会导致源区域内配点严重不足，PINN无法学习局部特征
        
        具体策略：
        1. 50% 均匀分布：空间坐标覆盖 [0,1]^3，确保全局物理一致性
        2. 50% 中心聚焦：空间坐标压缩至 [0.3,0.7]^3，密集采样热源区域
        3. 角度坐标：两组均采用完整角度域 [0,π]×[0,2π]，不压缩
        
        Args:
            n_collocation: 总配点数
            sampling_seed: Sobol序列随机种子
            
        Returns:
            inputs: [N, 5] 配点坐标 (x, y, z, theta, phi)
            u: [N, 1] 虚拟标签，填充NaN
        """
        from scipy.stats import qmc
        import numpy as np
        
        # 50/50分割配点数
        n_uniform = n_collocation // 2
        n_center = n_collocation - n_uniform  # 处理奇数情况
        
        # =========================================================================
        # 生成基础Sobol序列 (5D: x, y, z, theta, phi)
        # =========================================================================
        
        # 均匀分布组：全区域覆盖
        sampler_uniform = qmc.Sobol(d=5, scramble=False, seed=sampling_seed)
        points_uniform = sampler_uniform.random(n_uniform)
        
        # 中心聚焦组：热源区域加密
        sampler_center = qmc.Sobol(d=5, scramble=False, seed=sampling_seed + 1)
        points_center = sampler_center.random(n_center)
        
        # =========================================================================
        # 空间坐标映射 (x, y, z)
        # =========================================================================
        
        # 均匀组：映射至 [0, 1]^3
        xyz_uniform = points_uniform[:, :3]  # 已是 [0, 1]
        
        # 中心聚焦组：压缩至 [0.3, 0.7]^3，紧密包围中心 (0.5, 0.5, 0.5)
        xyz_center = 0.3 + 0.4 * points_center[:, :3]  # 映射公式: a + (b-a)*x, 其中 a=0.3, b=0.7
        
        # =========================================================================
        # 角度坐标映射 (theta, phi) - 两组均采用完整角度域
        # =========================================================================
        
        # theta ∈ [0, π] - 极角全覆盖
        theta_uniform = np.pi * points_uniform[:, 3]
        theta_center = np.pi * points_center[:, 3]
        
        # phi ∈ [0, 2π] - 方位角全覆盖
        phi_uniform = 2 * np.pi * points_uniform[:, 4]
        phi_center = 2 * np.pi * points_center[:, 4]
        
        # =========================================================================
        # 合并并转换为张量
        # =========================================================================
        
        # 堆叠坐标: (x, y, z, theta, phi)
        coords_uniform = np.stack([xyz_uniform[:, 0], xyz_uniform[:, 1], 
                                    xyz_uniform[:, 2], theta_uniform, phi_uniform], axis=1)
        coords_center = np.stack([xyz_center[:, 0], xyz_center[:, 1], 
                                   xyz_center[:, 2], theta_center, phi_center], axis=1)
        
        # 垂直拼接两组配点
        coords_combined = np.vstack([coords_uniform, coords_center])
        
        # 转为torch张量并放置到计算设备
        inputs = torch.tensor(coords_combined, dtype=torch.float32, device=self.dev)
        u = torch.full((n_collocation, 1), float('nan'), dtype=torch.float32, device=self.dev)
        
        return inputs, u
    
    def generate_boundary_points(self, n_boundary, sampling_seed=16):
        """
        生成边界点
        """
        from scipy.stats import qmc
        
        sampler = qmc.Sobol(d=5, scramble=False)
        if sampling_seed > 0:
            sampler.fast_forward(sampling_seed)
        
        points = sampler.random(n=n_boundary)
        points = torch.from_numpy(points).float().to(self.dev)
        
        x = points[:, 0].clone()
        y = points[:, 1].clone()
        z = points[:, 2].clone()
        theta = points[:, 3] * pi
        phi = points[:, 4] * 2 * pi
        
        # 分布在6个面上
        n_per_face = n_boundary // 6
        for i in range(6):
            mask = torch.arange(n_boundary, device=self.dev) // n_per_face == i
            if i == 0:      # x = 0
                x[mask] = 0.0
            elif i == 1:    # x = 1
                x[mask] = 1.0
            elif i == 2:    # y = 0
                y[mask] = 0.0
            elif i == 3:    # y = 1
                y[mask] = 1.0
            elif i == 4:    # z = 0
                z[mask] = 0.0
            elif i == 5:    # z = 1
                z[mask] = 1.0
        
        inputs = torch.stack([x, y, z, theta, phi], dim=-1)
        
        # 零Dirichlet边界
        u_boundary = torch.zeros(n_boundary, 1, device=self.dev)
        
        return inputs, u_boundary
