#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_3d_pure_absorption.py - 3D纯吸收案例验证（物理一致性修正版）

关键修正：
1. 射线追踪积分限和方向向量修正
2. 立体角积分权重归一化修正
3. 与PINN使用完全一致的积分定义
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import roots_legendre
import math

# ==========================================
# 0. 路径配置与导入
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.join(PROJECT_ROOT, 'Core')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CORE_PATH not in sys.path:
    sys.path.insert(0, CORE_PATH)

import ModelClassTorch2
from ModelClassTorch2 import Pinns
from EquationModels.RadTrans3D_Complex import RadTrans3D_Physics

# 设置顶刊风格字体
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = False
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10

# ==========================================
# 1. 物理参数
# ==========================================
KAPPA = 1.0
CENTER = np.array([0.5, 0.5, 0.5])

def source_term(x, y, z):
    """球形热源: max(0, 1.0 - 2.0*r)"""
    r = np.sqrt((x - CENTER[0])**2 + (y - CENTER[1])**2 + (z - CENTER[2])**2)
    return np.maximum(0.0, 1.0 - 2.0 * r)

def compute_exact_intensity_single(x, y, z, theta, phi, num_points=300):
    """
    计算单一方向 (theta, phi) 的精确强度 I(x,y,z,theta,phi)
    
    纯吸收 RTE 解：
    I(x,s) = ∫[0 to L] κ I_b(x - l s) exp(-κ l) dl
    
    其中 s = (sinθ cosφ, sinθ sinφ, cosθ) 是光线传播方向
    L 是从 x 沿 -s 方向到边界（光线来源处）的距离
    
    注意：对于流入边界（光线来源），I=0（因为无外部辐射进入）
    """
    pos = np.array([x, y, z], dtype=np.float64)
    
    # 方向向量（光线传播方向）
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    s_vec = np.array([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    ], dtype=np.float64)
    
    # 计算从 pos 沿 -s 方向到边界的距离 L
    # 射线方程：r(t) = pos - t * s, t >= 0
    # 找 t > 0 使得 r(t) 的某个坐标为 0 或 1
    
    t_candidates = []
    for i in range(3):
        if abs(s_vec[i]) > 1e-14:
            # pos[i] - t * s_vec[i] = 0  →  t = pos[i] / s_vec[i]
            # pos[i] - t * s_vec[i] = 1  →  t = (pos[i] - 1) / s_vec[i]
            
            t_to_0 = pos[i] / s_vec[i]      # 到达坐标0的距离
            t_to_1 = (pos[i] - 1.0) / s_vec[i]  # 到达坐标1的距离
            
            if t_to_0 > 1e-14:
                t_candidates.append(t_to_0)
            if t_to_1 > 1e-14:
                t_candidates.append(t_to_1)
    
    if len(t_candidates) == 0:
        return 0.0
    
    L = min(t_candidates)  # 最近边界的距离
    
    if L <= 1e-14:
        return 0.0
    
    # 使用 Gauss-Legendre 积分计算 ∫[0 to L] κ I_b exp(-κ l) dl
    xi, w = roots_legendre(num_points)
    
    # 将 [-1, 1] 映射到 [0, L]
    l_vals = 0.5 * L * (xi + 1.0)
    weights = 0.5 * L * w
    
    # 计算被积函数
    integrand = np.zeros(num_points)
    for i in range(num_points):
        # 当前积分点位置（沿 -s 方向回溯）
        curr_pos = pos - l_vals[i] * s_vec
        
        # 检查是否在有效区域内（应该在内）
        if np.any(curr_pos < -1e-10) or np.any(curr_pos > 1.0 + 1e-10):
            integrand[i] = 0.0
            continue
        
        # 源项值
        S_val = source_term(curr_pos[0], curr_pos[1], curr_pos[2])
        
        # Beer-Lambert 衰减
        attenuation = np.exp(-KAPPA * l_vals[i])
        
        integrand[i] = KAPPA * S_val * attenuation
    
    # Gauss-Legendre 求和
    intensity = np.sum(integrand * weights)
    
    return max(0.0, intensity)

# ==========================================
# 2. 精确 G(x) 计算（使用与PINN一致的立体角积分）
# ==========================================
class ExactGSolver:
    """
    精确解计算器
    
    G(x) = ∫∫ I(x, θ, φ) sin(θ) dθ dφ = ∫[-1,1] ∫[0,2π] I dμ dφ
    
    其中 μ = cos(θ)，与PINN的 compute_incident_radiation 定义一致
    """
    def __init__(self, n_theta=32, n_phi=64):
        """
        初始化求积点
        """
        self.n_theta = n_theta
        self.n_phi = n_phi
        
        # θ: 使用 μ = cos(θ) 的 Gauss-Legendre 求积
        # ∫[0,π] sin(θ) dθ = ∫[-1,1] dμ = 2
        mu, w_mu = roots_legendre(n_theta)  # μ ∈ [-1, 1]
        self.theta_q = np.arccos(-mu)  # θ ∈ [0, π]
        # w_mu 的和为 2，对应 ∫[-1,1] dμ
        self.w_theta = w_mu  # ✅ 正确：和为 2，不需要乘任何因子
        
        # φ: 均匀分布
        self.phi_q = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        self.phi_q += np.pi / n_phi  # 中点偏移
        self.w_phi = np.full(n_phi, 2.0 * np.pi / n_phi)  # 每个权重 = 2π/n_phi
        
        # 创建2D网格
        self.theta_grid, self.phi_grid = np.meshgrid(self.theta_q, self.phi_q, indexing='ij')
        
        # 方向向量
        sin_theta = np.sin(self.theta_grid)
        cos_theta = np.cos(self.theta_grid)
        cos_phi = np.cos(self.phi_grid)
        sin_phi = np.sin(self.phi_grid)
        
        self.dir_vec_x = sin_theta * cos_phi
        self.dir_vec_y = sin_theta * sin_phi
        self.dir_vec_z = cos_theta
        
        # 权重矩阵：w_θ[i] × w_φ[j]
        # G = Σ_i Σ_j I(θ_i, φ_j) × w_θ[i] × w_φ[j]
        theta_weights = self.w_theta.reshape(-1, 1)  # [n_theta, 1]
        phi_weights = self.w_phi.reshape(1, -1)      # [1, n_phi]
        self.weights = theta_weights * phi_weights   # [n_theta, n_phi]，广播乘法
        
        # 验证权重
        w_theta_sum = np.sum(self.w_theta)
        w_phi_sum = np.sum(self.w_phi)
        w_total_sum = np.sum(self.weights)
        
        print(f"[ExactGSolver] θ weights sum: {w_theta_sum:.6f} (should be 2.0)")
        print(f"[ExactGSolver] φ weights sum: {w_phi_sum:.6f} (should be 2π ≈ 6.283)")
        print(f"[ExactGSolver] Total weight sum: {w_total_sum:.6f} (should be 4π ≈ 12.566)")
    
    def compute_G(self, x, y, z):
        """计算 G(x)"""
        I_vals = np.zeros((self.n_theta, self.n_phi))
        
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                theta = self.theta_grid[i, j]
                phi = self.phi_grid[i, j]
                I_vals[i, j] = compute_exact_intensity_single(x, y, z, theta, phi, num_points=100)
        
        # 加权积分 G = ∫∫ I sin(θ) dθ dφ
        # 注意：w_θ 已经对应 sin(θ)dθ
        G = np.sum(I_vals * self.weights)
        
        return G
    
    def compute_G_field(self, X, Y, Z, verbose=True):
        """计算整个场的 G(x)"""
        G = np.zeros_like(X)
        total = X.size
        
        for idx in range(total):
            i = idx // X.shape[1] if len(X.shape) > 1 else idx
            j = idx % X.shape[1] if len(X.shape) > 1 else 0
            
            x_val = X.flat[idx]
            y_val = Y.flat[idx]
            z_val = Z.flat[idx]
            
            G.flat[idx] = self.compute_G(x_val, y_val, z_val)
            
            if verbose and total > 10 and idx % max(1, total//10) == 0:
                print(f"    Progress: {idx}/{total} ({100*idx/total:.0f}%)")
        
        return G

# ==========================================
# 3. PINN预测
# ==========================================
def load_pinn_G(model_path, x_tensor, y_tensor, z_tensor, engine):
    """使用RadTrans3D_Physics计算PINN预测的G(x)"""
    device = engine.dev
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    
    with torch.no_grad():
        G_tensor = engine.compute_incident_radiation(x_tensor, y_tensor, z_tensor, model)
    
    return G_tensor.cpu().numpy()

# ==========================================
# 4. 主程序
# ==========================================
def plot_G_comparison(model_path="Results_3D_CaseA/model.pkl"):
    """生成G(x)对比图"""
    
    print("="*70)
    print("3D Pure Absorption Validation: Physics-Consistent Version")
    print("="*70)
    
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine = RadTrans3D_Physics(
        kappa_val=1.0, sigma_s_val=0.0, g_val=0.0,
        n_theta=8, n_phi=16, dev=device
    )
    
    # 使用与PINN相同方向数的精确解求解器（公平对比）
    # 或使用更高精度：n_theta=32, n_phi=64
    exact_solver = ExactGSolver(n_theta=16, n_phi=32)  # 512方向，平衡精度与速度
    
    output_dir = 'Figures_3D_Validation'
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 图1: 中心线对比
    # ---------------------------------------------------------
    print("\n[1] Centerline comparison (y=z=0.5)...")
    
    n_points = 30  # 减少点数以加速
    x_line = np.linspace(0.05, 0.95, n_points)  # 避开边界（边界上I=0）
    y_line = np.full_like(x_line, 0.5)
    z_line = np.full_like(x_line, 0.5)
    
    print("  Computing exact G(x)...")
    G_exact_line = exact_solver.compute_G_field(x_line, y_line, z_line, verbose=True)
    
    if os.path.exists(model_path):
        print("  Computing PINN G(x)...")
        x_line_t = torch.tensor(x_line, dtype=torch.float32, device=device)
        y_line_t = torch.tensor(y_line, dtype=torch.float32, device=device)
        z_line_t = torch.tensor(z_line, dtype=torch.float32, device=device)
        G_pinn_line = load_pinn_G(model_path, x_line_t, y_line_t, z_line_t, engine)
    else:
        print(f"  [Error] Model not found: {model_path}")
        return
    
    # 误差统计
    error_line = np.abs(G_exact_line - G_pinn_line)
    rel_error = error_line / (np.abs(G_exact_line) + 1e-10)
    
    print(f"\n  Exact G range: [{G_exact_line.min():.4f}, {G_exact_line.max():.4f}]")
    print(f"  PINN G range:  [{G_pinn_line.min():.4f}, {G_pinn_line.max():.4f}]")
    print(f"  Max abs error: {error_line.max():.4e}")
    print(f"  Mean abs error: {error_line.mean():.4e}")
    print(f"  Max rel error: {rel_error.max():.2%}")
    
    # ==========================================
    # 顶刊质量可视化
    # ==========================================
    
    # 颜色方案（Nature/Science风格）
    color_exact = '#000000'      # 黑色
    color_pinn = '#E41A1C'       # 红色
    color_error = '#377EB8'      # 蓝色
    color_fill = '#E41A1C'       # 误差填充色
    
    # 创建图形
    fig = plt.figure(figsize=(14, 5))
    
    # ---------- 子图1: G(x)对比 ----------
    ax1 = fig.add_subplot(121)
    
    # 精确解：黑色实线+圆点
    line1 = ax1.plot(x_line, G_exact_line, color=color_exact, linestyle='-', 
                     linewidth=2.2, label='Exact Analytical', zorder=3)
    ax1.scatter(x_line[::3], G_exact_line[::3], c=color_exact, s=50, 
                marker='o', zorder=4, edgecolors='white', linewidth=0.5)
    
    # PINN：红色虚线+方块
    line2 = ax1.plot(x_line, G_pinn_line, color=color_pinn, linestyle='--', 
                     linewidth=2.0, label='PINN Prediction', zorder=3)
    ax1.scatter(x_line[::3], G_pinn_line[::3], c=color_pinn, s=45, 
                marker='s', zorder=4, edgecolors='white', linewidth=0.5)
    
    # 添加阴影区域表示误差范围
    ax1.fill_between(x_line, G_exact_line, G_pinn_line, alpha=0.15, 
                     color=color_pinn, label='Discrepancy')
    
    ax1.set_xlabel(r'Spatial Position $x$ ($y=z=0.5$)', fontsize=13, fontweight='normal')
    ax1.set_ylabel(r'Incident Radiation $G(x)$', fontsize=13, fontweight='normal')
    ax1.set_title(r'(a) Centerline Distribution Comparison', fontsize=13, fontweight='bold', pad=10)
    
    # 图例
    ax1.legend(loc='upper right', frameon=True, fancybox=False, 
               edgecolor='black', fontsize=10, framealpha=0.95)
    
    # 网格：淡化
    ax1.grid(True, linestyle='--', alpha=0.25, color='gray', linewidth=0.5)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, max(G_exact_line.max(), G_pinn_line.max()) * 1.15])
    
    # 添加统计文本框
    textstr = f'$L_2$ Error: {np.sqrt(np.mean(error_line**2)):.2e}\n' + \
              f'$L_\infty$ Error: {error_line.max():.2e}\n' + \
              f'Max Rel. Error: {rel_error.max()*100:.2f}%'
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9, edgecolor='black')
    ax1.text(0.03, 0.97, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props, family='monospace')
    
    # ---------- 子图2: Absolute Error（顶刊风格） ----------
    ax2 = fig.add_subplot(122)
    
    # 误差曲线
    ax2.semilogy(x_line, error_line, color=color_error, linestyle='-', 
                 linewidth=2.0, label='Absolute Error', zorder=3)
    
    # 添加渐变填充
    ax2.fill_between(x_line, error_line, alpha=0.2, color=color_error)
    
    # 统计线
    mean_error = error_line.mean()
    max_error = error_line.max()
    ax2.axhline(y=mean_error, color='#FF7F00', linestyle='--', linewidth=1.5, 
                label=f'Mean: {mean_error:.2e}', zorder=2)
    ax2.axhline(y=max_error, color='#984EA3', linestyle=':', linewidth=1.5, 
                label=f'Max: {max_error:.2e}', zorder=2)
    
    # 添加误差棒风格的标注
    ax2.errorbar([0.5], [mean_error], yerr=[[0], [max_error-mean_error]], 
                 fmt='none', ecolor='gray', capsize=5, capthick=2, alpha=0.5)
    
    ax2.set_xlabel(r'Spatial Position $x$', fontsize=13, fontweight='normal')
    ax2.set_ylabel(r'Absolute Error $|G_{\mathrm{exact}} - G_{\mathrm{PINN}}|$', 
                   fontsize=13, fontweight='normal')
    ax2.set_title(r'(b) Error Distribution (Log Scale)', fontsize=13, fontweight='bold', pad=10)
    
    # 图例
    ax2.legend(loc='upper right', frameon=True, fancybox=False, 
               edgecolor='black', fontsize=9, framealpha=0.95)
    
    # 网格
    ax2.grid(True, linestyle='--', alpha=0.25, color='gray', linewidth=0.5, which='both')
    ax2.set_xlim([0, 1])
    
    # 添加误差统计文本框
    error_text = f'Error Statistics:\n' + \
                 f'  Mean: {mean_error:.2e}\n' + \
                 f'  Std:  {error_line.std():.2e}\n' + \
                 f'  Max:  {max_error:.2e}'
    props2 = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='black')
    ax2.text(0.03, 0.97, error_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props2, family='monospace')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, 'G_CaseA_Centerline_HighPrecision.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'G_CaseA_Centerline_HighPrecision.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {output_dir}/G_CaseA_Centerline_HighPrecision.png")
    plt.close()
    
    # ---------------------------------------------------------
    # 图2: 单点详细验证（用于调试）
    # ---------------------------------------------------------
    print("\n[2] Single-point detailed validation...")
    
    x_test, y_test, z_test = 0.5, 0.5, 0.5
    
    print(f"  At point ({x_test}, {y_test}, {z_test}):")
    print(f"  Source I_b = {source_term(x_test, y_test, z_test):.4f}")
    
    # 计算几个方向的I值
    test_dirs = [
        (0, 0),           # +z
        (np.pi/2, 0),     # +x
        (np.pi/2, np.pi), # -x
    ]
    
    for theta_t, phi_t in test_dirs:
        I_val = compute_exact_intensity_single(x_test, y_test, z_test, theta_t, phi_t, num_points=200)
        print(f"  θ={theta_t:.2f}, φ={phi_t:.2f}: I = {I_val:.4f}")
    
    G_test = exact_solver.compute_G(x_test, y_test, z_test)
    print(f"  Integrated G = {G_test:.4f}")
    
    # ==========================================
    # 图2: 2D 中心截面 (z=0.5) 对比
    # ==========================================
    print("\n[2] Generating 2D center slice (z=0.5)...")
    
    n_grid = 40
    x_grid = np.linspace(0.05, 0.95, n_grid)
    y_grid = np.linspace(0.05, 0.95, n_grid)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_grid = np.full_like(X_grid, 0.5)
    
    # 精确解
    print("  Computing exact 2D G field...")
    G_exact_2d = exact_solver.compute_G_field(X_grid, Y_grid, Z_grid, verbose=True)
    
    # PINN预测
    print("  Computing PINN 2D G field...")
    x_flat = X_grid.reshape(-1)
    y_flat = Y_grid.reshape(-1)
    z_flat = np.full_like(x_flat, 0.5)
    x_flat_t = torch.tensor(x_flat, dtype=torch.float32, device=device)
    y_flat_t = torch.tensor(y_flat, dtype=torch.float32, device=device)
    z_flat_t = torch.tensor(z_flat, dtype=torch.float32, device=device)
    G_pinn_flat = load_pinn_G(model_path, x_flat_t, y_flat_t, z_flat_t, engine)
    G_pinn_2d = G_pinn_flat.reshape(n_grid, n_grid)
    
    # 误差
    Error_2d = np.abs(G_exact_2d - G_pinn_2d)
    Rel_error_2d = Error_2d / (np.abs(G_exact_2d) + 1e-10)
    
    # 创建顶刊质量的1x3图
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # 统一色标
    vmin = min(G_exact_2d.min(), G_pinn_2d.min())
    vmax = max(G_exact_2d.max(), G_pinn_2d.max())
    levels = np.linspace(vmin, vmax, 20)
    
    # (a) 精确解
    cf1 = axes2[0].contourf(X_grid, Y_grid, G_exact_2d, levels=levels, cmap='viridis', extend='both')
    axes2[0].set_title('(a) Exact Solution', fontsize=12, fontweight='bold')
    axes2[0].set_xlabel('$x$', fontsize=11)
    axes2[0].set_ylabel('$y$', fontsize=11)
    axes2[0].set_aspect('equal')
    cbar1 = fig2.colorbar(cf1, ax=axes2[0], fraction=0.046, pad=0.04)
    cbar1.set_label('$G_{\mathrm{exact}}$', fontsize=10)
    
    # (b) PINN预测
    cf2 = axes2[1].contourf(X_grid, Y_grid, G_pinn_2d, levels=levels, cmap='viridis', extend='both')
    axes2[1].set_title('(b) PINN Prediction', fontsize=12, fontweight='bold')
    axes2[1].set_xlabel('$x$', fontsize=11)
    axes2[1].set_ylabel('$y$', fontsize=11)
    axes2[1].set_aspect('equal')
    cbar2 = fig2.colorbar(cf2, ax=axes2[1], fraction=0.046, pad=0.04)
    cbar2.set_label('$G_{\mathrm{PINN}}$', fontsize=10)
    
    # (c) 误差（使用不同色图）
    err_max = Error_2d.max()
    err_levels = np.linspace(0, err_max, 20)
    cf3 = axes2[2].contourf(X_grid, Y_grid, Error_2d, levels=err_levels, cmap='Reds', extend='max')
    axes2[2].set_title(f'(c) Absolute Error\n(Max: {err_max:.2e}, Mean: {Error_2d.mean():.2e})', 
                       fontsize=12, fontweight='bold')
    axes2[2].set_xlabel('$x$', fontsize=11)
    axes2[2].set_ylabel('$y$', fontsize=11)
    axes2[2].set_aspect('equal')
    cbar3 = fig2.colorbar(cf3, ax=axes2[2], fraction=0.046, pad=0.04)
    cbar3.set_label('$|G_{\mathrm{exact}} - G_{\mathrm{PINN}}|$', fontsize=10)
    
    plt.suptitle(f'Case 3D-A (Pure Absorption): Incident Radiation at $z=0.5$\n' + 
                 f'$\\kappa={KAPPA}$, 512 quadrature directions', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G_CaseA_2D_HighPrecision.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'G_CaseA_2D_HighPrecision.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_dir}/G_CaseA_2D_HighPrecision.png")
    plt.close()
    
    # 2D误差统计
    print(f"\n  2D Error Statistics:")
    print(f"  Max Error: {err_max:.4e}")
    print(f"  Mean Error: {Error_2d.mean():.4e}")
    print(f"  Max Rel Error: {Rel_error_2d.max():.2%}")
    
    print("\n" + "="*70)
    print("Validation Complete!")
    print(f"Output: {output_dir}/")
    print("="*70)

if __name__ == "__main__":
    plot_G_comparison("Results_3D_CaseA/model.pkl")
