#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_3d_pure_absorption.py - 3D纯吸收案例验证（高精度数值积分版）

使用与PINN一致的Gauss-Legendre求积，确保公平对比
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

def compute_exact_intensity_single(x, y, z, s_vec, num_points=500):
    """
    计算单一方向的精确强度 I(x,y,z,s_vec)
    使用高精度反向射线追踪
    """
    pos = np.array([x, y, z])
    s_vec = np.array(s_vec)
    s_vec = s_vec / np.linalg.norm(s_vec)
    
    # 计算到边界的距离
    t_bounds = []
    for i in range(3):
        if abs(s_vec[i]) > 1e-10:
            t1 = -pos[i] / s_vec[i] if s_vec[i] > 0 else (1.0 - pos[i]) / s_vec[i]
            if t1 > 0:
                t_bounds.append(t1)
    
    L = min(t_bounds) if t_bounds else 0.0
    
    if L <= 0:
        return 0.0
    
    # 使用Gauss-Legendre积分替代梯形法则（更高精度）
    xi, w = roots_legendre(num_points)
    # 映射到 [0, L]
    l_vals = 0.5 * (xi + 1) * L
    weights = 0.5 * L * w
    
    integrand = np.zeros(num_points)
    for i in range(num_points):
        curr_pos = pos - l_vals[i] * s_vec
        S_val = source_term(curr_pos[0], curr_pos[1], curr_pos[2])
        integrand[i] = KAPPA * S_val * np.exp(-KAPPA * l_vals[i])
    
    intensity = np.sum(integrand * weights)
    return intensity

# ==========================================
# 2. 高精度 G(x) 计算（使用Gauss-Legendre求积）
# ==========================================
class ExactGSolver:
    """
    精确解计算器，使用与PINN一致的Gauss-Legendre求积
    """
    def __init__(self, n_theta=32, n_phi=64):
        """
        初始化求积点
        n_theta: 极角方向求积点数（默认32，高于PINN的8）
        n_phi: 方位角方向求积点数（默认64，高于PINN的16）
        """
        self.n_theta = n_theta
        self.n_phi = n_phi
        
        # Gauss-Legendre求积点（与PINN一致的方法）
        xi, w_xi = roots_legendre(n_theta)
        self.theta_q = np.arccos(-xi)  # 映射到 [0, π]
        self.w_theta = w_xi * np.pi  # 权重包含 π 因子
        
        # 均匀分布（与PINN一致）
        self.phi_q = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        self.phi_q += np.pi / n_phi  # 中点偏移
        self.w_phi = np.ones(n_phi) * (2 * np.pi / n_phi)
        
        # 创建2D网格
        self.theta_grid, self.phi_grid = np.meshgrid(self.theta_q, self.phi_q, indexing='ij')
        
        # 计算方向向量
        sin_theta = np.sin(self.theta_grid)
        cos_theta = np.cos(self.theta_grid)
        cos_phi = np.cos(self.phi_grid)
        sin_phi = np.sin(self.phi_grid)
        
        self.dir_vectors = np.stack([
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        ], axis=-1)  # [n_theta, n_phi, 3]
        
        # 立体角权重 dOmega = sin(theta) * dtheta * dphi
        theta_weights = self.w_theta.reshape(-1, 1)
        phi_weights = self.w_phi.reshape(1, -1)
        self.weights = theta_weights * phi_weights * sin_theta
        
        print(f"[ExactGSolver] Initialized: n_theta={n_theta}, n_phi={n_phi}, total_dirs={n_theta*n_phi}")
    
    def compute_G(self, x, y, z):
        """
        计算 G(x) = ∫∫ I(x,s) dΩ
        使用预计算的求积点进行数值积分
        """
        I_vals = np.zeros((self.n_theta, self.n_phi))
        
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                s_vec = self.dir_vectors[i, j]
                I_vals[i, j] = compute_exact_intensity_single(x, y, z, s_vec, num_points=200)
        
        # 加权积分
        G = np.sum(I_vals * self.weights)
        return G
    
    def compute_G_field(self, X, Y, Z, verbose=True):
        """
        计算整个场的 G(x)
        """
        G = np.zeros_like(X)
        total = X.size
        
        for idx in range(total):
            i = idx // X.shape[1] if len(X.shape) > 1 else idx
            j = idx % X.shape[1] if len(X.shape) > 1 else 0
            
            if len(X.shape) > 1:
                G[i, j] = self.compute_G(X[i,j], Y[i,j], Z[i,j])
            else:
                G[idx] = self.compute_G(X[idx], Y[idx], Z[idx])
            
            if verbose and idx % max(1, total//20) == 0:
                print(f"    Progress: {idx}/{total} ({100*idx/total:.1f}%)")
        
        return G

# ==========================================
# 3. PINN预测（使用物理引擎计算G）
# ==========================================
def load_pinn_G(model_path, x_tensor, y_tensor, z_tensor, engine):
    """
    使用RadTrans3D_Physics计算PINN预测的G(x)
    """
    device = engine.dev
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    
    with torch.no_grad():
        G_tensor = engine.compute_incident_radiation(x_tensor, y_tensor, z_tensor, model)
    
    return G_tensor.cpu().numpy()

# ==========================================
# 4. 主绘图程序
# ==========================================
def plot_G_comparison(model_path="Results_3D_CaseA/model.pkl"):
    """生成G(x)对比图"""
    
    print("="*70)
    print("3D Pure Absorption Validation: High-Precision G(x) Comparison")
    print("="*70)
    
    # 初始化物理引擎（纯吸收参数）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine = RadTrans3D_Physics(
        kappa_val=1.0,
        sigma_s_val=0.0,
        g_val=0.0,
        n_theta=8,
        n_phi=16,
        dev=device
    )
    
    # 初始化精确解求解器（更高精度）
    exact_solver = ExactGSolver(n_theta=32, n_phi=64)
    
    output_dir = 'Figures_3D_Validation'
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 图1: 沿中心线的 G(x) 对比
    # ---------------------------------------------------------
    print("\n[1] Generating centerline comparison...")
    
    n_points = 50  # 减少点数以加速计算
    x_line = np.linspace(0, 1, n_points)
    y_line = np.full_like(x_line, 0.5)
    z_line = np.full_like(x_line, 0.5)
    
    # 解析解 G（高精度）
    print("  Computing exact G(x) with high precision...")
    G_exact_line = exact_solver.compute_G_field(x_line, y_line, z_line, verbose=True)
    
    # PINN预测 G
    if os.path.exists(model_path):
        print("  Computing PINN G(x)...")
        x_line_t = torch.tensor(x_line, dtype=torch.float32, device=device)
        y_line_t = torch.tensor(y_line, dtype=torch.float32, device=device)
        z_line_t = torch.tensor(z_line, dtype=torch.float32, device=device)
        G_pinn_line = load_pinn_G(model_path, x_line_t, y_line_t, z_line_t, engine)
    else:
        print(f"  [错误] 找不到模型文件 {model_path}")
        return
    
    # 计算误差
    error_line = np.abs(G_exact_line - G_pinn_line)
    rel_error_line = error_line / (np.abs(G_exact_line) + 1e-10)
    
    print(f"  Centerline Max Error: {error_line.max():.6e}")
    print(f"  Centerline Mean Error: {error_line.mean():.6e}")
    print(f"  Centerline Max Rel Error: {rel_error_line.max():.2%}")
    
    # 绘图（统一风格）
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 解析解：黑色实线
    ax.plot(x_line, G_exact_line, 'k-', linewidth=2.5, 
            label='Exact Analytical (2048 dirs)', zorder=2)
    
    # PINN：红色虚线带圆圈
    ax.plot(x_line, G_pinn_line, color='#D62728', linestyle='--', 
            linewidth=2.0, marker='o', markersize=6, 
            markerfacecolor='none', markevery=5,
            label='PINN Prediction (128 dirs)', zorder=3)
    
    ax.set_xlabel(r'$x$ (optical depth)', fontsize=12)
    ax.set_ylabel(r'$G(x, 0.5, 0.5)$', fontsize=12)
    ax.set_title(r'3D Pure Absorption: $G(x)$ along Centerline ($y=z=0.5$)', fontsize=13)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G_CaseA_Centerline_HighPrecision.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'G_CaseA_Centerline_HighPrecision.pdf'))
    print(f"  Saved: {output_dir}/G_CaseA_Centerline_HighPrecision.png")
    plt.close()
    
    # ---------------------------------------------------------
    # 图2: 误差分析图
    # ---------------------------------------------------------
    print("\n[2] Generating error analysis plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绝对误差
    axes[0].semilogy(x_line, error_line, 'b-', linewidth=2)
    axes[0].set_xlabel(r'$x$', fontsize=12)
    axes[0].set_ylabel(r'Absolute Error $|G_{exact} - G_{PINN}|$', fontsize=12)
    axes[0].set_title(r'Absolute Error along Centerline')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].axhline(y=error_line.mean(), color='r', linestyle='--', 
                    label=f'Mean: {error_line.mean():.2e}')
    axes[0].legend()
    
    # 相对误差
    axes[1].plot(x_line, rel_error_line * 100, 'g-', linewidth=2)
    axes[1].set_xlabel(r'$x$', fontsize=12)
    axes[1].set_ylabel(r'Relative Error (%)', fontsize=12)
    axes[1].set_title(r'Relative Error along Centerline')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].axhline(y=rel_error_line.mean() * 100, color='r', linestyle='--',
                    label=f'Mean: {rel_error_line.mean()*100:.2f}%')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G_CaseA_Error_Analysis.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'G_CaseA_Error_Analysis.pdf'))
    print(f"  Saved: {output_dir}/G_CaseA_Error_Analysis.png")
    plt.close()
    
    # ---------------------------------------------------------
    # 图3: 中心截面的 2D 对比（低分辨率以加速）
    # ---------------------------------------------------------
    print("\n[3] Generating center slice (z=0.5) comparison...")
    print("  Note: Using coarse grid (25x25) for 2D due to computational cost")
    
    n_grid = 25
    x_grid = np.linspace(0, 1, n_grid)
    y_grid = np.linspace(0, 1, n_grid)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_grid = np.full_like(X_grid, 0.5)
    
    # 解析解（使用缓存或重新计算）
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
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin = min(G_exact_2d.min(), G_pinn_2d.min())
    vmax = max(G_exact_2d.max(), G_pinn_2d.max())
    levels = np.linspace(vmin, vmax, 50)
    
    cf1 = axes[0].contourf(X_grid, Y_grid, G_exact_2d, levels=levels, cmap='jet')
    axes[0].set_title('(a) Exact Solution $G(x,y,z=0.5)$')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    fig.colorbar(cf1, ax=axes[0], fraction=0.046, pad=0.04)
    
    cf2 = axes[1].contourf(X_grid, Y_grid, G_pinn_2d, levels=levels, cmap='jet')
    axes[1].set_title('(b) PINN Prediction $G(x,y,z=0.5)$')
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$y$')
    fig.colorbar(cf2, ax=axes[1], fraction=0.046, pad=0.04)
    
    error_levels = np.linspace(0, Error_2d.max(), 50)
    cf3 = axes[2].contourf(X_grid, Y_grid, Error_2d, levels=error_levels, cmap='magma')
    axes[2].set_title(f'(c) Absolute Error (Max: {Error_2d.max():.2e})')
    axes[2].set_xlabel('$x$')
    axes[2].set_ylabel('$y$')
    fig.colorbar(cf3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G_CaseA_2D_HighPrecision.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'G_CaseA_2D_HighPrecision.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_dir}/G_CaseA_2D_HighPrecision.png")
    plt.close()
    
    # 最终统计
    print("\n" + "="*70)
    print("Validation Summary (High Precision):")
    print(f"  Centerline - Max Error:     {error_line.max():.6e}")
    print(f"  Centerline - Mean Error:    {error_line.mean():.6e}")
    print(f"  Centerline - Max Rel Error: {rel_error_line.max():.2%}")
    print(f"  2D Slice   - Max Error:     {Error_2d.max():.6e}")
    print(f"  2D Slice   - Mean Error:    {Error_2d.mean():.6e}")
    print("="*70)
    print(f"\nAll figures saved to: {output_dir}/")

if __name__ == "__main__":
    plot_G_comparison("Results_3D_CaseA/model.pkl")
