#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_3d_pure_absorption.py - 3D纯吸收案例验证

NOTE: Source term decoupled from kappa (Mathematical formulation)
New analytical formula: I = ∫ S·exp(-κ·l) dl [old: ∫ κ·S·exp(-κ·l) dl]
where S = Ib(r) = max(0, 1 - 2r)

绘图风格与 plot_3d_paper_figures.py 保持一致
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
# 字体设置（与 plot_3d_paper_figures.py 完全一致）
# ==========================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = False
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.join(PROJECT_ROOT, 'Core')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CORE_PATH not in sys.path:
    sys.path.insert(0, CORE_PATH)

import ModelClassTorch2
from ModelClassTorch2 import Pinns
from EquationModels.RadTrans3D_Complex import RadTrans3D_Physics

# ==========================================
# 物理参数
# ==========================================
KAPPA = 5.0
CENTER = np.array([0.5, 0.5, 0.5])

def source_term(x, y, z):
    """球形热源 - 高度局部化: max(0, 1.0 - 5.0*r)，仅在r<0.2有源"""
    r = np.sqrt((x - CENTER[0])**2 + (y - CENTER[1])**2 + (z - CENTER[2])**2)
    return np.maximum(0.0, 1.0 - 5.0 * r)

def compute_exact_intensity_single(x, y, z, theta, phi, num_points=300):
    """计算单一方向 (theta, phi) 的精确强度 I(x,y,z,theta,phi)"""
    pos = np.array([x, y, z], dtype=np.float64)
    
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
    t_candidates = []
    for i in range(3):
        if abs(s_vec[i]) > 1e-14:
            t_to_0 = pos[i] / s_vec[i]
            t_to_1 = (pos[i] - 1.0) / s_vec[i]
            if t_to_0 > 1e-14:
                t_candidates.append(t_to_0)
            if t_to_1 > 1e-14:
                t_candidates.append(t_to_1)
    
    if len(t_candidates) == 0:
        return 0.0
    
    L = min(t_candidates)
    if L <= 1e-14:
        return 0.0
    
    # Gauss-Legendre 积分
    xi, w = roots_legendre(num_points)
    l_vals = 0.5 * L * (xi + 1.0)
    weights = 0.5 * L * w
    
    integrand = np.zeros(num_points)
    for i in range(num_points):
        curr_pos = pos - l_vals[i] * s_vec
        S_val = source_term(curr_pos[0], curr_pos[1], curr_pos[2])
        # NOTE: Source term decoupled from kappa
        # New integrand: S * exp(-kappa * l) [old: kappa * S * exp(-kappa * l)]
        integrand[i] = S_val * np.exp(-KAPPA * l_vals[i])
    
    intensity = np.sum(integrand * weights)
    return max(0.0, intensity)

# ==========================================
# 精确 G(x) 计算
# ==========================================
class ExactGSolver:
    def __init__(self, n_theta=32, n_phi=64):
        self.n_theta = n_theta
        self.n_phi = n_phi
        
        # theta: Gauss-Legendre
        mu, w_mu = roots_legendre(n_theta)
        self.theta_q = np.arccos(-mu)
        self.w_theta = w_mu
        
        # phi: 均匀分布
        self.phi_q = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        self.phi_q += np.pi / n_phi
        self.w_phi = np.full(n_phi, 2.0 * np.pi / n_phi)
        
        # 2D网格
        self.theta_grid, self.phi_grid = np.meshgrid(self.theta_q, self.phi_q, indexing='ij')
        
        # 权重
        theta_weights = self.w_theta.reshape(-1, 1)
        phi_weights = self.w_phi.reshape(1, -1)
        self.weights = theta_weights * phi_weights
    
    def compute_G(self, x, y, z):
        """计算 G(x)"""
        I_vals = np.zeros((self.n_theta, self.n_phi))
        
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                theta = self.theta_grid[i, j]
                phi = self.phi_grid[i, j]
                I_vals[i, j] = compute_exact_intensity_single(x, y, z, theta, phi, num_points=100)
        
        G = np.sum(I_vals * self.weights)
        return G
    
    def compute_G_field(self, X, Y, Z, verbose=True):
        """计算整个场的 G(x)"""
        G = np.zeros_like(X)
        total = X.size
        
        for idx in range(total):
            x_val = X.flat[idx]
            y_val = Y.flat[idx]
            z_val = Z.flat[idx]
            G.flat[idx] = self.compute_G(x_val, y_val, z_val)
            
            if verbose and total > 10 and idx % max(1, total//10) == 0:
                print(f"    Progress: {idx}/{total} ({100*idx/total:.0f}%)")
        
        return G

# ==========================================
# PINN预测
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
# 主程序
# ==========================================
def plot_G_comparison(model_path="Results_3D_CaseA/model.pkl"):
    """生成G(x)对比图 - 风格与 plot_3d_paper_figures.py 一致"""
    
    print("="*70)
    print("3D Pure Absorption Validation")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine = RadTrans3D_Physics(
        kappa_val=1.0, sigma_s_val=0.0, g_val=0.0,
        n_theta=8, n_phi=16, dev=device
    )
    
    exact_solver = ExactGSolver(n_theta=16, n_phi=32)
    
    output_dir = 'Figures_3D_Validation'
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # 图1: 中心线 G(x) 对比
    # ==========================================
    print("\n[1] Generating G(x) along centerline...")
    
    n_points = 50
    x_line = np.linspace(0.05, 0.95, n_points)
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
    
    # 绘图 - 风格与 plot_3d_paper_figures.py 一致
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 精确解：黑色实线（与 plot_3d_paper_figures 中 case line 风格一致）
    ax.plot(x_line, G_exact_line, 'k-', linewidth=2, label='Exact Analytical',
            marker='o', markevery=5, markersize=6)
    
    # PINN：红色虚线（对比色）
    ax.plot(x_line, G_pinn_line, color='red', linestyle='--', linewidth=2,
            label='PINN Prediction', marker='s', markevery=5, markersize=6)
    
    ax.set_xlabel(r'$x$ (optical depth)', fontsize=12)
    ax.set_ylabel(r'$G(x, 0.5, 0.5)$', fontsize=12)
    ax.set_title(r'Incident Radiation along Centerline ($y=z=0.5$)', fontsize=13)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G_CaseA_Centerline_HighPrecision.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'G_CaseA_Centerline_HighPrecision.pdf'))
    print(f"  Saved: {output_dir}/G_CaseA_Centerline_HighPrecision.png")
    plt.close()
    
    # ==========================================
    # 图2: 2D 中心截面 (z=0.5) 对比 - 风格与 plot_3d_paper_figures 热图一致
    # ==========================================
    print("\n[2] Generating center slice heatmaps...")
    
    n_grid = 50
    x_grid = np.linspace(0.05, 0.95, n_grid)
    y_grid = np.linspace(0.05, 0.95, n_grid)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_grid = np.full_like(X_grid, 0.5)
    
    print("  Computing exact 2D G field...")
    G_exact_2d = exact_solver.compute_G_field(X_grid, Y_grid, Z_grid, verbose=True)
    
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
    
    # 1x3 子图 - 与 plot_3d_paper_figures 风格一致
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    vmin = min(G_exact_2d.min(), G_pinn_2d.min())
    vmax = max(G_exact_2d.max(), G_pinn_2d.max())
    levels = np.linspace(vmin, vmax, 20)
    
    # (a) 精确解 - cmap='hot' 与 plot_3d_paper_figures 一致
    im1 = axes[0].contourf(X_grid, Y_grid, G_exact_2d, levels=levels, 
                           cmap='hot', extend='both')
    axes[0].set_xlabel(r'$x$', fontsize=11)
    axes[0].set_ylabel(r'$y$', fontsize=11)
    axes[0].set_title('Exact Solution', fontsize=11)
    axes[0].set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=axes[0], pad=0.02)
    cbar1.set_label(r'$G$', fontsize=10)
    
    # (b) PINN预测 - 统一色标
    im2 = axes[1].contourf(X_grid, Y_grid, G_pinn_2d, levels=levels,
                           cmap='hot', extend='both')
    axes[1].set_xlabel(r'$x$', fontsize=11)
    axes[1].set_ylabel(r'$y$', fontsize=11)
    axes[1].set_title('PINN Prediction', fontsize=11)
    axes[1].set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=axes[1], pad=0.02)
    cbar2.set_label(r'$G$', fontsize=10)
    
    # (c) 误差 - 使用 Reds 突出差异
    err_levels = np.linspace(0, Error_2d.max(), 20)
    im3 = axes[2].contourf(X_grid, Y_grid, Error_2d, levels=err_levels,
                           cmap='Reds', extend='max')
    axes[2].set_xlabel(r'$x$', fontsize=11)
    axes[2].set_ylabel(r'$y$', fontsize=11)
    axes[2].set_title(f'Absolute Error (Max: {Error_2d.max():.2e})', fontsize=11)
    axes[2].set_aspect('equal')
    cbar3 = plt.colorbar(im3, ax=axes[2], pad=0.02)
    cbar3.set_label(r'$|G_{exact} - G_{PINN}|$', fontsize=10)
    
    plt.suptitle(r'Incident Radiation at $z=0.5$ Plane', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G_CaseA_2D_HighPrecision.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'G_CaseA_2D_HighPrecision.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_dir}/G_CaseA_2D_HighPrecision.png")
    plt.close()
    
    print(f"\n  2D Error Statistics:")
    print(f"  Max Error: {Error_2d.max():.4e}")
    print(f"  Mean Error: {Error_2d.mean():.4e}")
    
    print("\n" + "="*70)
    print("Validation Complete!")
    print(f"Output: {output_dir}/")
    print("="*70)

if __name__ == "__main__":
    plot_G_comparison("Results_3D_CaseA/model.pkl")
