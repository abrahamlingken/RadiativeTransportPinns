#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_3d_pure_absorption.py - 3D纯吸收案例验证（G(x)对比版）

对比入射辐射 G(x) = ∫∫ I(x,s) dΩ：
- 解析解：通过数值积分所有方向的反向射线追踪结果
- PINN预测：使用RadTrans3D_Physics.compute_incident_radiation()

风格与 plot_3d_paper_figures.py 保持一致
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import integrate

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

# 设置顶刊风格字体（与 plot_3d_paper_figures.py 一致）
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
KAPPA = 1.0  # 与 Case 3D-A 一致
CENTER = np.array([0.5, 0.5, 0.5])

def source_term(x, y, z):
    """球形热源: max(0, 1.0 - 2.0*r)"""
    r = np.sqrt((x - CENTER[0])**2 + (y - CENTER[1])**2 + (z - CENTER[2])**2)
    return np.maximum(0.0, 1.0 - 2.0 * r)

def compute_exact_intensity_single(x, y, z, s_vec, num_points=100):
    """
    计算单一方向的精确强度 I(x,y,z,s_vec)
    使用反向射线追踪（Beer-Lambert定律）
    """
    pos = np.array([x, y, z])
    s_vec = np.array(s_vec)
    s_vec = s_vec / np.linalg.norm(s_vec)
    
    # 计算到边界的距离
    t_bounds = []
    for i in range(3):
        if s_vec[i] > 1e-8:
            t_bounds.append(pos[i] / s_vec[i])
        elif s_vec[i] < -1e-8:
            t_bounds.append((pos[i] - 1.0) / s_vec[i])
    
    L = min(t_bounds) if t_bounds else 0.0
    
    # 沿射线积分
    l_vals = np.linspace(0, L, num_points)
    integrand = np.zeros_like(l_vals)
    
    for i, l in enumerate(l_vals):
        curr_pos = pos - l * s_vec
        S_val = source_term(curr_pos[0], curr_pos[1], curr_pos[2])
        integrand[i] = KAPPA * S_val * np.exp(-KAPPA * l)
    
    try:
        intensity = np.trapezoid(integrand, l_vals)
    except AttributeError:
        intensity = np.trapz(integrand, l_vals)
    
    return intensity

def compute_exact_G_scalar(x, y, z, n_theta=16, n_phi=32):
    """
    计算精确解的 G(x) = ∫∫ I(x,s) sin(θ) dθ dφ / (4π)
    通过对多个方向的 I(x,s) 数值积分
    """
    # 生成方向网格
    theta_vals = np.linspace(0, np.pi, n_theta)
    phi_vals = np.linspace(0, 2*np.pi, n_phi)
    
    I_vals = np.zeros((n_theta, n_phi))
    
    for i, theta in enumerate(theta_vals):
        for j, phi in enumerate(phi_vals):
            s_vec = [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ]
            I_vals[i, j] = compute_exact_intensity_single(x, y, z, s_vec)
    
    # 立体角积分: dΩ = sin(θ) dθ dφ
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals, indexing='ij')
    integrand = I_vals * np.sin(theta_grid)
    
    G = np.trapz(np.trapz(integrand, phi_vals), theta_vals)
    
    return G

def compute_exact_G_field(X, Y, Z):
    """
    计算整个场的 G(x)
    向量化计算以提高速度
    """
    print("  正在计算解析解 G(x)... 这需要一些时间（对128个方向积分）")
    G = np.zeros_like(X)
    total = X.size
    
    for idx in range(total):
        i = idx // X.shape[1]
        j = idx % X.shape[1]
        G[i, j] = compute_exact_G_scalar(X[i,j], Y[i,j], Z[i,j])
        
        if idx % 100 == 0:
            print(f"    进度: {idx}/{total} ({100*idx/total:.1f}%)")
    
    return G

# ==========================================
# 2. PINN预测（使用物理引擎计算G）
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
# 3. 主绘图程序
# ==========================================
def plot_G_comparison(model_path="Results_3D_CaseA/model.pkl"):
    """生成G(x)对比图，风格与plot_3d_paper_figures一致"""
    
    print("="*70)
    print("3D Pure Absorption Validation: G(x) Comparison")
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
    
    output_dir = 'Figures_3D_Validation'
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 图1: 沿中心线的 G(x) 对比
    # ---------------------------------------------------------
    print("\n[1] Generating centerline comparison...")
    
    n_points = 100
    x_line = torch.linspace(0, 1, n_points, device=device)
    y_line = torch.ones_like(x_line) * 0.5
    z_line = torch.ones_like(x_line) * 0.5
    
    # 解析解 G
    print("  计算解析解 G(x)...")
    G_exact_line = np.array([
        compute_exact_G_scalar(x.item(), 0.5, 0.5) 
        for x in x_line.cpu()
    ])
    
    # PINN预测 G
    if os.path.exists(model_path):
        print("  计算PINN预测 G(x)...")
        G_pinn_line = load_pinn_G(model_path, x_line, y_line, z_line, engine)
    else:
        print(f"  [错误] 找不到模型文件 {model_path}")
        return
    
    # 绘图（与plot_3d_paper_figures一致的风格）
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_np = x_line.cpu().numpy()
    
    # 解析解：黑色实线
    ax.plot(x_np, G_exact_line, 'k-', linewidth=2.5, 
            label='Exact Analytical', zorder=2)
    
    # PINN：红色虚线带圆圈（与validate原脚本一致的颜色）
    ax.plot(x_np, G_pinn_line, color='#D62728', linestyle='--', 
            linewidth=2.0, marker='o', markersize=6, 
            markerfacecolor='none', markevery=10,
            label='PINN Prediction', zorder=3)
    
    ax.set_xlabel(r'$x$ (optical depth)', fontsize=12)
    ax.set_ylabel(r'$G(x, 0.5, 0.5)$', fontsize=12)
    ax.set_title(r'3D Pure Absorption: $G(x)$ along Centerline ($y=z=0.5$)', fontsize=13)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G_CaseA_Centerline_Validation.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'G_CaseA_Centerline_Validation.pdf'))
    print(f"  Saved: {output_dir}/G_CaseA_Centerline_Validation.png")
    plt.close()
    
    # ---------------------------------------------------------
    # 图2: 中心截面的 2D 对比
    # ---------------------------------------------------------
    print("\n[2] Generating center slice (z=0.5) comparison...")
    print("  注意：这是100x100网格，每个点需要对128个方向积分，计算量较大...")
    
    n_grid = 50  # 减小网格以平衡精度与速度
    x_grid = torch.linspace(0, 1, n_grid, device=device)
    y_grid = torch.linspace(0, 1, n_grid, device=device)
    X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
    
    x_flat = X_grid.reshape(-1)
    y_flat = Y_grid.reshape(-1)
    z_flat = torch.ones_like(x_flat) * 0.5
    
    # PINN预测（较快）
    print("  计算PINN预测的 2D G 场...")
    G_pinn_flat = load_pinn_G(model_path, x_flat, y_flat, z_flat, engine)
    G_pinn_2d = G_pinn_flat.reshape(n_grid, n_grid)
    
    # 解析解（较慢，可选：使用粗网格或插值）
    print("  计算解析解的 2D G 场（这可能需要几分钟）...")
    print("  提示：如果太慢，可以减小n_grid或跳过此图")
    
    # 为了速度，这里使用粗网格计算然后插值
    n_coarse = 20
    x_coarse = np.linspace(0, 1, n_coarse)
    y_coarse = np.linspace(0, 1, n_coarse)
    X_c, Y_c = np.meshgrid(x_coarse, y_coarse)
    
    G_exact_coarse = np.array([
        [compute_exact_G_scalar(X_c[i,j], Y_c[i,j], 0.5) for j in range(n_coarse)]
        for i in range(n_coarse)
    ])
    
    # 插值到细网格
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (x_coarse, y_coarse), 
        G_exact_coarse,
        bounds_error=False,
        fill_value=None
    )
    
    points = np.stack([X_grid.cpu().numpy().flatten(), Y_grid.cpu().numpy().flatten()], axis=-1)
    G_exact_2d = interp(points).reshape(n_grid, n_grid)
    
    # 计算误差
    Error_2d = np.abs(G_exact_2d - G_pinn_2d)
    
    # 绘图：1x3 子图（与validate原脚本一致）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin = min(G_exact_2d.min(), G_pinn_2d.min())
    vmax = max(G_exact_2d.max(), G_pinn_2d.max())
    levels = np.linspace(vmin, vmax, 50)
    
    # (a) 精确解
    cf1 = axes[0].contourf(X_grid.cpu().numpy(), Y_grid.cpu().numpy(), G_exact_2d, 
                           levels=levels, cmap='jet')
    axes[0].set_title('(a) Exact Solution $G(x,y,z=0.5)$')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    fig.colorbar(cf1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # (b) PINN预测
    cf2 = axes[1].contourf(X_grid.cpu().numpy(), Y_grid.cpu().numpy(), G_pinn_2d, 
                           levels=levels, cmap='jet')
    axes[1].set_title('(b) PINN Prediction $G(x,y,z=0.5)$')
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$y$')
    fig.colorbar(cf2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # (c) 误差
    error_levels = np.linspace(0, Error_2d.max(), 50)
    cf3 = axes[2].contourf(X_grid.cpu().numpy(), Y_grid.cpu().numpy(), Error_2d, 
                           levels=error_levels, cmap='magma')
    axes[2].set_title(f'(c) Absolute Error (Max: {Error_2d.max():.2e})')
    axes[2].set_xlabel('$x$')
    axes[2].set_ylabel('$y$')
    fig.colorbar(cf3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G_CaseA_2D_Validation.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'G_CaseA_2D_Validation.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_dir}/G_CaseA_2D_Validation.png")
    plt.close()
    
    # 输出误差统计
    print("\n" + "="*70)
    print("Validation Summary:")
    print(f"  Centerline - Max Error: {np.max(np.abs(G_exact_line - G_pinn_line)):.6e}")
    print(f"  Centerline - L2 Error:  {np.sqrt(np.mean((G_exact_line - G_pinn_line)**2)):.6e}")
    print(f"  2D Slice   - Max Error: {Error_2d.max():.6e}")
    print(f"  2D Slice   - Mean Error:{Error_2d.mean():.6e}")
    print("="*70)
    print(f"\nAll figures saved to: {output_dir}/")

if __name__ == "__main__":
    plot_G_comparison("Results_3D_CaseA/model.pkl")
