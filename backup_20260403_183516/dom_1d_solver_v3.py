#!/usr/bin/env python
"""
dom_1d_solver_v3.py - 边界对比图风格与PINN结果一致

风格对应：
- 蓝色实线: Exact Solution (DOM作为基准真值)
- 红色圆点: Reference values (PINN预测值)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import roots_legendre
import torch
import os

rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['legend.fontsize'] = 12
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11


def load_pinn_prediction(model_path, x, mu_quad):
    """
    加载PINN模型并预测边界值
    如果模型不存在，返回模拟数据用于展示风格
    """
    if model_path is None or not os.path.exists(model_path):
        print("  [Warning] PINN model not found, using mock data for demonstration")
        # 模拟PINN预测值（添加小误差）
        I_left = np.ones_like(mu_quad) * 0.95  # 左边界近似
        I_left[mu_quad < 0] *= 0.1  # 出射方向有小值
        I_right = np.ones_like(mu_quad) * 0.05
        I_right[mu_quad > 0] = np.linspace(0.1, 0.45, len(mu_quad[mu_quad > 0]))
        return I_left, I_right
    
    # 加载真实PINN模型
    print("  Loading PINN model from:", model_path)
    try:
        sys.path.insert(0, 'Core')
        from ModelClassTorch2 import Pinns
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        # 预测左边界 x=0
        x_left = torch.zeros(len(mu_quad), 1)
        mu_tensor = torch.tensor(mu_quad, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            I_left = model(torch.cat([x_left, mu_tensor], dim=1)).numpy().flatten()
        
        # 预测右边界 x=1
        x_right = torch.ones(len(mu_quad), 1)
        with torch.no_grad():
            I_right = model(torch.cat([x_right, mu_tensor], dim=1)).numpy().flatten()
        
        return I_left, I_right
    except Exception as e:
        print("  Error loading model:", e)
        return None, None


def dom_1d_solver(kappa=0.5, sigma_s=0.5, N_mu=20, Nx=200, 
                  max_iter=1000, tol=1e-10, pinn_model_path=None):
    """
    DOM solver with PINN comparison
    """
    beta = kappa + sigma_s
    
    # Angle discretization
    mu_quad, w_quad = roots_legendre(N_mu)
    pos_mask = mu_quad > 0
    neg_mask = mu_quad < 0
    mu_pos = mu_quad[pos_mask]
    mu_neg = mu_quad[neg_mask]
    
    # Space discretization
    x = np.linspace(0, 1, Nx)
    dx = x[1] - x[0]
    I = np.zeros((N_mu, Nx))
    S_scatter = np.zeros(Nx)
    
    print("Starting source iteration...")
    for iteration in range(max_iter):
        I_old = I.copy()
        
        # Forward sweep
        for j, mu in enumerate(mu_pos):
            idx = np.where(pos_mask)[0][j]
            I[idx, 0] = 1.0
            for i in range(1, Nx):
                I[idx, i] = (mu * I[idx, i-1] + dx * S_scatter[i]) / (mu + beta * dx)
        
        # Backward sweep
        for j, mu in enumerate(mu_neg):
            idx = np.where(neg_mask)[0][j]
            I[idx, -1] = 0.0
            mu_abs = abs(mu)
            for i in range(Nx-2, -1, -1):
                I[idx, i] = (mu_abs * I[idx, i+1] + dx * S_scatter[i]) / (mu_abs + beta * dx)
        
        S_scatter = 0.5 * sigma_s * np.dot(w_quad, I)
        
        change = np.linalg.norm(I - I_old) / (np.linalg.norm(I_old) + 1e-10)
        if iteration % 50 == 0:
            print("  Iter %d: change = %.2e" % (iteration, change))
        if change < tol:
            print("  Converged at iteration %d" % iteration)
            break
    
    heat_flux = np.dot(mu_quad * w_quad, I)
    
    # ==================== 主解图 ====================
    X_grid, Mu_grid = np.meshgrid(x, mu_quad)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    levels = np.linspace(0, 1, 21)
    contour = ax.contourf(X_grid, Mu_grid, I, levels=levels, cmap='jet', extend='both')
    
    cbar = plt.colorbar(contour, ax=ax, extend='both')
    cbar.set_label(r'$I(x, \mu)$', fontsize=14, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=11)
    
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$\mu$', fontsize=14)
    ax.set_title(r'DOM Reference Solution: $I(x, \mu)$', fontsize=16, pad=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(-1, 1, 9))
    ax.tick_params(labelsize=12)
    ax.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('dom_solution.png', dpi=300, bbox_inches='tight')
    print("Saved to dom_solution.png")
    plt.close()
    
    # ==================== 边界对比图（与03_boundary_comparison.png风格一致）====================
    
    # 尝试加载PINN预测值
    print("\nLoading PINN predictions for boundary comparison...")
    I_pinn_left, I_pinn_right = load_pinn_prediction(pinn_model_path, x, mu_quad)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 选择用于标记的mu值（均匀分布）
    step = 2
    mu_markers = mu_quad[::step]
    
    # 左图：左边界 x=0
    ax1 = axes[0]
    # 蓝色实线: Exact Solution (DOM)
    ax1.plot(mu_quad, I[:, 0], 'b-', linewidth=2.5, label='Exact Solution')
    
    # 红色圆点: PINN Prediction / Reference values
    if I_pinn_left is not None:
        I_pinn_markers_left = I_pinn_left[::step]
        ax1.scatter(mu_markers, I_pinn_markers_left, c='red', s=80, zorder=5,
                   label='Reference values')
    
    ax1.set_xlabel(r'$\mu = \cos(\theta)$', fontsize=14)
    ax1.set_ylabel(r'$I^-(x = 0)$', fontsize=14)
    ax1.set_title(r'Boundary comparison at $x = 0$', fontsize=14)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-0.05, 1.1])
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # 右图：右边界 x=1
    ax2 = axes[1]
    # 蓝色实线: Exact Solution (DOM)
    ax2.plot(mu_quad, I[:, -1], 'b-', linewidth=2.5, label='Exact Solution')
    
    # 红色圆点: PINN Prediction / Reference values
    if I_pinn_right is not None:
        I_pinn_markers_right = I_pinn_right[::step]
        ax2.scatter(mu_markers, I_pinn_markers_right, c='red', s=80, zorder=5,
                   label='Reference values')
    
    ax2.set_xlabel(r'$\mu = \cos(\theta)$', fontsize=14)
    ax2.set_ylabel(r'$I^+(x = 1)$', fontsize=14)
    ax2.set_title(r'Boundary comparison at $x = 1$', fontsize=14)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-0.05, 1.1])
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('dom_boundaries.png', dpi=300, bbox_inches='tight')
    print("Saved to dom_boundaries.png")
    plt.close()
    
    return x, mu_quad, w_quad, I, heat_flux


if __name__ == "__main__":
    import sys
    
    # 可以指定PINN模型路径进行真实对比
    # pinn_path = "Results_1D_CaseA/TrainedModel/model.pkl"
    # 在代码底部修改
    pinn_path = "Results_1D_CaseA/TrainedModel/model.pkl"  # 指定真实模型  
    
    x, mu, w, I_solution, q = dom_1d_solver(
        kappa=0.5, sigma_s=0.5, N_mu=20, Nx=200, 
        max_iter=1000, tol=1e-10, pinn_model_path=pinn_path
    )
    
    print("\n========== Results ==========")
    print("I_max: %.6f" % I_solution.max())
    print("I_min: %.6f" % I_solution.min())
    print("Heat flux at x=0: %.6f" % q[0])
    print("Heat flux at x=1: %.6f" % q[-1])
