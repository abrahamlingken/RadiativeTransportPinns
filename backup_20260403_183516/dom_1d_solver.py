#!/usr/bin/env python
"""
dom_1d_solver.py
一维稳态辐射传输方程离散坐标法(DOM)求解器 - 期刊级绘图风格
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import roots_legendre

# 设置期刊级绘图参数
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['legend.fontsize'] = 12
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11


def dom_1d_solver(kappa=0.5, sigma_s=0.5, N_mu=20, Nx=200, max_iter=1000, tol=1e-10):
    """
    DOM求解1D RTE - 严格按期刊格式输出
    """
    beta = kappa + sigma_s
    
    # 角度离散 (Gauss-Legendre)
    mu_quad, w_quad = roots_legendre(N_mu)
    
    # 分离正负方向
    pos_mask = mu_quad > 0
    neg_mask = mu_quad < 0
    mu_pos = mu_quad[pos_mask]
    mu_neg = mu_quad[neg_mask]
    
    # 空间离散
    x = np.linspace(0, 1, Nx)
    dx = x[1] - x[0]
    I = np.zeros((N_mu, Nx))
    S_scatter = np.zeros(Nx)
    
    print("Starting source iteration...")
    for iteration in range(max_iter):
        I_old = I.copy()
        
        # 正向方向 (mu > 0)
        for j, mu in enumerate(mu_pos):
            idx = np.where(pos_mask)[0][j]
            I[idx, 0] = 1.0
            for i in range(1, Nx):
                I[idx, i] = (mu * I[idx, i-1] + dx * S_scatter[i]) / (mu + beta * dx)
        
        # 反向方向 (mu < 0)
        for j, mu in enumerate(mu_neg):
            idx = np.where(neg_mask)[0][j]
            I[idx, -1] = 0.0
            mu_abs = abs(mu)
            for i in range(Nx-2, -1, -1):
                I[idx, i] = (mu_abs * I[idx, i+1] + dx * S_scatter[i]) / (mu_abs + beta * dx)
        
        # 更新散射源项
        S_scatter = 0.5 * sigma_s * np.dot(w_quad, I)
        
        # 收敛判断
        change = np.linalg.norm(I - I_old) / (np.linalg.norm(I_old) + 1e-10)
        if iteration % 50 == 0:
            print("  Iter %d: change = %.2e" % (iteration, change))
        if change < tol:
            print("  Converged at iteration %d" % iteration)
            break
    
    # 计算辐射热流
    heat_flux = np.dot(mu_quad * w_quad, I)
    
    # ==================== 期刊级绘图 ====================
    X_grid, Mu_grid = np.meshgrid(x, mu_quad)
    
    # 创建图形 - 参考 net_sol.png 的单栏大图风格
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # 绘制等高线 - jet colormap + extend='both'（与 net_sol.png 完全一致）
    levels = np.linspace(0, 1, 21)
    contour = ax.contourf(X_grid, Mu_grid, I, levels=levels, cmap='jet', extend='both')
    
    # 颜色条 - 右侧，带三角形端点，标签旋转（net_sol.png 风格）
    cbar = plt.colorbar(contour, ax=ax, extend='both')
    cbar.set_label(r'$I(x, \mu)$', fontsize=14, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=11)
    
    # 坐标轴标签 - LaTeX 格式
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$\mu$', fontsize=14)
    
    # 标题 - 使用 LaTeX 格式（参考 PINN 预测图的标题风格）
    ax.set_title(r'DOM Reference Solution: $I(x, \mu)$', fontsize=16, pad=15)
    
    # 刻度设置
    ax.set_xlim([0, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(-1, 1, 9))
    ax.tick_params(labelsize=12)
    
    # 添加 mu=0 分界线（白色虚线，与 net_sol.png 一致）
    ax.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('dom_solution.png', dpi=300, bbox_inches='tight')
    print("Saved to dom_solution.png")
    plt.close()
    
    # ==================== 边界对比图（参考 u0.png 风格）====================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：左边界 x=0
    ax1 = axes[0]
    # DOM 解用蓝色实线
    ax1.plot(mu_quad, I[:, 0], 'b-', linewidth=2.5, label='DOM Solution')
    # 理论参考值用红色圆点
    mu_ref_pos = mu_quad[mu_quad > 0]
    ax1.scatter(mu_ref_pos[::2], np.ones_like(mu_ref_pos[::2]), 
                c='red', s=80, zorder=5, label='Reference values')
    
    ax1.set_xlabel(r'$\mu = \cos(\theta)$', fontsize=14)
    ax1.set_ylabel(r'$u^-(x=0)$', fontsize=14)
    ax1.set_title(r'Boundary at $x = 0$', fontsize=14)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-0.05, 1.1])
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # 右图：右边界 x=1
    ax2 = axes[1]
    ax2.plot(mu_quad, I[:, -1], 'b-', linewidth=2.5, label='DOM Solution')
    mu_ref_neg = mu_quad[mu_quad < 0]
    ax2.scatter(mu_ref_neg[::2], np.zeros_like(mu_ref_neg[::2]), 
                c='red', s=80, zorder=5, label='Reference values')
    
    ax2.set_xlabel(r'$\mu = \cos(\theta)$', fontsize=14)
    ax2.set_ylabel(r'$u^+(x=1)$', fontsize=14)
    ax2.set_title(r'Boundary at $x = 1$', fontsize=14)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-0.05, 1.1])
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('dom_boundaries.png', dpi=300, bbox_inches='tight')
    print("Saved to dom_boundaries.png")
    plt.show()
    
    return x, mu_quad, w_quad, I, heat_flux


if __name__ == "__main__":
    x, mu, w, I_solution, q = dom_1d_solver(
        kappa=0.5,
        sigma_s=0.5,
        N_mu=20,
        Nx=200,
        max_iter=1000,
        tol=1e-10
    )
    
    print("\n========== Results ==========")
    print("I_max: %.6f" % I_solution.max())
    print("I_min: %.6f" % I_solution.min())
    print("Heat flux at x=0: %.6f" % q[0])
    print("Heat flux at x=1: %.6f" % q[-1])
