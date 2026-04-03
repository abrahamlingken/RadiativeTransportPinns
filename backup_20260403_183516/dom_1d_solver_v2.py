#!/usr/bin/env python
"""
dom_1d_solver_v2.py - 改进版边界图取点
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import roots_legendre

rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['legend.fontsize'] = 11
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11


def dom_1d_solver(kappa=0.5, sigma_s=0.5, N_mu=20, Nx=200, max_iter=1000, tol=1e-10):
    """DOM solver"""
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
    
    # ==================== 改进的边界对比图 ====================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 选取用于标记的mu值（均匀分布，每隔2个取一个）
    step = 2  # 调整此值可改变点的密度
    mu_markers = mu_quad[::step]
    I_left_markers = I[::step, 0]
    I_right_markers = I[::step, -1]
    
    # 左图：左边界 x=0
    ax1 = axes[0]
    # DOM解 - 蓝色实线（全范围）
    ax1.plot(mu_quad, I[:, 0], 'b-', linewidth=2.5, label='DOM Solution')
    
    # 区分入射和出射方向的参考值
    # 入射方向 (mu > 0): I = 1 (边界条件)
    mu_incident_0 = mu_markers[mu_markers > 0]
    I_incident_0 = np.ones_like(mu_incident_0)
    # 出射方向 (mu < 0): I = DOM计算值
    mu_outgoing_0 = mu_markers[mu_markers < 0]
    idx_out_0 = np.where(mu_markers < 0)[0]
    I_outgoing_0 = I_left_markers[mu_markers < 0]
    
    # 用红色标记入射边界条件
    ax1.scatter(mu_incident_0, I_incident_0, c='red', s=80, zorder=5, 
                label='Incident (BC: I=1)')
    # 用橙色标记出射计算值
    ax1.scatter(mu_outgoing_0, I_outgoing_0, c='orange', s=60, marker='s', zorder=5,
                label='Outgoing (Computed)')
    
    ax1.set_xlabel(r'$\mu = \cos(\theta)$', fontsize=14)
    ax1.set_ylabel(r'$I(x=0, \mu)$', fontsize=14)
    ax1.set_title(r'Boundary at $x = 0$', fontsize=14)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-0.05, 1.1])
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    
    # 右图：右边界 x=1
    ax2 = axes[1]
    # DOM解 - 蓝色实线（全范围）
    ax2.plot(mu_quad, I[:, -1], 'b-', linewidth=2.5, label='DOM Solution')
    
    # 入射方向 (mu < 0): I = 0 (边界条件)
    mu_incident_1 = mu_markers[mu_markers < 0]
    I_incident_1 = np.zeros_like(mu_incident_1)
    # 出射方向 (mu > 0): I = DOM计算值
    mu_outgoing_1 = mu_markers[mu_markers > 0]
    I_outgoing_1 = I_right_markers[mu_markers > 0]
    
    # 用红色标记入射边界条件
    ax2.scatter(mu_incident_1, I_incident_1, c='red', s=80, zorder=5,
                label='Incident (BC: I=0)')
    # 用橙色标记出射计算值
    ax2.scatter(mu_outgoing_1, I_outgoing_1, c='orange', s=60, marker='s', zorder=5,
                label='Outgoing (Computed)')
    
    ax2.set_xlabel(r'$\mu = \cos(\theta)$', fontsize=14)
    ax2.set_ylabel(r'$I(x=1, \mu)$', fontsize=14)
    ax2.set_title(r'Boundary at $x = 1$', fontsize=14)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-0.05, 1.1])
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('dom_boundaries.png', dpi=300, bbox_inches='tight')
    print("Saved to dom_boundaries.png")
    plt.close()
    
    return x, mu_quad, w_quad, I, heat_flux


if __name__ == "__main__":
    x, mu, w, I_solution, q = dom_1d_solver(
        kappa=0.5, sigma_s=0.5, N_mu=20, Nx=200, max_iter=1000, tol=1e-10
    )
    
    print("\n========== Results ==========")
    print("I_max: %.6f" % I_solution.max())
    print("I_min: %.6f" % I_solution.min())
    print("Heat flux at x=0: %.6f" % q[0])
    print("Heat flux at x=1: %.6f" % q[-1])
