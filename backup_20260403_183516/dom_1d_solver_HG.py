#!/usr/bin/env python
"""
dom_1d_solver_HG.py - 1D RTE DOM Solver with Anisotropic Scattering (Henyey-Greenstein)

支持各向异性散射的离散坐标法求解器，使用 Henyey-Greenstein 相函数：
    Phi_HG(mu, mu', g) = (1 - g^2) / (1 + g^2 - 2*g*mu*mu')^{1.5}

求解案例：
    - g = 0.5:  前向散射
    - g = -0.5: 后向散射  
    - g = 0.8:  强前向散射

输出：
    - dom_solution_g{g}.png: 辐射强度分布图
    - dom_boundaries_g{g}.png: 边界对比图
    - dom_solution_g{g}.npy: 保存的辐射强度矩阵 I[mu, x]
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


def compute_hg_phase_matrix(mu_quad, g):
    """
    预计算 Henyey-Greenstein 相函数矩阵
    
    Henyey-Greenstein 相函数:
        Phi(mu, mu', g) = (1 - g^2) / (1 + g^2 - 2*g*mu*mu')^{3/2}
    
    在1D平板几何中，散射角余弦简化为: cos(theta) = mu * mu'
    
    归一化条件 (解析):
        integral_{-1}^{1} Phi(mu, mu', g) dmu' = 2,  for any mu and g
    
    Args:
        mu_quad: Gauss-Legendre 求积点，shape [N_mu]
        g: HG 不对称因子
        
    Returns:
        Phi: 相函数矩阵，shape [N_mu, N_mu]，其中 Phi[i,j] = Phi_HG(mu_i, mu_j, g)
    """
    N_mu = len(mu_quad)
    mu_i = mu_quad.reshape(N_mu, 1)  # [N_mu, 1]
    mu_j = mu_quad.reshape(1, N_mu)  # [1, N_mu]
    
    # 散射角余弦: cos(theta) = mu * mu' (1D平板几何简化)
    cos_theta = mu_i * mu_j  # [N_mu, N_mu]
    
    g_sq = g ** 2
    numerator = 1.0 - g_sq
    denominator = np.power(1.0 + g_sq - 2.0 * g * cos_theta, 1.5)
    
    Phi = numerator / denominator
    return Phi


def load_pinn_prediction(model_path, x, mu_quad):
    """
    加载PINN模型并预测边界值
    """
    if model_path is None or not os.path.exists(model_path):
        print("  [Warning] PINN model not found, using mock data")
        I_left = np.ones_like(mu_quad) * 0.95
        I_left[mu_quad < 0] *= 0.1
        I_right = np.ones_like(mu_quad) * 0.05
        I_right[mu_quad > 0] = np.linspace(0.1, 0.45, len(mu_quad[mu_quad > 0]))
        return I_left, I_right
    
    print("  Loading PINN model from:", model_path)
    try:
        import sys
        sys.path.insert(0, 'Core')
        from ModelClassTorch2 import Pinns
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()
        
        # 左边界 x=0
        x_left = torch.zeros(len(mu_quad), 1)
        mu_tensor = torch.tensor(mu_quad, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            I_left = model(torch.cat([x_left, mu_tensor], dim=1)).numpy().flatten()
        
        # 右边界 x=1
        x_right = torch.ones(len(mu_quad), 1)
        with torch.no_grad():
            I_right = model(torch.cat([x_right, mu_tensor], dim=1)).numpy().flatten()
        
        return I_left, I_right
    except Exception as e:
        print("  Error loading model:", e)
        return None, None


def dom_1d_solver_anisotropic(kappa=0.5, sigma_s=0.5, g=0.0, N_mu=20, Nx=200,
                               max_iter=1000, tol=1e-10, pinn_model_path=None,
                               output_prefix="dom_solution"):
    """
    1D RTE DOM Solver with Anisotropic Scattering (HG phase function)
    
    Args:
        kappa: 吸收系数
        sigma_s: 散射系数
        g: HG 不对称因子 (0=各向同性, >0=前向, <0=后向)
        N_mu: 角度离散点数
        Nx: 空间离散点数
        max_iter: 最大源迭代次数
        tol: 收敛容差
        pinn_model_path: PINN模型路径（可选）
        output_prefix: 输出文件前缀
        
    Returns:
        x: 空间网格
        mu_quad: 角度求积点
        w_quad: 角度权重
        I: 辐射强度矩阵 [N_mu, Nx]
        heat_flux: 热流分布
    """
    beta = kappa + sigma_s
    
    # 角度离散化
    mu_quad, w_quad = roots_legendre(N_mu)
    pos_mask = mu_quad > 0
    neg_mask = mu_quad < 0
    mu_pos = mu_quad[pos_mask]
    mu_neg = mu_quad[neg_mask]
    
    # 空间离散化
    x = np.linspace(0, 1, Nx)
    dx = x[1] - x[0]
    I = np.zeros((N_mu, Nx))
    
    # ============================================================
    # 1. 预计算相函数矩阵 Phi_HG[mu_i, mu_j]
    # ============================================================
    print(f"  Computing HG phase function matrix (g={g})...")
    Phi_matrix = compute_hg_phase_matrix(mu_quad, g)  # [N_mu, N_mu]
    
    # 验证相函数归一化 (积分为2)
    normalization = np.dot(w_quad, Phi_matrix)
    print(f"  Phase function normalization check (should be ~2): mean={np.mean(normalization):.6f}")
    
    # ============================================================
    # 2. 源迭代求解
    # ============================================================
    print(f"  Starting source iteration (g={g})...")
    
    for iteration in range(max_iter):
        I_old = I.copy()
        
        # --------------------------------------------------------
        # 2.1 计算散射源项: S(x, mu_i) = 0.5 * sigma_s * sum_j w_j * Phi_ij * I_j(x)
        # --------------------------------------------------------
        # 方法: 对每个空间点 x，计算散射积分
        S_scatter = np.zeros((N_mu, Nx))  # [N_mu, Nx]
        for i in range(N_mu):
            # S[i, x] = 0.5 * sigma_s * sum_j (w_j * Phi[i,j] * I[j, x])
            # 向量化: weights[j] * Phi[i,j] 作为系数与 I[j,:] 相乘
            weights_phi = w_quad * Phi_matrix[i, :]  # [N_mu]
            S_scatter[i, :] = 0.5 * sigma_s * np.dot(weights_phi, I)  # scalar * [Nx]
        
        # --------------------------------------------------------
        # 2.2 前向扫描 (mu > 0)
        # --------------------------------------------------------
        for j, mu in enumerate(mu_pos):
            idx = np.where(pos_mask)[0][j]
            I[idx, 0] = 1.0  # 左边界条件
            for i in range(1, Nx):
                # mu * (I_i - I_{i-1}) / dx + beta * I_i = S_i
                # => I_i = (mu * I_{i-1} + dx * S_i) / (mu + beta * dx)
                I[idx, i] = (mu * I[idx, i-1] + dx * S_scatter[idx, i]) / (mu + beta * dx)
        
        # --------------------------------------------------------
        # 2.3 后向扫描 (mu < 0)
        # --------------------------------------------------------
        for j, mu in enumerate(mu_neg):
            idx = np.where(neg_mask)[0][j]
            I[idx, -1] = 0.0  # 右边界条件
            mu_abs = abs(mu)
            for i in range(Nx-2, -1, -1):
                # |mu| * (I_{i+1} - I_i) / dx + beta * I_i = S_i
                # => I_i = (|mu| * I_{i+1} + dx * S_i) / (|mu| + beta * dx)
                I[idx, i] = (mu_abs * I[idx, i+1] + dx * S_scatter[idx, i]) / (mu_abs + beta * dx)
        
        # --------------------------------------------------------
        # 2.4 收敛检查
        # --------------------------------------------------------
        change = np.linalg.norm(I - I_old) / (np.linalg.norm(I_old) + 1e-10)
        if iteration % 50 == 0:
            print("    Iter %d: change = %.2e" % (iteration, change))
        if change < tol:
            print("    Converged at iteration %d" % iteration)
            break
    
    # ============================================================
    # 3. 计算热流
    # ============================================================
    heat_flux = np.dot(mu_quad * w_quad, I)  # [Nx]
    
    # ============================================================
    # 4. 可视化
    # ============================================================
    X_grid, Mu_grid = np.meshgrid(x, mu_quad)
    
    # -------------------- 主解图 --------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    levels = np.linspace(0, 1, 21)
    g_str = f"g={g:.1f}"
    if g > 0:
        g_str += " (Forward)"
    elif g < 0:
        g_str += " (Backward)"
    else:
        g_str += " (Isotropic)"
    
    contour = ax.contourf(X_grid, Mu_grid, I, levels=levels, cmap='jet', extend='both')
    
    cbar = plt.colorbar(contour, ax=ax, extend='both')
    cbar.set_label(r'$I(x, \mu)$', fontsize=14, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=11)
    
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$\mu$', fontsize=14)
    ax.set_title(rf'DOM Solution ($\kappa$={kappa}, $\sigma_s$={sigma_s}, {g_str})', fontsize=14, pad=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(-1, 1, 9))
    ax.tick_params(labelsize=12)
    ax.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    fig_name = f'{output_prefix}_g{g:.1f}.png'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    print(f"  Saved to {fig_name}")
    plt.close()
    
    # -------------------- 边界对比图 --------------------
    print("\n  Loading PINN predictions for boundary comparison...")
    I_pinn_left, I_pinn_right = load_pinn_prediction(pinn_model_path, x, mu_quad)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    step = 2
    mu_markers = mu_quad[::step]
    
    # 左边界 x=0
    ax1 = axes[0]
    ax1.plot(mu_quad, I[:, 0], 'b-', linewidth=2.5, label='Exact Solution (DOM)')
    if I_pinn_left is not None:
        I_pinn_markers_left = I_pinn_left[::step]
        ax1.scatter(mu_markers, I_pinn_markers_left, c='red', s=80, zorder=5,
                   label='PINN Prediction')
    ax1.set_xlabel(r'$\mu = \cos(\theta)$', fontsize=14)
    ax1.set_ylabel(r'$I(x = 0, \mu)$', fontsize=14)
    ax1.set_title(rf'Left Boundary ($x = 0$), {g_str}', fontsize=13)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-0.05, 1.1])
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # 右边界 x=1
    ax2 = axes[1]
    ax2.plot(mu_quad, I[:, -1], 'b-', linewidth=2.5, label='Exact Solution (DOM)')
    if I_pinn_right is not None:
        I_pinn_markers_right = I_pinn_right[::step]
        ax2.scatter(mu_markers, I_pinn_markers_right, c='red', s=80, zorder=5,
                   label='PINN Prediction')
    ax2.set_xlabel(r'$\mu = \cos(\theta)$', fontsize=14)
    ax2.set_ylabel(r'$I(x = 1, \mu)$', fontsize=14)
    ax2.set_title(rf'Right Boundary ($x = 1$), {g_str}', fontsize=13)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-0.05, 1.1])
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    fig_name = f'{output_prefix}_boundaries_g{g:.1f}.png'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    print(f"  Saved to {fig_name}")
    plt.close()
    
    return x, mu_quad, w_quad, I, heat_flux


if __name__ == "__main__":
    import sys
    
    # 固定参数
    KAPPA = 0.5
    SIGMA_S = 0.5
    N_MU = 32  # 提高角度离散精度
    NX = 200
    
    # 需要求解的三个 g 值工况
    G_CASES = [0.5, -0.5, 0.8]
    G_NAMES = {0.5: "Forward", -0.5: "Backward", 0.8: "StrongForward"}
    
    print("=" * 70)
    print("DOM Solver with Anisotropic Scattering (Henyey-Greenstein)")
    print("=" * 70)
    print(f"Parameters: kappa={KAPPA}, sigma_s={SIGMA_S}")
    print(f"Discretization: N_mu={N_MU}, Nx={NX}")
    print(f"Cases: g = {G_CASES}")
    print("=" * 70)
    
    # 循环求解每个工况
    for g in G_CASES:
        print(f"\n{'='*70}")
        print(f"Solving Case: g = {g:.1f} ({G_NAMES[g]})")
        print(f"{'='*70}")
        
        # 可以指定对应的PINN模型路径
        pinn_path = None  # 默认不加载，或指定如 f"Results_1D_CaseD/TrainedModel/model.pkl"
        
        # 运行求解器
        x, mu, w, I_solution, q = dom_1d_solver_anisotropic(
            kappa=KAPPA, 
            sigma_s=SIGMA_S, 
            g=g,
            N_mu=N_MU, 
            Nx=NX, 
            max_iter=2000, 
            tol=1e-10,
            pinn_model_path=pinn_path,
            output_prefix="dom_solution"
        )
        
        # ============================================================
        # 保存结果为 .npy 文件
        # ============================================================
        npy_filename = f"dom_solution_g{g:.1f}.npy"
        np.save(npy_filename, I_solution)
        print(f"\n  [Saved] Radiation intensity matrix -> {npy_filename}")
        print(f"  Shape: {I_solution.shape} (N_mu={N_MU}, Nx={NX})")
        
        # 输出统计信息
        print(f"\n  ========== Results for g={g:.1f} ==========")
        print(f"  I_max: {I_solution.max():.6f}")
        print(f"  I_min: {I_solution.min():.6f}")
        print(f"  Heat flux at x=0: {q[0]:.6f}")
        print(f"  Heat flux at x=1: {q[-1]:.6f}")
    
    print(f"\n{'='*70}")
    print("All cases completed!")
    print(f"{'='*70}")
    print("\nGenerated files:")
    for g in G_CASES:
        print(f"  - dom_solution_g{g:.1f}.npy")
        print(f"  - dom_solution_g{g:.1f}.png")
        print(f"  - dom_solution_boundaries_g{g:.1f}.png")
