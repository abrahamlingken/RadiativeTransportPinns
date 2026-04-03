#!/usr/bin/env python
"""
validate_pure_absorption.py

一维稳态纯吸收介质辐射传输方程的精确解析解计算与 PINN 预测对比验证。

物理问题描述:
- 求解域: x ∈ [0, 1], μ ∈ [-1, 1]
- 吸收系数: kappa = 0.5 (纯吸收，无散射，无发射)
- 边界条件: 
    * 左边界入射 u(0, μ>0) = 1
    * 右边界无入射 u(1, μ<0) = 0

精确解析解 (Beer-Lambert 定律):
    u_exact(x, μ) = exp(-kappa * x / μ)  for μ > 0
    u_exact(x, μ) = 0                    for μ ≤ 0

作者: Lk
日期: 2026-03-19
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 设置 matplotlib 支持 LaTeX 渲染（可选，提升期刊级质量）
# 如果系统未安装 LaTeX，请注释掉以下行
try:
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
except:
    rcParams['text.usetex'] = False

rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['legend.fontsize'] = 12
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11

# ==============================================================================
# 物理参数配置
# ==============================================================================
KAPPA_DEFAULT = 0.5  # 默认吸收系数
MU_EPSILON = 1e-10   # μ 接近零时的保护阈值

# 网格分辨率
N_X = 500  # x 方向网格数
N_MU = 500  # μ 方向网格数


def compute_exact_solution(X, Mu, kappa=KAPPA_DEFAULT):
    """
    计算纯吸收介质的精确解析解 (Beer-Lambert 定律)。
    
    参数:
    ------
    X : np.ndarray
        空间坐标网格，形状为 (N_mu, N_x)
    Mu : np.ndarray
        方向余弦网格，形状为 (N_mu, N_x)
    kappa : float, optional
        吸收系数，默认为 KAPPA_DEFAULT
        
    返回:
    ------
    u_exact : np.ndarray
        精确解矩阵，与输入网格同形状
    """
    # 初始化解矩阵
    u_exact = np.zeros_like(X, dtype=np.float64)
    
    # 对于 μ > 0 的区域: u = exp(-kappa * x / μ)
    # 使用布尔掩码进行向量化计算
    positive_mu_mask = Mu > MU_EPSILON
    
    # Beer-Lambert 衰减公式
    u_exact[positive_mu_mask] = np.exp(
        -kappa * X[positive_mu_mask] / Mu[positive_mu_mask]
    )
    
    return u_exact


def generate_grid(n_x=N_X, n_mu=N_MU):
    """
    生成计算网格。
    
    参数:
    ------
    n_x : int
        x 方向网格数
    n_mu : int
        μ 方向网格数
        
    返回:
    ------
    X, Mu : np.ndarray
        网格坐标矩阵
    x, mu : np.ndarray
        一维坐标向量
    """
    x = np.linspace(0, 1, n_x)
    mu = np.linspace(-1, 1, n_mu)
    X, Mu = np.meshgrid(x, mu, indexing='xy')
    return X, Mu, x, mu


def load_pinn_prediction_mock(X_shape, Mu_shape, method='approximate'):
    """
    【Mock 接口】加载 PINN 预测结果。
    
    参数:
    ------
    X_shape : tuple
        期望的输出形状 (n_mu, n_x)
    Mu_shape : tuple
        期望的输出形状 (n_mu, n_x)
    method : str
        Mock 数据生成方法
        
    返回:
    ------
    u_pred : np.ndarray
        PINN 预测矩阵
    """
    X, Mu, _, _ = generate_grid(X_shape[1], X_shape[0])
    u_exact = compute_exact_solution(X, Mu, kappa=KAPPA_DEFAULT)
    
    if method == 'zeros':
        return np.zeros(X_shape)
    elif method == 'exact':
        return u_exact
    elif method == 'approximate':
        np.random.seed(42)
        noise = np.random.normal(0, 0.01 * u_exact.max(), X_shape)
        u_pred = u_exact + noise
        u_pred = np.clip(u_pred, 0, None)
        return u_pred
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_exact_contour(X, Mu, u_exact, save_path=None):
    """
    图1: 精确解的全局等高线图
    
    风格参考: net_sol.png
    - jet colormap
    - extend='both'
    - 颜色条标签旋转270度
    - figsize=(10, 8)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 使用 net_sol.png 风格: jet colormap + extend='both'
    levels = np.linspace(u_exact.min(), u_exact.max(), 21)
    im = ax.contourf(X, Mu, u_exact, levels=levels, cmap='jet', extend='both')
    
    # 标签和标题
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$\mu$', fontsize=14)
    ax.set_title(r'Exact Solution: $I(x, \mu)$', fontsize=16)
    
    # 刻度设置
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(-1, 1, 9))
    ax.tick_params(labelsize=12)
    
    # 颜色条 - 与 net_sol.png 一致
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$I(x, \mu)$', fontsize=14, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[图1已保存] {save_path}")
    
    return fig, ax


def plot_cross_sections(x, u_exact, mu_vector, save_path=None):
    """
    图2: 特定角度的空间衰减曲线
    
    风格参考: u0.png, u1.png
    - 蓝色实线
    - 红色圆点标记
    - 点状网格线
    - 左上角图例
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 选取的前向角度
    mu_selected = [1.0, 0.8, 0.5, 0.2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 标准颜色
    
    for i, mu_val in enumerate(mu_selected):
        idx = np.argmin(np.abs(mu_vector - mu_val))
        actual_mu = mu_vector[idx]
        u_slice = u_exact[idx, :]
        
        # 绘制曲线 - 参考 u0.png/u1.png 风格
        ax.plot(
            x, u_slice, 
            color=colors[i], 
            linewidth=2.5, 
            label=rf'$\mu = {actual_mu:.1f}$'
        )
    
    # 坐标轴设置
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$u(x, \mu)$', fontsize=14)
    ax.set_title(r'Spatial Attenuation at Different Forward Angles ($\mu > 0$)', fontsize=14)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(labelsize=11)
    
    # 图例 - 左上角，参考 u0.png
    ax.legend(loc='upper right', framealpha=0.9, fontsize=12)
    
    # 网格 - 点状线，参考 u0.png/u1.png
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # 添加公式注释
    formula_text = r'$u(x, \mu) = \exp(-\kappa x / \mu)$' + '\n' + rf'$\kappa = {KAPPA_DEFAULT}$'
    ax.text(0.65, 0.6, formula_text, fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[图2已保存] {save_path}")
    
    return fig, ax


def plot_boundary_comparison(x, mu_vector, u_exact, save_path=None):
    """
    图3: 边界对比图 (参考 u0.png, u1.png 风格)
    
    绘制 x=0 和 x=1 处的角度分布
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 获取边界值
    idx_x0 = 0  # x = 0
    idx_x1 = -1  # x = 1
    
    # x=0 处的边界值
    u_x0 = u_exact[:, idx_x0]
    # x=1 处的边界值
    u_x1 = u_exact[:, idx_x1]
    
    # 左图: x=0
    ax1 = axes[0]
    ax1.plot(mu_vector, u_x0, 'b-', linewidth=2.5, label='Exact Solution')
    
    # 标记理论参考点
    mu_ref = np.array([-0.9, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    u_ref = np.zeros_like(mu_ref)
    for i, m in enumerate(mu_ref):
        if m > 0:
            u_ref[i] = np.exp(-KAPPA_DEFAULT * 0 / m)  # = 1 for all m>0 at x=0
    ax1.scatter(mu_ref[mu_ref > 0], u_ref[mu_ref > 0], 
                c='red', s=80, zorder=5, label='Reference values')
    ax1.scatter(mu_ref[mu_ref == 0], u_ref[mu_ref == 0], 
                c='red', s=80, zorder=5)
    
    ax1.set_xlabel(r'$\mu = \cos(\theta)$', fontsize=14)
    ax1.set_ylabel(r'$u^-(x=0)$', fontsize=14)
    ax1.set_title(r'Boundary at $x = 0$', fontsize=14)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-0.05, 1.1])
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # 右图: x=1
    ax2 = axes[1]
    ax2.plot(mu_vector, u_x1, 'b-', linewidth=2.5, label='Exact Solution')
    
    # 标记理论参考点
    for i, m in enumerate(mu_ref):
        if m > 0:
            u_ref[i] = np.exp(-KAPPA_DEFAULT * 1 / m)
        else:
            u_ref[i] = 0
    ax2.scatter(mu_ref[mu_ref > 0.001], u_ref[mu_ref > 0.001], 
                c='red', s=80, zorder=5, label='Reference values')
    
    ax2.set_xlabel(r'$\mu = \cos(\theta)$', fontsize=14)
    ax2.set_ylabel(r'$u^+(x=1)$', fontsize=14)
    ax2.set_title(r'Boundary at $x = 1$', fontsize=14)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-0.05, 1.1])
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[图3已保存] {save_path}")
    
    return fig, axes


def compute_log_error(u_exact, u_pred, epsilon=1e-10):
    """
    计算对数尺度误差。
    """
    log_u_exact = np.log(u_exact + epsilon)
    log_u_pred = np.log(u_pred + epsilon)
    log_error = np.abs(log_u_exact - log_u_pred)
    return log_error


def plot_error_comparison(X, Mu, u_exact, u_pred, save_path=None):
    """
    图4: 误差分析三栏图
    
    风格统一使用 jet colormap 和 extend='both'
    """
    # 计算各种误差
    abs_error = np.abs(u_exact - u_pred)
    
    relative_error = np.zeros_like(abs_error)
    mask = u_exact > 1e-10
    relative_error[mask] = abs_error[mask] / u_exact[mask]
    
    log_error = compute_log_error(u_exact, u_pred, epsilon=1e-10)
    
    # 三栏图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ========== 子图1: 绝对误差 ==========
    ax1 = axes[0]
    levels_abs = np.linspace(0, np.percentile(abs_error, 99), 21)
    im1 = ax1.contourf(X, Mu, abs_error, levels=levels_abs, cmap='jet', extend='both')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label(r'$|u_{\mathrm{exact}} - u_{\mathrm{PINN}}|$', fontsize=12, rotation=270, labelpad=20)
    
    ax1.set_xlabel(r'$x$', fontsize=13)
    ax1.set_ylabel(r'$\mu$', fontsize=13)
    ax1.set_title('Absolute Error', fontsize=13)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([-1, 1])
    
    # ========== 子图2: 相对误差 (%) ==========
    ax2 = axes[1]
    rel_err_percent = relative_error * 100
    levels_rel = np.linspace(0, min(np.percentile(rel_err_percent[mask], 99), 100), 21)
    im2 = ax2.contourf(X, Mu, rel_err_percent, levels=levels_rel, cmap='jet', extend='both')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label(r'Relative Error (\%)', fontsize=12, rotation=270, labelpad=20)
    
    ax2.set_xlabel(r'$x$', fontsize=13)
    ax2.set_ylabel(r'$\mu$', fontsize=13)
    ax2.set_title('Relative Error', fontsize=13)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-1, 1])
    
    # ========== 子图3: 对数尺度误差 ==========
    ax3 = axes[2]
    log_err_max = min(np.percentile(log_error, 98), 5.0)
    levels_log = np.linspace(0, log_err_max, 21)
    im3 = ax3.contourf(X, Mu, log_error, levels=levels_log, cmap='jet', extend='both')
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label(r'$|\ln(u_{\mathrm{exact}} + \epsilon) - \ln(u_{\mathrm{PINN}} + \epsilon)|$', 
                    fontsize=11, rotation=270, labelpad=20)
    
    ax3.set_xlabel(r'$x$', fontsize=13)
    ax3.set_ylabel(r'$\mu$', fontsize=13)
    ax3.set_title('Log-scale Error (Improved)', fontsize=13)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([-1, 1])
    
    plt.tight_layout()
    
    # 打印统计信息
    print("\n" + "="*70)
    print("误差统计信息")
    print("="*70)
    print(f"【绝对误差】")
    print(f"  最大值 (L∞):   {abs_error.max():.6e}")
    print(f"  均值 (L1):     {abs_error.mean():.6e}")
    
    if mask.any():
        print(f"\n【相对误差】")
        print(f"  最大值:        {relative_error[mask].max()*100:.4f}%")
        print(f"  均值:          {relative_error[mask].mean()*100:.4f}%")
    
    print(f"\n【对数尺度误差】(ε = 1e-10)")
    print(f"  最大值:        {log_error.max():.4f}")
    print(f"  均值:          {log_error.mean():.4f}")
    print(f"  <0.1 的网格点: {(log_error < 0.1).sum()} ({(log_error < 0.1).mean()*100:.1f}%)")
    print("="*70)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[图4已保存] {save_path}")
    
    return fig, axes


def main():
    """主函数"""
    print("="*70)
    print("一维稳态纯吸收介质辐射传输方程 - 解析解验证")
    print("="*70)
    print(f"\n物理参数:")
    print(f"  - 吸收系数 kappa = {KAPPA_DEFAULT}")
    print(f"  - 求解域: x ∈ [0, 1], μ ∈ [-1, 1]")
    print(f"  - 网格分辨率: {N_X} × {N_MU}")
    
    # 生成网格
    print("\n[步骤 1/5] 生成计算网格...")
    X, Mu, x, mu = generate_grid(N_X, N_MU)
    
    print("[步骤 2/5] 计算精确解析解...")
    u_exact = compute_exact_solution(X, Mu, kappa=KAPPA_DEFAULT)
    print(f"  - 精确解范围: [{u_exact.min():.6f}, {u_exact.max():.6f}]")
    
    # 创建输出目录
    output_dir = "validation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[步骤 3/5] 生成可视化图表...")
    
    # 图1: 精确解等高线
    print("  - 绘制图1: 精确解全局等高线图...")
    plot_exact_contour(
        X, Mu, u_exact, 
        save_path=f"{output_dir}/01_exact_solution.png"
    )
    
    # 图2: 特定角度衰减曲线
    print("  - 绘制图2: 特定角度空间衰减曲线...")
    plot_cross_sections(
        x, u_exact, mu,
        save_path=f"{output_dir}/02_cross_sections.png"
    )
    
    # 图3: 边界对比 (参考 u0.png, u1.png 风格)
    print("  - 绘制图3: 边界对比图...")
    plot_boundary_comparison(
        x, mu, u_exact,
        save_path=f"{output_dir}/03_boundary_comparison.png"
    )
    
    # 图4: 误差分析
    print("  - 绘制图4: 误差分析图...")
    u_pred = load_pinn_prediction_mock(X.shape, Mu.shape, method='approximate')
    plot_error_comparison(
        X, Mu, u_exact, u_pred,
        save_path=f"{output_dir}/04_error_analysis.png"
    )
    
    print("\n" + "="*70)
    print("验证完成!")
    print(f"所有图表已保存到: ./{output_dir}/")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    main()
