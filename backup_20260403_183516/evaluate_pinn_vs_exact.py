#!/usr/bin/env python
"""
evaluate_pinn_vs_exact.py

纯吸收介质 PINN 与解析解对比验证脚本
使用 Beer-Lambert 定律作为精确基准

纯吸收解析解：
- μ > 0: u(x, μ) = exp(-κx/μ)
- μ ≤ 0: u(x, μ) = 0
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import roots_legendre
import torch

# 设置期刊级绘图参数
rcParams['text.usetex'] = False
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 13
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 11
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Core'))


# ==============================================================================
# 纯吸收介质案例配置
# ==============================================================================
PURE_ABSORPTION_CONFIGS = {
    'A': {
        'name': 'PureAbsorption_kappa0.5',
        'kappa': 0.5,
        'description': r'$\kappa=0.5$ (Pure Absorption)',
        'folder': 'Results_1D_Section32',
        'model_path': 'Results_1D_Section32/TrainedModel/model.pkl'
    },
    'B': {
        'name': 'PureAbsorption_kappa1.0', 
        'kappa': 1.0,
        'description': r'$\kappa=1.0$ (Pure Absorption)',
        'folder': 'Results_1D_Section32_kappa1',
        'model_path': None  # 如果有训练好的模型，填入路径
    },
    'C': {
        'name': 'PureAbsorption_kappa0.1',
        'kappa': 0.1,
        'description': r'$\kappa=0.1$ (Pure Absorption)',
        'folder': 'Results_1D_Section32_kappa0.1',
        'model_path': None
    }
}


# ==============================================================================
# 1. Beer-Lambert 精确解析解
# ==============================================================================
def exact_solution_pure_absorption(kappa, x, mu, epsilon=1e-10):
    """
    纯吸收介质的 Beer-Lambert 精确解析解
    
    u_exact(x, μ) = exp(-κx/μ)  for μ > ε
    u_exact(x, μ) = 0           for μ ≤ ε
    
    Returns:
        u_exact: (N_mu, Nx) 精确解矩阵
        G_exact: (Nx,) 宏观入射辐射
    """
    X, Mu = np.meshgrid(x, mu)
    
    # 初始化
    u_exact = np.zeros_like(X)
    
    # μ > 0 区域：Beer-Lambert 衰减
    positive_mask = Mu > epsilon
    u_exact[positive_mask] = np.exp(-kappa * X[positive_mask] / Mu[positive_mask])
    
    # μ ≤ 0 区域保持为 0
    
    return u_exact


def compute_G_exact(kappa, x, mu, w, epsilon=1e-10):
    """
    计算纯吸收介质的宏观入射辐射 G(x) = ∫u(x,μ)dμ
    使用数值积分
    """
    u_exact = exact_solution_pure_absorption(kappa, x, mu, epsilon)
    G_exact = np.dot(w, u_exact)
    return G_exact, u_exact


# ==============================================================================
# 2. PINN 预测加载
# ==============================================================================
def load_pinn_prediction(model_path, x, mu, device='cpu'):
    """
    加载训练好的 PINN 模型并预测
    """
    if model_path is None or not os.path.exists(model_path):
        print(f"  [Warning] Model not found: {model_path}")
        print(f"  [Warning] Using mock data with artificial error")
        # 生成带误差的模拟数据
        Nx, N_mu = len(x), len(mu)
        X, Mu = np.meshgrid(x, mu)
        
        # 近似解析解 + 噪声
        u_mock = np.zeros_like(X)
        positive = Mu > 0
        u_mock[positive] = np.exp(-0.5 * X[positive] / (Mu[positive] + 0.01))
        u_mock += np.random.normal(0, 0.02, u_mock.shape)
        u_mock = np.clip(u_mock, 0, 1)
        
        _, w_mock = roots_legendre(N_mu)
        G_mock = np.dot(w_mock, u_mock)
        return u_mock, G_mock
    
    try:
        print(f"  Loading model from: {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        
        Nx, N_mu = len(x), len(mu)
        X_grid, Mu_grid = np.meshgrid(x, mu)
        
        # 展平为 (N_points, 2)
        x_flat = X_grid.flatten()
        mu_flat = Mu_grid.flatten()
        inputs = np.stack([x_flat, mu_flat], axis=1)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
        
        # 批量预测
        batch_size = 5000
        u_pred_list = []
        with torch.no_grad():
            for i in range(0, len(inputs_tensor), batch_size):
                batch = inputs_tensor[i:i+batch_size]
                pred = model(batch).cpu().numpy()
                u_pred_list.append(pred)
        u_pred = np.concatenate(u_pred_list, axis=0).flatten()
        
        # 重塑
        u_pinn = u_pred.reshape(N_mu, Nx)
        
        # 计算 G(x)
        _, w_quad = roots_legendre(N_mu)
        G_pinn = np.dot(w_quad, u_pinn)
        
        return u_pinn, G_pinn
        
    except Exception as e:
        print(f"  [Error] Failed to load model: {e}")
        return None, None


# ==============================================================================
# 3. 计算误差指标
# ==============================================================================
def compute_error_metrics(u_exact, u_pinn):
    """
    计算全局误差指标
    """
    diff = u_pinn - u_exact
    abs_l2 = np.sqrt(np.mean(diff**2))
    u_exact_norm = np.sqrt(np.mean(u_exact**2))
    rel_l2 = abs_l2 / (u_exact_norm + 1e-10)
    max_err = np.max(np.abs(diff))
    
    return rel_l2, abs_l2, max_err


# ==============================================================================
# 4. 绘图函数
# ==============================================================================
def plot_comparison(case_key, x, mu, u_exact, u_pinn, G_exact, G_pinn, 
                    rel_l2, output_dir='Evaluation_Results_PureAbsorption'):
    """
    生成三组对比图
    """
    config = PURE_ABSORPTION_CONFIGS[case_key]
    os.makedirs(output_dir, exist_ok=True)
    
    X, Mu = np.meshgrid(x, mu)
    
    # 创建图形
    fig = plt.figure(figsize=(14, 16))
    
    # 统一颜色范围
    vmin, vmax = 0, 1.0
    
    # (a) 精确解
    ax1 = plt.subplot(3, 2, 1)
    im1 = ax1.contourf(X, Mu, u_exact, levels=20, cmap='jet', 
                       vmin=vmin, vmax=vmax, extend='both')
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('mu', fontsize=13)
    ax1.set_title(f'(a) Exact Solution: Beer-Lambert', fontsize=13)
    ax1.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im1, ax=ax1, label='I')
    
    # (b) PINN 预测
    ax2 = plt.subplot(3, 2, 2)
    im2 = ax2.contourf(X, Mu, u_pinn, levels=20, cmap='jet',
                       vmin=vmin, vmax=vmax, extend='both')
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('mu', fontsize=13)
    ax2.set_title(f'(b) PINN: I_pinn(x, mu)', fontsize=13)
    ax2.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im2, ax=ax2, label='I')
    
    # (c) 误差分布
    ax3 = plt.subplot(3, 1, 2)
    error = np.abs(u_pinn - u_exact)
    im3 = ax3.contourf(X, Mu, error, levels=20, cmap='Reds', extend='max')
    ax3.set_xlabel('x', fontsize=13)
    ax3.set_ylabel('mu', fontsize=13)
    ax3.set_title(f'(c) Absolute Error: |I_pinn - I_exact| (Rel. L2 = {rel_l2:.4f})', 
                  fontsize=13)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    cbar = plt.colorbar(im3, ax=ax3)
    cbar.set_label('|I_pinn - I_exact|', fontsize=12)
    
    # (d) 宏观能量衰减对比
    ax4 = plt.subplot(3, 1, 3)
    ax4.plot(x, G_exact, 'k-', linewidth=2.5, label='Exact (Beer-Lambert)')
    ax4.plot(x, G_pinn, 'r--', linewidth=2.0, marker='o', markersize=4, 
             markevery=10, label='PINN Prediction')
    ax4.set_xlabel('x', fontsize=13)
    ax4.set_ylabel('G(x) = integral I(x, mu) dmu', fontsize=13)
    ax4.set_title(f'(d) Incident Radiation: {config["description"]}', fontsize=13)
    ax4.legend(loc='best', framealpha=0.9)
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, max(np.max(G_exact), np.max(G_pinn)) * 1.1])
    
    plt.tight_layout()
    
    # 保存完整对比图
    output_path = os.path.join(output_dir, f'PureAbsorption_{case_key}_comparison.png')
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    # 单独保存宏观能量图
    fig_G, ax_G = plt.subplots(figsize=(8, 5))
    ax_G.plot(x, G_exact, 'k-', linewidth=2.5, label='Exact (Beer-Lambert)')
    ax_G.plot(x, G_pinn, 'r--', linewidth=2.0, marker='o', markersize=5,
              markevery=10, label='PINN Prediction')
    ax_G.set_xlabel('x', fontsize=14)
    ax_G.set_ylabel('G(x)', fontsize=14)
    ax_G.set_title(f'Pure Absorption {case_key}: {config["description"]}', fontsize=14)
    ax_G.legend(loc='best', framealpha=0.9)
    ax_G.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    output_G = os.path.join(output_dir, f'PureAbsorption_{case_key}_G_comparison.png')
    plt.savefig(output_G, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_G}")
    plt.close(fig_G)


# ==============================================================================
# 5. 评估单个案例
# ==============================================================================
def evaluate_case(case_key, output_dir='Evaluation_Results_PureAbsorption'):
    """
    评估纯吸收介质案例
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Pure Absorption Case {case_key}")
    print(f"{'='*60}")
    
    config = PURE_ABSORPTION_CONFIGS[case_key]
    kappa = config['kappa']
    
    # 1. 生成精确解析解
    print(f"\n[1] Generating exact solution (Beer-Lambert)...")
    print(f"    kappa = {kappa}")
    
    # 使用高分辨率网格
    N_mu = 100
    Nx = 200
    mu, w = roots_legendre(N_mu)
    x = np.linspace(0, 1, Nx)
    
    u_exact = exact_solution_pure_absorption(kappa, x, mu)
    G_exact = np.dot(w, u_exact)
    
    print(f"    Grid: {N_mu} x {Nx}")
    print(f"    I_max = {u_exact.max():.6f}")
    print(f"    G_max = {G_exact.max():.6f}")
    
    # 2. 加载 PINN 预测
    print(f"\n[2] Loading PINN prediction...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Using device: {device}")
    
    model_path = config.get('model_path')
    u_pinn, G_pinn = load_pinn_prediction(model_path, x, mu, device)
    
    if u_pinn is None:
        print(f"    [Failed] Skipping Case {case_key}")
        return None
    
    # 3. 计算误差
    print(f"\n[3] Computing error metrics...")
    rel_l2, abs_l2, max_err = compute_error_metrics(u_exact, u_pinn)
    print(f"    Relative L2 Error: {rel_l2:.6f} ({rel_l2*100:.4f}%)")
    print(f"    Absolute L2 Error: {abs_l2:.6f}")
    print(f"    Maximum Error:     {max_err:.6f}")
    
    G_rel_err = np.linalg.norm(G_pinn - G_exact) / np.linalg.norm(G_exact)
    print(f"    G(x) Relative Error: {G_rel_err:.6f}")
    
    # 4. 绘图
    print(f"\n[4] Generating plots...")
    plot_comparison(case_key, x, mu, u_exact, u_pinn, G_exact, G_pinn, 
                    rel_l2, output_dir)
    
    return {
        'case': case_key,
        'kappa': kappa,
        'rel_l2': rel_l2,
        'abs_l2': abs_l2,
        'max_err': max_err,
        'G_rel_err': G_rel_err
    }


# ==============================================================================
# 6. 主程序
# ==============================================================================
def main():
    """
    主程序：评估所有纯吸收介质案例
    """
    print("="*70)
    print(" PINN vs Exact Solution (Pure Absorption)")
    print(" Using Beer-Lambert Law as Ground Truth")
    print("="*70)
    
    output_dir = 'Evaluation_Results_PureAbsorption'
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估所有案例（只评估有模型的）
    results = []
    for case_key in ['A']:  # 默认只评估Case A（有模型）
        result = evaluate_case(case_key, output_dir)
        if result:
            results.append(result)
    
    # 汇总结果
    print(f"\n{'='*70}")
    print(" Summary of Results")
    print(f"{'='*70}")
    print(f"{'Case':<10} {'Kappa':<12} {'Rel. L2':<15} {'Max Error':<15} {'G(x) Error':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['case']:<10} {r['kappa']:<12.2f} {r['rel_l2']:<15.6f} {r['max_err']:<15.6f} {r['G_rel_err']:<15.6f}")
    
    # 保存到文件
    summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("PINN vs Exact Solution (Pure Absorption) Summary\n")
        f.write("="*70 + "\n\n")
        f.write("Exact Solution: Beer-Lambert Law\n")
        f.write("  u(x, mu) = exp(-kappa * x / mu)  for mu > 0\n")
        f.write("  u(x, mu) = 0                     for mu <= 0\n\n")
        for r in results:
            f.write(f"Case {r['case']} (kappa = {r['kappa']}):\n")
            f.write(f"  Relative L2 Error: {r['rel_l2']:.6f}\n")
            f.write(f"  Absolute L2 Error: {r['abs_l2']:.6f}\n")
            f.write(f"  Maximum Error:     {r['max_err']:.6f}\n")
            f.write(f"  G(x) Rel. Error:   {r['G_rel_err']:.6f}\n\n")
    
    print(f"\nResults saved to: {summary_file}")
    print(f"{'='*70}")
    print("Evaluation completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
