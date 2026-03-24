#!/usr/bin/env python
"""
evaluate_anisotropic.py - 各向异性散射 PINN vs DOM 综合评估脚本

评估 Cases D/E/F（各向异性散射）：
    - Case D: g = 0.5 (前向散射)
    - Case E: g = -0.5 (后向散射)
    - Case F: g = 0.8 (强前向散射)

输出：
    - Anisotropic_CaseD_Eval.png 等 (1x3 组合图)
    - evaluation_anisotropic_summary.txt (误差统计)
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

# ==============================================================================
# 学术级绘图设置 (400 DPI, Serif 字体, LaTeX 标签)
# ==============================================================================
rcParams['text.usetex'] = True  # 启用 LaTeX
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern']
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['figure.dpi'] = 400

# 添加项目根目录和 Core 目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
CORE_PATH = os.path.join(PROJECT_ROOT, 'Core')
if CORE_PATH not in sys.path:
    sys.path.insert(0, CORE_PATH)


# ==============================================================================
# 各向异性散射案例配置
# ==============================================================================
ANISOTROPIC_CASE_CONFIGS = {
    'D': {
        'name': 'CaseD_Forward',
        'description': r'Forward Scattering ($g = 0.5$)',
        'g': 0.5,
        'kappa': 0.5,
        'sigma_s': 0.5,
        'folder': os.path.join(PROJECT_ROOT, 'Results_1D_CaseD'),
        'dom_file': os.path.join(PROJECT_ROOT, 'dom_solution_g0.5.npy')
    },
    'E': {
        'name': 'CaseE_Backward',
        'description': r'Backward Scattering ($g = -0.5$)',
        'g': -0.5,
        'kappa': 0.5,
        'sigma_s': 0.5,
        'folder': os.path.join(PROJECT_ROOT, 'Results_1D_CaseE'),
        'dom_file': os.path.join(PROJECT_ROOT, 'dom_solution_g-0.5.npy')
    },
    'F': {
        'name': 'CaseF_StrongForward',
        'description': r'Strong Forward Scattering ($g = 0.8$)',
        'g': 0.8,
        'kappa': 0.5,
        'sigma_s': 0.5,
        'folder': os.path.join(PROJECT_ROOT, 'Results_1D_CaseF'),
        'dom_file': os.path.join(PROJECT_ROOT, 'dom_solution_g0.8.npy')
    }
}


# ==============================================================================
# 1. 加载 DOM 结果 (从 .npy 文件)
# ==============================================================================
def load_dom_solution(case_key, N_mu=100):
    """
    加载预计算的 DOM 结果
    
    Returns:
        x: 空间网格 (Nx,)
        mu: 角度网格 (N_mu,)
        w: 角度权重 (N_mu,)
        u_dom: 辐射强度矩阵 (N_mu, Nx)
        G_dom: 宏观入射辐射 (Nx,)
    """
    config = ANISOTROPIC_CASE_CONFIGS[case_key]
    dom_file = config['dom_file']
    
    if not os.path.exists(dom_file):
        print(f"  [Warning] DOM file not found: {dom_file}")
        print(f"  [Warning] Please run dom_1d_solver_HG.py first")
        return None, None, None, None, None
    
    # 加载 DOM 辐射强度矩阵
    u_dom = np.load(dom_file)  # shape: (N_mu, Nx)
    N_mu_loaded, Nx = u_dom.shape
    
    # 重建网格
    x = np.linspace(0, 1, Nx)
    mu, w = roots_legendre(N_mu_loaded)
    
    # 计算宏观入射辐射 G(x) = integral_{-1}^{1} I(x, mu) dmu
    G_dom = np.dot(w, u_dom)
    
    print(f"  Loaded DOM solution: {N_mu_loaded} x {Nx} grid")
    print(f"  g = {config['g']}, kappa = {config['kappa']}, sigma_s = {config['sigma_s']}")
    
    return x, mu, w, u_dom, G_dom


# ==============================================================================
# 2. 加载 PINN 预测
# ==============================================================================
def load_pinn_prediction(case_key, x, mu, device='cpu'):
    """
    加载训练好的 PINN 模型并预测
    
    Returns:
        u_pinn: 辐射强度矩阵 (N_mu, Nx)
        G_pinn: 宏观入射辐射 (Nx,)
    """
    config = ANISOTROPIC_CASE_CONFIGS[case_key]
    model_path = os.path.join(config['folder'], 'TrainedModel', 'model.pkl')
    
    if not os.path.exists(model_path):
        print(f"  [Warning] Model not found: {model_path}")
        return None, None
    
    try:
        print(f"  Loading PINN model: {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        
        # 创建网格数据
        Nx, N_mu = len(x), len(mu)
        X_grid, Mu_grid = np.meshgrid(x, mu)
        
        # 展平为 (N_points, 2) 格式
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
        
        # 重塑为 (N_mu, Nx)
        u_pinn = u_pred.reshape(N_mu, Nx)
        
        # 计算 G(x)
        _, w_quad = roots_legendre(N_mu)
        G_pinn = np.dot(w_quad, u_pinn)
        
        return u_pinn, G_pinn
        
    except Exception as e:
        print(f"  [Error] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ==============================================================================
# 3. 计算误差指标
# ==============================================================================
def compute_error_metrics(u_dom, u_pinn):
    """
    计算全局相对 L2 误差
    
    Relative L2 Error = ||u_pinn - u_dom||_2 / ||u_dom||_2
    """
    diff = u_pinn - u_dom
    abs_l2 = np.sqrt(np.mean(diff**2))
    u_dom_norm = np.sqrt(np.mean(u_dom**2))
    rel_l2 = abs_l2 / (u_dom_norm + 1e-10)
    max_err = np.max(np.abs(diff))
    
    return rel_l2, abs_l2, max_err


# ==============================================================================
# 4. 学术级绘图函数 (2x2 组合图，避免拥挤)
# ==============================================================================
def plot_comparison_2x2(case_key, x, mu, u_dom, u_pinn, G_dom, G_pinn,
                        rel_l2, output_dir='Evaluation_Results_Anisotropic'):
    """
    生成 2x2 组合评估图
    
    子图布局:
        [左上] (a) DOM 解 I_dom(x, mu)
        [右上] (b) PINN 解 I_pinn(x, mu)
        [左下] (c) 绝对误差场 |u_pinn - u_dom|
        [右下] (d) G(x) 对比曲线
    """
    config = ANISOTROPIC_CASE_CONFIGS[case_key]
    os.makedirs(output_dir, exist_ok=True)
    
    X, Mu = np.meshgrid(x, mu)
    error = np.abs(u_pinn - u_dom)
    
    # 统一色标范围
    vmin, vmax = 0, max(np.max(u_dom), np.max(u_pinn)) * 1.05
    
    # 创建图形 (2行2列)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # -------------------------------------------------------------------------
    # 子图 (a): DOM 解
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    im_dom = ax_a.contourf(X, Mu, u_dom, levels=20, cmap='jet',
                           vmin=vmin, vmax=vmax, extend='both')
    ax_a.set_xlabel(r'$x$', fontsize=12)
    ax_a.set_ylabel(r'$\mu = \cos(\theta)$', fontsize=12)
    ax_a.set_title(r'(a) DOM: $I_{\mathrm{dom}}(x, \mu)$', fontsize=12, pad=5)
    ax_a.axhline(y=0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_a.set_xticks(np.linspace(0, 1, 5))
    ax_a.set_yticks([-1, -0.5, 0, 0.5, 1])
    
    # colorbar
    cbar_a = plt.colorbar(im_dom, ax=ax_a, pad=0.02)
    cbar_a.set_label(r'$I(x, \mu)$', fontsize=10, rotation=270, labelpad=15)
    
    # -------------------------------------------------------------------------
    # 子图 (b): PINN 解
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    im_pinn = ax_b.contourf(X, Mu, u_pinn, levels=20, cmap='jet',
                            vmin=vmin, vmax=vmax, extend='both')
    ax_b.set_xlabel(r'$x$', fontsize=12)
    ax_b.set_ylabel(r'$\mu = \cos(\theta)$', fontsize=12)
    ax_b.set_title(r'(b) PINN: $I_{\mathrm{pinn}}(x, \mu)$', fontsize=12, pad=5)
    ax_b.axhline(y=0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_b.set_xticks(np.linspace(0, 1, 5))
    ax_b.set_yticks([-1, -0.5, 0, 0.5, 1])
    
    # colorbar (共享色标，但单独显示)
    cbar_b = plt.colorbar(im_pinn, ax=ax_b, pad=0.02)
    cbar_b.set_label(r'$I(x, \mu)$', fontsize=10, rotation=270, labelpad=15)
    
    # -------------------------------------------------------------------------
    # 子图 (c): 绝对误差场
    # -------------------------------------------------------------------------
    ax_c = axes[1, 0]
    im_err = ax_c.contourf(X, Mu, error, levels=20, cmap='coolwarm', extend='max')
    ax_c.set_xlabel(r'$x$', fontsize=12)
    ax_c.set_ylabel(r'$\mu = \cos(\theta)$', fontsize=12)
    ax_c.set_title(r'(c) Absolute Error: $|I_{\mathrm{pinn}} - I_{\mathrm{dom}}|$' + 
                   f' (Rel. L$_2$ = {rel_l2:.4f})', fontsize=12, pad=5)
    ax_c.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_c.set_xticks(np.linspace(0, 1, 5))
    ax_c.set_yticks([-1, -0.5, 0, 0.5, 1])
    
    # colorbar
    cbar_c = plt.colorbar(im_err, ax=ax_c, pad=0.02)
    cbar_c.set_label(r'$|\Delta I|$', fontsize=10, rotation=270, labelpad=15)
    
    # -------------------------------------------------------------------------
    # 子图 (d): G(x) 对比
    # -------------------------------------------------------------------------
    ax_d = axes[1, 1]
    
    # DOM (黑色实线)
    ax_d.plot(x, G_dom, 'k-', linewidth=2.5, label=r'DOM (Ground Truth)')
    
    # PINN (红色散点)
    ax_d.scatter(x[::5], G_pinn[::5], c='red', s=40, marker='o',
                label=r'PINN Prediction', zorder=5)
    
    ax_d.set_xlabel(r'$x$', fontsize=12)
    ax_d.set_ylabel(r'$G(x) = \int_{-1}^{1} I(x, \mu) \, \mathrm{d}\mu$', fontsize=12)
    ax_d.set_title(r'(d) Incident Radiation $G(x)$ Comparison', fontsize=12, pad=5)
    ax_d.legend(loc='best', framealpha=0.95, edgecolor='gray')
    ax_d.grid(True, linestyle='--', alpha=0.4)
    ax_d.set_xlim([0, 1])
    ax_d.set_xticks(np.linspace(0, 1, 5))
    
    G_max = max(np.max(G_dom), np.max(G_pinn))
    ax_d.set_ylim([0, G_max * 1.15])
    
    # 添加案例信息文本
    textstr = config['description']
    ax_d.text(0.98, 0.97, textstr, transform=ax_d.transAxes,
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, edgecolor='gray'))
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    # 保存图片
    output_path = os.path.join(output_dir, f'Anisotropic_Case{case_key}_Eval.png')
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ==============================================================================
# 5. 评估单个案例
# ==============================================================================
def evaluate_case(case_key, output_dir='Evaluation_Results_Anisotropic'):
    """
    评估单个各向异性散射案例
    """
    config = ANISOTROPIC_CASE_CONFIGS[case_key]
    
    print(f"\n{'='*70}")
    print(f"Evaluating Case {case_key}: {config['name']}")
    print(f"{'='*70}")
    
    # 1. 加载 DOM 结果
    print(f"\n[1] Loading DOM solution...")
    x, mu, w, u_dom, G_dom = load_dom_solution(case_key)
    
    if u_dom is None:
        print(f"  [Failed] DOM solution not available. Skipping.")
        return None
    
    # 2. 加载 PINN 预测
    print(f"\n[2] Loading PINN prediction...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Using device: {device}")
    u_pinn, G_pinn = load_pinn_prediction(case_key, x, mu, device)
    
    if u_pinn is None:
        print(f"  [Failed] PINN model not available. Skipping.")
        return None
    
    # 3. 计算误差指标
    print(f"\n[3] Computing error metrics...")
    rel_l2, abs_l2, max_err = compute_error_metrics(u_dom, u_pinn)
    print(f"    Relative L2 Error: {rel_l2:.6f} ({rel_l2*100:.4f}%)")
    print(f"    Absolute L2 Error: {abs_l2:.6f}")
    print(f"    Maximum Error:     {max_err:.6f}")
    
    G_rel_err = np.linalg.norm(G_pinn - G_dom) / np.linalg.norm(G_dom)
    print(f"    G(x) Relative Error: {G_rel_err:.6f}")
    
    # 4. 生成 2x2 组合图
    print(f"\n[4] Generating 2x2 comparison plot...")
    plot_comparison_2x2(case_key, x, mu, u_dom, u_pinn, G_dom, G_pinn,
                        rel_l2, output_dir)
    
    return {
        'case': case_key,
        'g': config['g'],
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
    主程序：评估所有各向异性散射案例
    """
    print("="*70)
    print(" Anisotropic Scattering Evaluation: PINN vs DOM")
    print(" Cases D/E/F: Henyey-Greenstein Phase Function")
    print("="*70)
    
    output_dir = 'Evaluation_Results_Anisotropic'
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估所有案例
    results = []
    for case_key in ['D', 'E', 'F']:
        result = evaluate_case(case_key, output_dir)
        if result:
            results.append(result)
    
    if not results:
        print(f"\n{'='*70}")
        print("No cases were successfully evaluated!")
        print("Please ensure:")
        print("  1. DOM solutions exist (run dom_1d_solver_HG.py first)")
        print("  2. PINN models are trained (run train_1d_multicase_anisotropic.py)")
        print(f"{'='*70}")
        return
    
    # 汇总结果
    print(f"\n{'='*70}")
    print(" Summary of Results")
    print(f"{'='*70}")
    print(f"{'Case':<8} {'g':<10} {'Rel. L2 Error':<18} {'Max Error':<15} {'G(x) Error':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['case']:<8} {r['g']:<10.1f} {r['rel_l2']:<18.6f} {r['max_err']:<15.6f} {r['G_rel_err']:<15.6f}")
    
    # 保存到文件
    summary_file = os.path.join(output_dir, 'evaluation_anisotropic_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Anisotropic Scattering: PINN vs DOM Evaluation Summary\n")
        f.write("="*70 + "\n\n")
        f.write("Case D: Forward Scattering (g = 0.5)\n")
        f.write("Case E: Backward Scattering (g = -0.5)\n")
        f.write("Case F: Strong Forward Scattering (g = 0.8)\n\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Case':<8} {'g':<10} {'Rel. L2 Error':<18} {'Max Error':<15} {'G(x) Error':<15}\n")
        f.write("-"*70 + "\n")
        for r in results:
            f.write(f"{r['case']:<8} {r['g']:<10.1f} {r['rel_l2']:<18.6f} {r['max_err']:<15.6f} {r['G_rel_err']:<15.6f}\n")
        f.write("\n" + "="*70 + "\n")
        for r in results:
            f.write(f"\nCase {r['case']} (g = {r['g']:.1f}):\n")
            f.write(f"  Relative L2 Error: {r['rel_l2']:.6f} ({r['rel_l2']*100:.4f}%)\n")
            f.write(f"  Absolute L2 Error: {r['abs_l2']:.6f}\n")
            f.write(f"  Maximum Error:     {r['max_err']:.6f}\n")
            f.write(f"  G(x) Rel. Error:   {r['G_rel_err']:.6f}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"{'='*70}")
    print("Evaluation completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
