#!/usr/bin/env python
"""
evaluate_anisotropic.py - 各向异性散射 PINN vs DOM 综合评估脚本

评估 Cases D/E/F（各向异性散射）：
    - Case D: g = 0.5 (前向散射)
    - Case E: g = -0.5 (后向散射)
    - Case F: g = 0.8 (强前向散射)

输出：
    - Anisotropic_CaseD_Eval.png 等 (组合图)
    - Anisotropic_CaseD_G_comparison.png 等 (单独的G(x)对比图)
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
# 绘图设置 (与 evaluate_pinn_vs_exact.py 保持一致)
# ==============================================================================
rcParams['text.usetex'] = False
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 13
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 11
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11

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
        'description': r'Forward Scattering (g = 0.5)',
        'g': 0.5,
        'kappa': 0.5,
        'sigma_s': 0.5,
        'folder': os.path.join(PROJECT_ROOT, 'Results_1D_CaseD'),
        'dom_file': os.path.join(PROJECT_ROOT, 'dom_solution_g0.5.npy')
    },
    'E': {
        'name': 'CaseE_Backward',
        'description': r'Backward Scattering (g = -0.5)',
        'g': -0.5,
        'kappa': 0.5,
        'sigma_s': 0.5,
        'folder': os.path.join(PROJECT_ROOT, 'Results_1D_CaseE'),
        'dom_file': os.path.join(PROJECT_ROOT, 'dom_solution_g-0.5.npy')
    },
    'F': {
        'name': 'CaseF_StrongForward',
        'description': r'Strong Forward Scattering (g = 0.8)',
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
# 4. 绘图函数 (与 evaluate_pinn_vs_exact.py 风格一致)
# ==============================================================================
def plot_comparison(case_key, x, mu, u_dom, u_pinn, G_dom, G_pinn,
                    rel_l2, output_dir='Evaluation_Results_Anisotropic'):
    """
    生成对比图 (与 evaluate_pinn_vs_exact.py 风格一致)
    
    子图布局 (3行2列):
        [左上] (a) DOM 解
        [右上] (b) PINN 解
        [中, 跨两列] (c) 绝对误差场
        [下, 跨两列] (d) G(x) 对比曲线
    """
    config = ANISOTROPIC_CASE_CONFIGS[case_key]
    os.makedirs(output_dir, exist_ok=True)
    
    X, Mu = np.meshgrid(x, mu)
    
    # 统一色标范围
    vmin, vmax = 0, max(np.max(u_dom), np.max(u_pinn)) * 1.05
    
    # 创建图形 (3行2列，与 evaluate_pinn_vs_exact.py 一致)
    fig = plt.figure(figsize=(14, 16))
    
    # -------------------------------------------------------------------------
    # 子图 (a): DOM 解
    # -------------------------------------------------------------------------
    ax1 = plt.subplot(3, 2, 1)
    im1 = ax1.contourf(X, Mu, u_dom, levels=20, cmap='jet',
                       vmin=vmin, vmax=vmax, extend='both')
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('mu', fontsize=13)
    ax1.set_title('(a) DOM Solution', fontsize=13)
    ax1.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im1, ax=ax1, label='I')
    
    # -------------------------------------------------------------------------
    # 子图 (b): PINN 解
    # -------------------------------------------------------------------------
    ax2 = plt.subplot(3, 2, 2)
    im2 = ax2.contourf(X, Mu, u_pinn, levels=20, cmap='jet',
                       vmin=vmin, vmax=vmax, extend='both')
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('mu', fontsize=13)
    ax2.set_title('(b) PINN: I_pinn(x, mu)', fontsize=13)
    ax2.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im2, ax=ax2, label='I')
    
    # -------------------------------------------------------------------------
    # 子图 (c): 绝对误差场 (跨两列)
    # -------------------------------------------------------------------------
    ax3 = plt.subplot(3, 1, 2)
    error = np.abs(u_pinn - u_dom)
    im3 = ax3.contourf(X, Mu, error, levels=20, cmap='Reds', extend='max')
    ax3.set_xlabel('x', fontsize=13)
    ax3.set_ylabel('mu', fontsize=13)
    ax3.set_title(f'(c) Absolute Error: |I_pinn - I_dom| (Rel. L2 = {rel_l2:.4f})',
                  fontsize=13)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    cbar = plt.colorbar(im3, ax=ax3)
    cbar.set_label('|I_pinn - I_dom|', fontsize=12)
    
    # -------------------------------------------------------------------------
    # 子图 (d): G(x) 对比 (跨两列)
    # -------------------------------------------------------------------------
    ax4 = plt.subplot(3, 1, 3)
    ax4.plot(x, G_dom, 'k-', linewidth=2.5, label='DOM (Ground Truth)')
    ax4.plot(x, G_pinn, 'r--', linewidth=2.0, marker='o', markersize=4,
             markevery=10, label='PINN Prediction')
    ax4.set_xlabel('x', fontsize=13)
    ax4.set_ylabel('G(x) = integral I(x, mu) dmu', fontsize=13)
    ax4.set_title(f'(d) Incident Radiation: {config["description"]}', fontsize=13)
    ax4.legend(loc='best', framealpha=0.9)
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, max(np.max(G_dom), np.max(G_pinn)) * 1.1])
    
    plt.tight_layout()
    
    # 保存完整对比图
    output_path = os.path.join(output_dir, f'Anisotropic_Case{case_key}_comparison.png')
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    # -------------------------------------------------------------------------
    # 单独保存 G(x) 对比图 (与 evaluate_pinn_vs_exact.py 一致)
    # -------------------------------------------------------------------------
    fig_G, ax_G = plt.subplots(figsize=(8, 5))
    ax_G.plot(x, G_dom, 'k-', linewidth=2.5, label='DOM (Ground Truth)')
    ax_G.plot(x, G_pinn, 'r--', linewidth=2.0, marker='o', markersize=5,
              markevery=10, label='PINN Prediction')
    ax_G.set_xlabel('x', fontsize=14)
    ax_G.set_ylabel('G(x)', fontsize=14)
    ax_G.set_title(f'Anisotropic {case_key}: {config["description"]}', fontsize=14)
    ax_G.legend(loc='best', framealpha=0.9)
    ax_G.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    output_G = os.path.join(output_dir, f'Anisotropic_Case{case_key}_G_comparison.png')
    plt.savefig(output_G, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_G}")
    plt.close(fig_G)


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
    
    # 4. 生成对比图
    print(f"\n[4] Generating comparison plots...")
    plot_comparison(case_key, x, mu, u_dom, u_pinn, G_dom, G_pinn,
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
