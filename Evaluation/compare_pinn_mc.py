#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_pinn_mc.py - 对比PINN与Monte Carlo验证结果

功能：
1. 加载PINN训练好的模型
2. 计算并保存G场（积分辐射强度）
3. 与MC结果对比并计算误差
4. 生成对比图和误差分析报告

用法：
    python compare_pinn_mc.py --case 3D_B --mc-path Solvers/MC/MC3D_Results/MC_G_3D_B_CaseB_Isotropic.npz
    python compare_pinn_mc.py --case 3D_C --mc-path Solvers/MC/MC3D_Results/MC_G_3D_C_CaseC_Forward.npz
"""

import sys
import os
import json
import argparse
import numpy as np
import torch

# ========================================================================
# Path setup
# ========================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CORE_PATH = os.path.join(PROJECT_ROOT, 'Core')
if CORE_PATH not in sys.path:
    sys.path.insert(0, CORE_PATH)

from ModelClassTorch2 import Pinns


# ========================================================================
# Case configurations
# ========================================================================
CASE_CONFIGS = {
    '3D_A': {
        'name': 'Case3D_A_PureAbsorption',
        'kappa': 5.0,
        'sigma_s': 0.0,
        'g': 0.0,
        'folder': 'Results_3D_CaseA',
        'mc_expected': 'MC_G_3D_A_CaseA_PureAbsorption.npz'
    },
    '3D_B': {
        'name': 'Case3D_B_Isotropic',
        'kappa': 0.5,
        'sigma_s': 4.5,
        'g': 0.0,
        'folder': 'Results_3D_CaseB',
        'mc_expected': 'MC_G_3D_B_CaseB_Isotropic.npz'
    },
    '3D_C': {
        'name': 'Case3D_C_ForwardScattering',
        'kappa': 0.5,
        'sigma_s': 4.5,
        'g': 0.8,
        'folder': 'Results_3D_CaseC',
        'mc_expected': 'MC_G_3D_C_CaseC_Forward.npz'
    }
}


def load_pinn_model(case_key, device='cpu'):
    """加载PINN模型"""
    config = CASE_CONFIGS[case_key]
    folder_path = os.path.join(PROJECT_ROOT, config['folder'])
    model_path = os.path.join(folder_path, 'model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading PINN model from: {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    return model, folder_path


def compute_G_pinn(model, n_theta=16, n_phi=32, nx=51, ny=51, nz=51, device='cpu'):
    """
    计算PINN的G场（积分辐射强度）
    
    G(x,y,z) = ∫∫ I(x,y,z,θ,φ) sin(θ) dθ dφ
    
    使用梯形法则在方向角上积分
    """
    print(f"Computing G field on {nx}x{ny}x{nz} grid...")
    
    # 创建空间网格
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    
    # 创建方向角网格
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    d_theta = np.pi / (n_theta - 1)
    d_phi = 2 * np.pi / (n_phi - 1)
    
    # 创建所有网格点
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    G = np.zeros((nx, ny, nz))
    
    # 批次处理以减少内存使用
    batch_size = 1000
    n_batches = (coords.shape[0] + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i_batch in range(n_batches):
            if i_batch % 10 == 0:
                print(f"  Batch {i_batch+1}/{n_batches}")
            
            start = i_batch * batch_size
            end = min((i_batch + 1) * batch_size, coords.shape[0])
            batch_coords = coords[start:end]
            n_points = batch_coords.shape[0]
            
            # 对每个空间点，在所有方向上积分
            I_sum = np.zeros(n_points)
            
            for i_theta in range(n_theta):
                for i_phi in range(n_phi):
                    th = theta[i_theta]
                    ph = phi[i_phi]
                    
                    # 创建输入：[x, y, z, theta, phi]
                    inputs = np.zeros((n_points, 5))
                    inputs[:, :3] = batch_coords
                    inputs[:, 3] = th
                    inputs[:, 4] = ph
                    
                    # 预测
                    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)
                    I_pred = model(inputs_tensor).cpu().numpy().flatten()
                    
                    # 累加，带sin(theta)权重
                    I_sum += I_pred * np.sin(th) * d_theta * d_phi
            
            # 将结果放回G数组
            for i_pt, idx in enumerate(range(start, end)):
                ix = idx // (ny * nz)
                iy = (idx // nz) % ny
                iz = idx % nz
                G[ix, iy, iz] = I_sum[i_pt]
    
    return G, x, y, z


def load_mc_results(mc_path):
    """加载MC结果"""
    print(f"Loading MC results from: {mc_path}")
    data = np.load(mc_path)
    G_mc = data['G']
    x_mc = data['x']
    y_mc = data['y']
    z_mc = data['z']
    
    print(f"  MC grid: {G_mc.shape}")
    print(f"  MC G range: [{G_mc.min():.4f}, {G_mc.max():.4f}]")
    print(f"  MC G center: {G_mc[G_mc.shape[0]//2, G_mc.shape[1]//2, G_mc.shape[2]//2]:.4f}")
    
    return G_mc, x_mc, y_mc, z_mc


def compute_errors(G_pinn, G_mc, x_pinn, y_pinn, z_pinn, x_mc, y_mc, z_mc):
    """计算误差（处理不同网格尺寸）"""
    # 如果网格尺寸不同，对MC结果进行插值
    if G_pinn.shape != G_mc.shape:
        print(f"Grid size mismatch: PINN {G_pinn.shape} vs MC {G_mc.shape}")
        print("Interpolating MC results to PINN grid...")
        
        from scipy.interpolate import RegularGridInterpolator
        
        interp = RegularGridInterpolator(
            (x_mc, y_mc, z_mc),
            G_mc,
            bounds_error=False,
            fill_value=0
        )
        
        # 创建PINN网格的查询点
        Xp, Yp, Zp = np.meshgrid(x_pinn, y_pinn, z_pinn, indexing='ij')
        points = np.stack([Xp.ravel(), Yp.ravel(), Zp.ravel()], axis=1)
        
        G_mc_interp = interp(points).reshape(G_pinn.shape)
    else:
        G_mc_interp = G_mc
    
    # 计算各种误差指标
    diff = G_pinn - G_mc_interp
    
    # 相对误差（避免除以零）
    mask = G_mc_interp > 0.01 * G_mc_interp.max()  # 只在显著区域计算相对误差
    rel_error = np.abs(diff[mask]) / (np.abs(G_mc_interp[mask]) + 1e-10)
    
    errors = {
        'L1_absolute': np.mean(np.abs(diff)),
        'L2_absolute': np.sqrt(np.mean(diff**2)),
        'Linf_absolute': np.max(np.abs(diff)),
        'L1_relative': np.mean(rel_error),
        'L2_relative': np.sqrt(np.mean(rel_error**2)),
        'Linf_relative': np.max(rel_error),
        'mask_coverage': np.sum(mask) / mask.size
    }
    
    return errors, diff, G_mc_interp


def plot_comparison(G_pinn, G_mc, diff, x, y, z, case_key, output_folder):
    """生成对比图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    print("Generating comparison plots...")
    
    # 取中间切片
    mid_x = G_pinn.shape[0] // 2
    mid_y = G_pinn.shape[1] // 2
    mid_z = G_pinn.shape[2] // 2
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # 定义绘图函数
    def plot_slice(ax, data, title, cmap='viridis', vmin=None, vmax=None):
        im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=[y[0], y[-1], z[0], z[-1]])
        ax.set_title(title)
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        plt.colorbar(im, ax=ax)
        return im
    
    # 获取统一的颜色范围
    vmin = min(G_pinn.min(), G_mc.min())
    vmax = max(G_pinn.max(), G_mc.max())
    
    # X中间切片
    plot_slice(axes[0, 0], G_pinn[mid_x, :, :], 'PINN: G(x=0.5, y, z)', vmin=vmin, vmax=vmax)
    plot_slice(axes[0, 1], G_mc[mid_x, :, :], 'MC: G(x=0.5, y, z)', vmin=vmin, vmax=vmax)
    plot_slice(axes[0, 2], diff[mid_x, :, :], 'Difference', cmap='RdBu_r')
    plot_slice(axes[0, 3], np.abs(diff[mid_x, :, :]) / (np.abs(G_mc[mid_x, :, :]) + 1e-10), 
               'Relative Error', cmap='hot')
    
    # Y中间切片
    plot_slice(axes[1, 0], G_pinn[:, mid_y, :], 'PINN: G(x, y=0.5, z)', vmin=vmin, vmax=vmax)
    plot_slice(axes[1, 1], G_mc[:, mid_y, :], 'MC: G(x, y=0.5, z)', vmin=vmin, vmax=vmax)
    plot_slice(axes[1, 2], diff[:, mid_y, :], 'Difference', cmap='RdBu_r')
    plot_slice(axes[1, 3], np.abs(diff[:, mid_y, :]) / (np.abs(G_mc[:, mid_y, :]) + 1e-10), 
               'Relative Error', cmap='hot')
    
    # Z中间切片
    plot_slice(axes[2, 0], G_pinn[:, :, mid_z], 'PINN: G(x, y, z=0.5)', vmin=vmin, vmax=vmax)
    plot_slice(axes[2, 1], G_mc[:, :, mid_z], 'MC: G(x, y, z=0.5)', vmin=vmin, vmax=vmax)
    plot_slice(axes[2, 2], diff[:, :, mid_z], 'Difference', cmap='RdBu_r')
    plot_slice(axes[2, 3], np.abs(diff[:, :, mid_z]) / (np.abs(G_mc[:, :, mid_z]) + 1e-10), 
               'Relative Error', cmap='hot')
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, f'comparison_{case_key}.png')
    plt.savefig(plot_path, dpi=300)
    print(f"  Saved: {plot_path}")
    plt.close()
    
    # 中心线对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 沿x轴中心线
    center = G_pinn.shape[1] // 2
    axes[0].plot(x, G_pinn[:, center, center], 'b-', label='PINN', linewidth=2)
    axes[0].plot(x, G_mc[:, center, center], 'r--', label='MC', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('G')
    axes[0].set_title('Center line: G(x, 0.5, 0.5)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 相对误差沿中心线
    rel_err_line = np.abs(G_pinn[:, center, center] - G_mc[:, center, center]) / \
                   (np.abs(G_mc[:, center, center]) + 1e-10)
    axes[1].semilogy(x, rel_err_line, 'g-', linewidth=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Relative Error')
    axes[1].set_title('Relative Error along center line')
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    line_path = os.path.join(output_folder, f'centerline_{case_key}.png')
    plt.savefig(line_path, dpi=300)
    print(f"  Saved: {line_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare PINN with Monte Carlo validation')
    parser.add_argument('--case', type=str, required=True,
                       choices=['3D_A', '3D_B', '3D_C'],
                       help='Case to compare')
    parser.add_argument('--mc-path', type=str, default=None,
                       help='Path to MC results .npz file (auto-detected if not specified)')
    parser.add_argument('--n-theta', type=int, default=16,
                       help='Number of theta angles for integration (default: 16)')
    parser.add_argument('--n-phi', type=int, default=32,
                       help='Number of phi angles for integration (default: 32)')
    parser.add_argument('--grid', type=int, default=51,
                       help='Grid resolution (default: 51)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computation')
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"PINN vs Monte Carlo Comparison - {args.case}")
    print("="*70)
    
    # 自动检测MC路径
    if args.mc_path is None:
        mc_path = os.path.join(PROJECT_ROOT, 'Solvers', 'MC', 'MC3D_Results',
                               CASE_CONFIGS[args.case]['mc_expected'])
        if os.path.exists(mc_path):
            args.mc_path = mc_path
        else:
            raise FileNotFoundError(f"MC results not found. Please specify --mc-path")
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载模型
    model, folder_path = load_pinn_model(args.case, device)
    
    # 计算PINN的G场
    G_pinn, x, y, z = compute_G_pinn(
        model, 
        n_theta=args.n_theta, 
        n_phi=args.n_phi,
        nx=args.grid, ny=args.grid, nz=args.grid,
        device=device
    )
    
    print(f"\nPINN G field:")
    print(f"  Shape: {G_pinn.shape}")
    print(f"  Range: [{G_pinn.min():.4f}, {G_pinn.max():.4f}]")
    print(f"  Center: {G_pinn[args.grid//2, args.grid//2, args.grid//2]:.4f}")
    
    # 加载MC结果
    G_mc, x_mc, y_mc, z_mc = load_mc_results(args.mc_path)
    
    # 计算误差
    errors, diff, G_mc_interp = compute_errors(G_pinn, G_mc, x, y, z, x_mc, y_mc, z_mc)
    
    print("\n" + "="*70)
    print("Error Analysis")
    print("="*70)
    print(f"{'Metric':<25} {'Value':<15}")
    print("-"*70)
    for key, val in errors.items():
        print(f"{key:<25} {val:.6f}")
    print("="*70)
    
    # 保存结果
    print("\nSaving results...")
    
    # 保存G场
    np.savez(
        os.path.join(folder_path, f'G_field_{args.case}.npz'),
        G_pinn=G_pinn,
        G_mc=G_mc_interp,
        diff=diff,
        x=x, y=y, z=z,
        errors=errors
    )
    
    # 保存误差报告
    with open(os.path.join(folder_path, f'error_report_{args.case}.json'), 'w') as f:
        json.dump(errors, f, indent=2)
    
    # 生成对比图
    plot_comparison(G_pinn, G_mc_interp, diff, x, y, z, args.case, folder_path)
    
    print(f"\nResults saved to: {folder_path}")
    print("="*70)


if __name__ == "__main__":
    main()
