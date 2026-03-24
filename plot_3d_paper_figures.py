#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_3d_paper_figures.py - 3D RTE 顶刊论文绘图脚本 (OOP版本)

生成论文质量的可视化图表：
1. 入射辐射 G(x) 沿x轴分布对比
2. 中心截面热图
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置顶刊字体
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = False
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 添加 Core 目录（用于 torch.load 找到 ModelClassTorch2）
CORE_PATH = os.path.join(PROJECT_ROOT, 'Core')
if CORE_PATH not in sys.path:
    sys.path.insert(0, CORE_PATH)

from EquationModels import RadTrans3D_Complex as Ec
from ModelClassTorch2 import Pinns  # 确保加载时能找到类


def load_model_and_compute_G(case_folder, x_tensor, y_tensor, z_tensor, engine):
    """
    加载模型并计算入射辐射 G(x)
    
    Args:
        case_folder: 案例文件夹路径
        x_tensor, y_tensor, z_tensor: 空间坐标张量
        engine: RadTrans3D_Physics 实例
    
    Returns:
        G: 入射辐射 numpy 数组
    """
    model_path = os.path.join(case_folder, 'model.pkl')
    
    if not os.path.exists(model_path):
        print(f"  [Warning] Model not found: {model_path}")
        return None
    
    # 加载模型 (使用 engine.dev 作为 map_location)
    model = torch.load(model_path, map_location=engine.dev, weights_only=False)
    model.eval()
    
    # 使用 engine 的方法计算入射辐射
    with torch.no_grad():
        G_tensor = engine.compute_incident_radiation(x_tensor, y_tensor, z_tensor, model)
    
    G = G_tensor.cpu().numpy()
    
    print(f"  Loaded {case_folder}: G range = [{G.min():.4f}, {G.max():.4f}]")
    
    return G


def main():
    """主程序：生成论文图表"""
    
    # 案例配置
    cases = {
        'Case 3D-A (Pure Absorption)': {
            'folder': 'Results_3D_CaseA',
            'color': 'blue',
            'linestyle': '-',
            'marker': 'o'
        },
        'Case 3D-B (Isotropic)': {
            'folder': 'Results_3D_CaseB',
            'color': 'red',
            'linestyle': '--',
            'marker': 's'
        },
        'Case 3D-C (Forward Scattering)': {
            'folder': 'Results_3D_CaseC',
            'color': 'green',
            'linestyle': '-.',
            'marker': '^'
        }
    }
    
    # 创建输出目录
    output_dir = 'Figures_3D'
    os.makedirs(output_dir, exist_ok=True)
    
    # 实例化物理引擎 (OOP核心！)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine = Ec.RadTrans3D_Physics(dev=device)
    
    print("="*70)
    print("Generating 3D Paper Figures")
    print("="*70)
    print(f"Device: {device}")
    
    # ========================================================================
    # 图1: G(x) 沿x轴分布 (y=z=0.5, 中心线)
    # ========================================================================
    print("\n[1] Generating G(x) along centerline...")
    
    n_points = 100
    x_1d = torch.linspace(0, 1, n_points, device=engine.dev)
    y_1d = torch.ones_like(x_1d) * 0.5
    z_1d = torch.ones_like(x_1d) * 0.5
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for case_name, config in cases.items():
        G = load_model_and_compute_G(
            config['folder'], x_1d, y_1d, z_1d, engine
        )
        
        if G is not None:
            x_np = x_1d.cpu().numpy()
            ax.plot(x_np, G, 
                   label=case_name,
                   color=config['color'],
                   linestyle=config['linestyle'],
                   linewidth=2,
                   marker=config['marker'],
                   markevery=10,
                   markersize=6)
    
    ax.set_xlabel(r'$x$ (optical depth)', fontsize=12)
    ax.set_ylabel(r'$G(x, 0.5, 0.5)$', fontsize=12)
    ax.set_title(r'Incident Radiation along Centerline ($y=z=0.5$)', fontsize=13)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G_along_centerline.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'G_along_centerline.pdf'))
    print(f"  Saved: {output_dir}/G_along_centerline.png")
    plt.close()
    
    # ========================================================================
    # 图2: 中心截面热图 (z=0.5)
    # ========================================================================
    print("\n[2] Generating center slice heatmaps...")
    
    n_grid = 50
    x_grid = torch.linspace(0, 1, n_grid, device=engine.dev)
    y_grid = torch.linspace(0, 1, n_grid, device=engine.dev)
    X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
    
    # 展平
    x_flat = X_grid.reshape(-1)
    y_flat = Y_grid.reshape(-1)
    z_flat = torch.ones_like(x_flat) * 0.5
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (case_name, config) in enumerate(cases.items()):
        G = load_model_and_compute_G(
            config['folder'], x_flat, y_flat, z_flat, engine
        )
        
        if G is not None:
            G_2d = G.reshape(n_grid, n_grid)
            
            im = axes[idx].contourf(X_grid.cpu().numpy(), Y_grid.cpu().numpy(), G_2d,
                                     levels=20, cmap='hot', extend='both')
            axes[idx].set_xlabel(r'$x$', fontsize=11)
            axes[idx].set_ylabel(r'$y$', fontsize=11)
            axes[idx].set_title(case_name, fontsize=11)
            axes[idx].set_aspect('equal')
            
            # colorbar
            cbar = plt.colorbar(im, ax=axes[idx], pad=0.02)
            cbar.set_label(r'$G$', fontsize=10)
    
    plt.suptitle(r'Incident Radiation at $z=0.5$ Plane', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G_center_slice.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'G_center_slice.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_dir}/G_center_slice.png")
    plt.close()
    
    # ========================================================================
    # 图3: 3D 体渲染数据导出 (VTK格式，用于ParaView)
    # ========================================================================
    print("\n[3] Exporting 3D VTK data for ParaView...")
    
    try:
        from pyevtk.hl import gridToVTK
        
        n_vtk = 30
        x_vtk = np.linspace(0, 1, n_vtk)
        y_vtk = np.linspace(0, 1, n_vtk)
        z_vtk = np.linspace(0, 1, n_vtk)
        X_vtk, Y_vtk, Z_vtk = np.meshgrid(x_vtk, y_vtk, z_vtk, indexing='ij')
        
        x_vtk_t = torch.tensor(X_vtk.flatten(), device=engine.dev, dtype=torch.float32)
        y_vtk_t = torch.tensor(Y_vtk.flatten(), device=engine.dev, dtype=torch.float32)
        z_vtk_t = torch.tensor(Z_vtk.flatten(), device=engine.dev, dtype=torch.float32)
        
        for case_name, config in cases.items():
            G = load_model_and_compute_G(
                config['folder'], x_vtk_t, y_vtk_t, z_vtk_t, engine
            )
            
            if G is not None:
                G_vtk = G.reshape(n_vtk, n_vtk, n_vtk)
                
                # 保存VTK
                case_short = case_name.replace(' ', '_').replace('(', '').replace(')', '')
                vtk_path = os.path.join(output_dir, f'G_3D_{case_short}')
                gridToVTK(vtk_path, X_vtk, Y_vtk, Z_vtk, 
                         pointData={"G": G_vtk})
                print(f"  Saved: {vtk_path}.vts")
    
    except ImportError:
        print("  [Skip] pyevtk not installed, skipping VTK export")
    
    print("\n" + "="*70)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
