#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rmc3d_solver.py - 3D Reverse Monte Carlo Solver for RTE

基于MATLAB代码RMC3D的Python实现，适配本项目Case A/B/C参数：
- Case A: κ=5.0, σs=0.0, g=0.0 (纯吸收)
- Case B: κ=0.5, σs=4.5, g=0.0 (各向同性散射)
- Case C: κ=0.5, σs=4.5, g=0.8 (前向散射)

源项: S(r) = max(0, 1 - 5*r), r < 0.2
边界: 真空边界 (无入射)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from numba import jit, prange
import os
import time

# 设置绘图风格（与项目一致）
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = False
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10

# ==========================================
# 物理参数配置 (Case A/B/C)
# ==========================================
CASE_CONFIGS = {
    'A': {'kappa': 5.0, 'sigma_s': 0.0, 'g': 0.0, 'name': 'Pure Absorption'},
    'B': {'kappa': 0.5, 'sigma_s': 4.5, 'g': 0.0, 'name': 'Isotropic Scattering'},
    'C': {'kappa': 0.5, 'sigma_s': 4.5, 'g': 0.8, 'name': 'Forward Scattering'}
}

# 网格参数
NX, NY, NZ = 51, 51, 51
XLEN, YLEN, ZLEN = 1.0, 1.0, 1.0

# 蒙特卡洛参数
N_RAY = 500000  # 射线数量（可根据精度需求调整）
BATCH_SIZE = 10000  # 批处理大小，避免内存溢出

# 源项中心
CENTER = np.array([0.5, 0.5, 0.5])


def source_term(x, y, z):
    """
    源项分布: S(r) = max(0, 1 - 5*r)
    仅在 r < 0.2 区域内非零
    """
    r = np.sqrt((x - CENTER[0])**2 + (y - CENTER[1])**2 + (z - CENTER[2])**2)
    return np.maximum(0.0, 1.0 - 5.0 * r)


def get_optical_properties(xyz, kappa, sigma_s):
    """
    获取位置处的光学性质
    本项目使用均匀介质，但保留接口以支持非均匀介质
    """
    ke = np.full(xyz.shape[0], kappa + sigma_s)  # 消光系数
    albedo = np.full(xyz.shape[0], sigma_s / (kappa + sigma_s) if (kappa + sigma_s) > 0 else 0.0)
    source = source_term(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    return ke, albedo, source


@jit(nopython=True, cache=True)
def intersect_boundary(x0, y0, z0, ux, uy, uz, xbound, ybound, zbound, dx, dy, dz):
    """
    计算射线与网格边界的交点（Numba加速）
    改编自MATLAB代码 IntersectBoundary.m
    """
    n = len(x0)
    xyzb = np.empty((n, 3))
    distL = np.empty(n)
    
    for i in range(n):
        # 确定当前网格索引
        ix = int(np.ceil(x0[i] / dx))
        iy = int(np.ceil(y0[i] / dy))
        iz = int(np.ceil(z0[i] / dz))
        
        # 确定边界（根据方向）
        if ux[i] > 0:
            bx = xbound[ix] if ix < len(xbound) else xbound[-1]
        else:
            bx = xbound[ix - 1] if ix > 0 else xbound[0]
            
        if uy[i] > 0:
            by = ybound[iy] if iy < len(ybound) else ybound[-1]
        else:
            by = ybound[iy - 1] if iy > 0 else ybound[0]
            
        if uz[i] > 0:
            bz = zbound[iz] if iz < len(zbound) else zbound[-1]
        else:
            bz = zbound[iz - 1] if iz > 0 else zbound[0]
        
        # 计算到三个边界的距离
        tx = (bx - x0[i]) / ux[i] if abs(ux[i]) > 1e-14 else 1e10
        ty = (by - y0[i]) / uy[i] if abs(uy[i]) > 1e-14 else 1e10
        tz = (bz - z0[i]) / uz[i] if abs(uz[i]) > 1e-14 else 1e10
        
        # 取最小正距离
        t_min = min(abs(tx), abs(ty), abs(tz))
        
        xyzb[i, 0] = x0[i] + ux[i] * t_min
        xyzb[i, 1] = y0[i] + uy[i] * t_min
        xyzb[i, 2] = z0[i] + uz[i] * t_min
        
        distL[i] = t_min
        
    return xyzb, distL


@jit(nopython=True, cache=True)
def hg_scatter(ux, uy, uz, g, n):
    """
    Henyey-Greenstein散射（替代MATLAB的简单散射模型）
    更符合本项目的物理模型
    """
    # 采样散射角theta
    if abs(g) < 1e-10:  # 各向同性
        cos_theta = 1.0 - 2.0 * np.random.rand(n)
    else:
        # HG相函数采样
        s = (1.0 - g**2) / (1.0 - g + 2.0 * g * np.random.rand(n))
        cos_theta = (1.0 + g**2 - s**2) / (2.0 * g)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * np.random.rand(n)
    
    # 建立局部坐标系
    # e3 = 入射方向
    e3x, e3y, e3z = ux, uy, uz
    
    # e1 垂直于e3 (使用叉乘找正交向量)
    ex1 = np.zeros(n)
    ey1 = np.zeros(n)
    ez1 = np.zeros(n)
    
    for i in range(n):
        if abs(e3z[i]) < 0.9:
            ex1[i] = 0.0
            ey1[i] = 0.0
            ez1[i] = 1.0
        else:
            ex1[i] = 1.0
            ey1[i] = 0.0
            ez1[i] = 0.0
    
    # e1 = e3 × temp, 然后归一化
    e1x = e3y * ez1 - e3z * ey1
    e1y = e3z * ex1 - e3x * ez1
    e1z = e3x * ey1 - e3y * ex1
    norm1 = np.sqrt(e1x**2 + e1y**2 + e1z**2)
    e1x /= norm1
    e1y /= norm1
    e1z /= norm1
    
    # e2 = e3 × e1
    e2x = e3y * e1z - e3z * e1y
    e2y = e3z * e1x - e3x * e1z
    e2z = e3x * e1y - e3y * e1x
    
    # 新方向 = sinθ(cosφ·e1 + sinφ·e2) + cosθ·e3
    ux_new = sin_theta * (np.cos(phi) * e1x + np.sin(phi) * e2x) + cos_theta * e3x
    uy_new = sin_theta * (np.cos(phi) * e1y + np.sin(phi) * e2y) + cos_theta * e3y
    uz_new = sin_theta * (np.cos(phi) * e1z + np.sin(phi) * e2z) + cos_theta * e3z
    
    # 归一化
    norm = np.sqrt(ux_new**2 + uy_new**2 + uz_new**2)
    return ux_new/norm, uy_new/norm, uz_new/norm


def rmc_solve(case_key, output_dir='RMC3D_Results'):
    """
    主求解函数 - 改编自MATLAB的RMCtransparent.m
    """
    config = CASE_CONFIGS[case_key]
    kappa = config['kappa']
    sigma_s = config['sigma_s']
    g = config['g']
    
    print(f"\n{'='*70}")
    print(f"RMC3D Solver - Case {case_key}: {config['name']}")
    print(f"Parameters: κ={kappa}, σs={sigma_s}, g={g}")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 网格设置
    dx, dy, dz = XLEN/NX, YLEN/NY, ZLEN/NZ
    xbound = np.linspace(0, XLEN, NX+1)
    ybound = np.linspace(0, YLEN, NY+1)
    zbound = np.linspace(0, ZLEN, NZ+1)
    
    xcenter = np.linspace(0.5*dx, XLEN-0.5*dx, NX)
    ycenter = np.linspace(0.5*dy, YLEN-0.5*dy, NY)
    zcenter = np.linspace(0.5*dz, ZLEN-0.5*dz, NZ)
    
    # 创建网格点（用于统计）
    X_grid, Y_grid, Z_grid = np.meshgrid(xcenter, ycenter, zcenter, indexing='ij')
    
    # 统计数组：记录每个网格的能量沉积
    # 扩展网格包含边界（与MATLAB一致）
    energy_deposit = np.zeros((NX+2, NY+2, NZ+2))
    
    # 探测器位置（从中心观测）
    xyz0 = np.array([0.5, 0.5, 0.5])
    
    start_time = time.time()
    
    # 分批处理射线
    n_batches = (N_RAY + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\nStarting Monte Carlo simulation...")
    print(f"Total rays: {N_RAY}, Batch size: {BATCH_SIZE}, Batches: {n_batches}")
    
    for batch in range(n_batches):
        batch_start = time.time()
        n_ray_batch = min(BATCH_SIZE, N_RAY - batch * BATCH_SIZE)
        
        # 初始化射线（从探测器位置发射）
        x0 = np.full(n_ray_batch, xyz0[0])
        y0 = np.full(n_ray_batch, xyz0[1])
        z0 = np.full(n_ray_batch, xyz0[2])
        
        # 随机方向（各向同性发射）
        theta = np.arccos(1.0 - 2.0 * np.random.rand(n_ray_batch))
        phi = 2.0 * np.pi * np.random.rand(n_ray_batch)
        ux = np.sin(theta) * np.cos(phi)
        uy = np.sin(theta) * np.sin(phi)
        uz = np.cos(theta)
        
        # 射线追踪循环
        active = np.ones(n_ray_batch, dtype=bool)
        
        while np.any(active):
            n_active = np.sum(active)
            
            # 采样随机光学厚度
            xi = np.random.rand(n_active)
            rand_opt_thi = -np.log(1.0 - xi)
            
            # 计算到边界的交点
            xyz_inter, dis = intersect_boundary(
                x0[active], y0[active], z0[active],
                ux[active], uy[active], uz[active],
                xbound, ybound, zbound, dx, dy, dz
            )
            
            # 获取中点位置的光学性质
            xyz_mid = 0.5 * (xyz_inter + np.column_stack([x0[active], y0[active], z0[active]]))
            ke, albedo, _ = get_optical_properties(xyz_mid, kappa, sigma_s)
            
            # 实际光学厚度
            opt_thi = ke * dis
            
            # 判断哪些射线发生相互作用
            ijk_done = rand_opt_thi <= opt_thi  # 在到达边界前发生散射/吸收
            ijk_boundary = rand_opt_thi > opt_thi  # 到达边界
            
            # 处理到达边界的射线
            if np.any(ijk_boundary):
                idx_boundary = np.where(active)[0][ijk_boundary]
                xyz_b = xyz_inter[ijk_boundary]
                
                # 统计边界出射（扩展到边界网格）
                ix = np.clip(np.ceil(xyz_b[:, 0] / dx).astype(int), 0, NX) + 1
                iy = np.clip(np.ceil(xyz_b[:, 1] / dy).astype(int), 0, NY) + 1
                iz = np.clip(np.ceil(xyz_b[:, 2] / dz).astype(int), 0, NZ) + 1
                
                for i in range(len(ix)):
                    energy_deposit[ix[i], iy[i], iz[i]] += 1
                
                # 这些射线终止
                active[idx_boundary] = False
            
            # 处理发生相互作用的射线
            if np.any(ijk_done):
                idx_done = np.where(active)[0][ijk_done]
                
                # 计算相互作用位置
                x_done = x0[active][ijk_done] + rand_opt_thi[ijk_done] / ke[ijk_done] * ux[active][ijk_done]
                y_done = y0[active][ijk_done] + rand_opt_thi[ijk_done] / ke[ijk_done] * uy[active][ijk_done]
                z_done = z0[active][ijk_done] + rand_opt_thi[ijk_done] / ke[ijk_done] * uz[active][ijk_done]
                
                # 边界检查
                x_done = np.clip(x_done, 1e-10, XLEN-1e-10)
                y_done = np.clip(y_done, 1e-10, YLEN-1e-10)
                z_done = np.clip(z_done, 1e-10, ZLEN-1e-10)
                
                # 判断吸收或散射
                xi2 = np.random.rand(len(x_done))
                ijk_absorb = xi2 > albedo[ijk_done]  # 被吸收
                ijk_scatter = xi2 <= albedo[ijk_done]  # 被散射
                
                # 处理吸收
                if np.any(ijk_absorb):
                    ix = np.clip(np.ceil(x_done[ijk_absorb] / dx).astype(int), 1, NX)
                    iy = np.clip(np.ceil(y_done[ijk_absorb] / dy).astype(int), 1, NY)
                    iz = np.clip(np.ceil(z_done[ijk_absorb] / dz).astype(int), 1, NZ)
                    for i in range(len(ix)):
                        energy_deposit[ix[i], iy[i], iz[i]] += 1
                
                # 处理散射 - 更新位置和方向
                if np.any(ijk_scatter):
                    idx_scatter = idx_done[ijk_scatter]
                    x0[idx_scatter] = x_done[ijk_scatter]
                    y0[idx_scatter] = y_done[ijk_scatter]
                    z0[idx_scatter] = z_done[ijk_scatter]
                    
                    # HG散射更新方向
                    ux_new, uy_new, uz_new = hg_scatter(
                        ux[active][ijk_done][ijk_scatter],
                        uy[active][ijk_done][ijk_scatter],
                        uz[active][ijk_done][ijk_scatter],
                        g,
                        np.sum(ijk_scatter)
                    )
                    ux[idx_scatter] = ux_new
                    uy[idx_scatter] = uy_new
                    uz[idx_scatter] = uz_new
                    
                    # 这些射线继续追踪
                    # 其他射线（被吸收或到边界）标记为inactive
                    inactive_mask = np.zeros(len(idx_done), dtype=bool)
                    inactive_mask[ijk_absorb] = True
                    active[idx_done[inactive_mask]] = False
                else:
                    # 全部吸收，无散射
                    active[idx_done] = False
            
            # 更新未完成的射线位置到边界
            if np.any(active):
                x0[active] = xyz_inter[ijk_boundary, 0]
                y0[active] = xyz_inter[ijk_boundary, 1]
                z0[active] = xyz_inter[ijk_boundary, 2]
        
        batch_time = time.time() - batch_start
        if (batch + 1) % 10 == 0 or batch == 0:
            print(f"  Batch {batch+1}/{n_batches} completed ({batch_time:.2f}s)")
    
    # 归一化得到辐射分布函数
    # 提取内部网格（排除边界）
    G_field = energy_deposit[1:NX+1, 1:NY+1, 1:NZ+1] / N_RAY
    
    # 转换为入射辐射 G(x) = 4π * (能量沉积/源项)
    # 这里简化为直接比例关系
    G_field = G_field * 4.0 * np.pi
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.1f}s")
    print(f"G_field range: [{G_field.min():.4f}, {G_field.max():.4f}]")
    
    # 保存结果
    np.savez(os.path.join(output_dir, f'RMC_Case{case_key}_results.npz'),
             G=G_field, X=X_grid, Y=Y_grid, Z=Z_grid,
             kappa=kappa, sigma_s=sigma_s, g=g,
             xcenter=xcenter, ycenter=ycenter, zcenter=zcenter)
    
    return G_field, X_grid, Y_grid, Z_grid, xcenter, ycenter, zcenter


def plot_results(G, X, Y, Z, xcenter, ycenter, zcenter, case_key, output_dir='RMC3D_Results'):
    """
    生成与项目风格一致的可视化结果
    """
    print(f"\nGenerating plots for Case {case_key}...")
    
    config = CASE_CONFIGS[case_key]
    os.makedirs(output_dir, exist_ok=True)
    
    # 找到中心索引
    ix_center = len(xcenter) // 2
    iy_center = len(ycenter) // 2
    iz_center = len(zcenter) // 2
    
    # 1. 中心线对比图 (与validate_3d_pure_absorption.py风格一致)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 沿x方向中心线 (y=z=0.5)
    G_centerline = G[:, iy_center, iz_center]
    ax.plot(xcenter, G_centerline, 'k-', linewidth=2, marker='o', 
            markevery=5, markersize=6, label=f'RMC (N={N_RAY})')
    
    ax.set_xlabel(r'$x$ (optical depth)', fontsize=12)
    ax.set_ylabel(r'$G(x, 0.5, 0.5)$', fontsize=12)
    ax.set_title(f'Case {case_key}: {config["name"]}\n' + 
                 f'κ={config["kappa"]}, σs={config["sigma_s"]}, g={config["g"]}',
                 fontsize=13)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'RMC_Case{case_key}_Centerline.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, f'RMC_Case{case_key}_Centerline.pdf'))
    plt.close()
    
    # 2. 2D截面热图 (与plot_3d_paper_figures风格一致)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 三个正交截面
    slices = [
        (G[ix_center, :, :], 'y', 'z', ycenter, zcenter, f'x={xcenter[ix_center]:.2f}'),
        (G[:, iy_center, :], 'x', 'z', xcenter, zcenter, f'y={ycenter[iy_center]:.2f}'),
        (G[:, :, iz_center], 'x', 'y', xcenter, ycenter, f'z={zcenter[iz_center]:.2f}')
    ]
    
    vmin, vmax = G.min(), G.max()
    levels = np.linspace(vmin, vmax, 20)
    
    for idx, (data, xlab, ylab, xcoord, ycoord, title) in enumerate(slices):
        im = axes[idx].contourf(xcoord, ycoord, data.T, levels=levels, 
                                cmap='hot', extend='both')
        axes[idx].set_xlabel(rf'${xlab}$', fontsize=11)
        axes[idx].set_ylabel(rf'${ylab}$', fontsize=11)
        axes[idx].set_title(title, fontsize=11)
        axes[idx].set_aspect('equal')
        cbar = plt.colorbar(im, ax=axes[idx], pad=0.02)
        cbar.set_label(r'$G$', fontsize=10)
    
    plt.suptitle(f'Case {case_key}: {config["name"]} - Center Slices', 
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'RMC_Case{case_key}_2DSlices.png'), 
                dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'RMC_Case{case_key}_2DSlices.pdf'), 
                bbox_inches='tight')
    plt.close()
    
    # 3. 3D体积渲染投影（可选）
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 最大强度投影
    max_proj_z = np.max(G, axis=2)
    im = ax.contourf(xcenter, ycenter, max_proj_z.T, levels=20, cmap='hot')
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_title(f'Case {case_key}: Max Intensity Projection (along z)', fontsize=13)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=r'$G_{max}$')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'RMC_Case{case_key}_MIP.png'), dpi=600)
    plt.close()
    
    print(f"  Plots saved to {output_dir}/")


def main():
    """
    主程序：运行所有Case
    """
    print("="*70)
    print("RMC3D Solver - Monte Carlo Verification for 3D RTE Cases")
    print("="*70)
    
    for case in ['A', 'B', 'C']:
        G, X, Y, Z, xc, yc, zc = rmc_solve(case)
        plot_results(G, X, Y, Z, xc, yc, zc, case)
        
        # 打印中心点值（与PINN对比）
        ix, iy, iz = len(xc)//2, len(yc)//2, len(zc)//2
        print(f"\nCase {case} Center G-value: {G[ix, iy, iz]:.4f}")
    
    print("\n" + "="*70)
    print("All cases completed!")
    print("="*70)


if __name__ == "__main__":
    main()
