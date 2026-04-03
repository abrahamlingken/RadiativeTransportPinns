#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RMC3D Case A/B/C Solver
=======================
严格基于MATLAB代码RMC3D的Python实现
用于验证PINN在Case A/B/C上的结果

物理参数：
- Case A: κ=5.0, σs=0.0, g=0.0 (纯吸收)
- Case B: κ=0.5, σs=4.5, g=0.0 (各向同性散射)
- Case C: κ=0.5, σs=4.5, g=0.8 (前向散射)

源项: S(r) = max(0, 1 - 5*r)
计算域: [0,1]³ 单位立方体
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from numba import jit
import os
import time

# 绘图设置（与项目一致）
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = False
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10

# ==============================================================================
# 案例配置
# ==============================================================================
CASES = {
    'A': {'kappa': 5.0, 'sigma_s': 0.0, 'g': 0.0, 'name': 'Pure Absorption'},
    'B': {'kappa': 0.5, 'sigma_s': 4.5, 'g': 0.0, 'name': 'Isotropic Scattering'},
    'C': {'kappa': 0.5, 'sigma_s': 4.5, 'g': 0.8, 'name': 'Forward Scattering'}
}

# 计算参数
NRAY = 200000       # 蒙特卡洛射线数
NX, NY, NZ = 51, 51, 51  # 网格数
XLEN, YLEN, ZLEN = 1.0, 1.0, 1.0  # 域大小
CENTER = np.array([0.5, 0.5, 0.5])  # 源中心

# 派生参数
dx, dy, dz = XLEN/NX, YLEN/NY, ZLEN/NZ
xbound = np.linspace(0, XLEN, NX+1)
ybound = np.linspace(0, YLEN, NY+1)
zbound = np.linspace(0, ZLEN, NZ+1)
xcenter = np.linspace(0.5*dx, XLEN-0.5*dx, NX)
ycenter = np.linspace(0.5*dy, YLEN-0.5*dy, NY)
zcenter = np.linspace(0.5*dz, ZLEN-0.5*dz, NZ)


# ==============================================================================
# 核心函数（Numba加速）
# ==============================================================================
@jit(nopython=True, cache=True)
def source_term_val(x, y, z):
    """源项 S(r) = max(0, 1 - 5*r)"""
    r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2)
    s = 1.0 - 5.0 * r
    return s if s > 0 else 0.0


@jit(nopython=True, cache=True)
def intersect_boundary_numba(x0, y0, z0, ux, uy, uz, xb, yb, zb, dx, dy, dz):
    """
    计算射线与边界的交点（改编自MATLAB的IntersectBoundary.m）
    """
    # 确定当前网格索引
    ix = int(np.ceil(x0 / dx))
    iy = int(np.ceil(y0 / dy))
    iz = int(np.ceil(z0 / dz))
    
    # 确保索引在有效范围内
    ix = max(1, min(ix, NX))
    iy = max(1, min(iy, NY))
    iz = max(1, min(iz, NZ))
    
    # 根据方向确定边界
    if ux > 0:
        bx = xb[ix] if ix < len(xb) else xb[-1]
    else:
        bx = xb[ix-1] if ix > 0 else xb[0]
        
    if uy > 0:
        by = yb[iy] if iy < len(yb) else yb[-1]
    else:
        by = yb[iy-1] if iy > 0 else yb[0]
        
    if uz > 0:
        bz = zb[iz] if iz < len(zb) else zb[-1]
    else:
        bz = zb[iz-1] if iz > 0 else zb[0]
    
    # 计算到三个边界的距离
    tx = (bx - x0) / ux if abs(ux) > 1e-14 else 1e10
    ty = (by - y0) / uy if abs(uy) > 1e-14 else 1e10
    tz = (bz - z0) / uz if abs(uz) > 1e-14 else 1e10
    
    # 取最小正距离
    t_min = min(abs(tx), abs(ty), abs(tz))
    
    xb_new = x0 + ux * t_min
    yb_new = y0 + uy * t_min
    zb_new = z0 + uz * t_min
    
    # 边界检查
    xb_new = max(0.0, min(XLEN, xb_new))
    yb_new = max(0.0, min(YLEN, yb_new))
    zb_new = max(0.0, min(ZLEN, zb_new))
        
    dist = np.sqrt((xb_new-x0)**2 + (yb_new-y0)**2 + (zb_new-z0)**2)
    
    return xb_new, yb_new, zb_new, dist


@jit(nopython=True, cache=True)
def scatter_hg(ux, uy, uz, g):
    """
    Henyey-Greenstein散射（改编自MATLAB的scatter.m）
    返回新的方向向量
    """
    # 采样散射角（基于HG相函数）
    if abs(g) < 1e-10:  # 各向同性
        cos_theta = 1.0 - 2.0 * np.random.rand()
    else:
        # HG相函数采样
        s = (1.0 - g**2) / (1.0 - g + 2.0 * g * np.random.rand())
        cos_theta = (1.0 + g**2 - s**2) / (2.0 * g)
        cos_theta = max(-1.0, min(1.0, cos_theta))
    
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * np.random.rand()
    
    # 建立局部坐标系
    denom = np.sqrt((uz-uy)**2 + (ux-uz)**2 + (uy-ux)**2)
    if denom < 1e-14:
        ex1, ey1, ez1 = 1.0, 0.0, 0.0
    else:
        ex1 = (uz - uy) / denom
        ey1 = (ux - uz) / denom
        ez1 = (uy - ux) / denom
    
    # e2 = u × e1
    ex2 = uy * ez1 - uz * ey1
    ey2 = uz * ex1 - ux * ez1
    ez2 = ux * ey1 - uy * ex1
    
    # 新方向
    esx = sin_theta * (np.cos(phi) * ex1 + np.sin(phi) * ex2) + cos_theta * ux
    esy = sin_theta * (np.cos(phi) * ey1 + np.sin(phi) * ey2) + cos_theta * uy
    esz = sin_theta * (np.cos(phi) * ez1 + np.sin(phi) * ez2) + cos_theta * uz
    
    # 归一化
    norm = np.sqrt(esx**2 + esy**2 + esz**2)
    if norm > 1e-14:
        return esx/norm, esy/norm, esz/norm
    else:
        return ux, uy, uz


# ==============================================================================
# 主求解器
# ==============================================================================
def rmc_solve(case_key, output_dir='RMC3D_Results'):
    """
    RMC主求解器 - 改编自MATLAB的RMCtransparent.m
    """
    cfg = CASES[case_key]
    kappa = cfg['kappa']
    sigma_s = cfg['sigma_s']
    g = cfg['g']
    ke_val = kappa + sigma_s  # 消光系数
    albedo_val = sigma_s / ke_val if ke_val > 0 else 0.0
    
    print(f"\n{'='*70}")
    print(f"RMC3D Solver - Case {case_key}: {cfg['name']}")
    print(f"κ={kappa}, σs={sigma_s}, g={g}, Nray={NRAY}")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化统计数组（包含边界）
    RDFs = np.zeros((NX+2, NY+2, NZ+2))
    
    # 探测器位置（中心点）
    xyz0 = np.array([0.5, 0.5, 0.5])
    
    start_time = time.time()
    
    # 逐条射线追踪（简化版，避免复杂的批处理索引问题）
    for ray_idx in range(NRAY):
        if ray_idx % 20000 == 0 and ray_idx > 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {ray_idx}/{NRAY} ({100*ray_idx/NRAY:.1f}%), Time: {elapsed:.1f}s")
        
        # 初始化射线
        x, y, z = xyz0[0], xyz0[1], xyz0[2]
        
        # 随机方向（各向同性发射）
        theta = np.arccos(1.0 - 2.0 * np.random.rand())
        phi = 2.0 * np.pi * np.random.rand()
        ux = np.sin(theta) * np.cos(phi)
        uy = np.sin(theta) * np.sin(phi)
        uz = np.cos(theta)
        
        # 追踪循环
        while True:
            # 计算到边界的交点
            xb, yb, zb, dis = intersect_boundary_numba(
                x, y, z, ux, uy, uz,
                xbound, ybound, zbound, dx, dy, dz
            )
            
            # 采样随机光学厚度
            xi = np.random.rand()
            rand_opt_thi = -np.log(1.0 - xi)
            
            # 到边界的光学厚度
            opt_thi = ke_val * dis
            
            if rand_opt_thi > opt_thi:
                # 到达边界 - 记录
                ix = int(np.ceil(xb / dx))
                iy = int(np.ceil(yb / dy))
                iz = int(np.ceil(zb / dz))
                
                # 边界索引调整
                if ix < 1: ix = 1
                if ix > NX: ix = NX + 2
                else: ix = ix + 1
                
                if iy < 1: iy = 1
                if iy > NY: iy = NY + 2
                else: iy = iy + 1
                
                if iz < 1: iz = 1
                if iz > NZ: iz = NZ + 2
                else: iz = iz + 1
                
                RDFs[ix, iy, iz] += 1
                break
            else:
                # 在介质内发生作用
                x_new = x + rand_opt_thi / ke_val * ux
                y_new = y + rand_opt_thi / ke_val * uy
                z_new = z + rand_opt_thi / ke_val * uz
                
                # 边界限制
                x_new = max(1e-10, min(XLEN-1e-10, x_new))
                y_new = max(1e-10, min(YLEN-1e-10, y_new))
                z_new = max(1e-10, min(ZLEN-1e-10, z_new))
                
                # 判断吸收或散射
                if sigma_s > 0:
                    xi2 = np.random.rand()
                    if xi2 > albedo_val:
                        # 被吸收
                        ix = int(np.ceil(x_new / dx))
                        iy = int(np.ceil(y_new / dy))
                        iz = int(np.ceil(z_new / dz))
                        ix = max(1, min(NX, ix))
                        iy = max(1, min(NY, iy))
                        iz = max(1, min(NZ, iz))
                        RDFs[ix+1, iy+1, iz+1] += 1
                        break
                    else:
                        # 散射 - 更新位置和方向
                        x, y, z = x_new, y_new, z_new
                        ux, uy, uz = scatter_hg(ux, uy, uz, g)
                else:
                    # 纯吸收介质，直接记录吸收位置
                    ix = int(np.ceil(x_new / dx))
                    iy = int(np.ceil(y_new / dy))
                    iz = int(np.ceil(z_new / dz))
                    ix = max(1, min(NX, ix))
                    iy = max(1, min(NY, iy))
                    iz = max(1, min(NZ, iz))
                    RDFs[ix+1, iy+1, iz+1] += 1
                    break
    
    # 后处理
    RDF = RDFs[1:NX+1, 1:NY+1, 1:NZ+1] / NRAY
    G = RDF * 4.0 * np.pi
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"G range: [{G.min():.4f}, {G.max():.4f}]")
    ix_c, iy_c, iz_c = NX//2, NY//2, NZ//2
    print(f"Center G: {G[ix_c, iy_c, iz_c]:.4f}")
    
    # 保存结果
    np.savez(os.path.join(output_dir, f'RMC_Case{case_key}_results.npz'),
             G=G, xcenter=xcenter, ycenter=ycenter, zcenter=zcenter,
             kappa=kappa, sigma_s=sigma_s, g=g)
    
    return G


# ==============================================================================
# 可视化
# ==============================================================================
def plot_results(G, case_key, output_dir='RMC3D_Results'):
    """生成与项目风格一致的可视化结果"""
    cfg = CASES[case_key]
    print(f"\nPlotting Case {case_key}...")
    
    ix_c, iy_c, iz_c = NX//2, NY//2, NZ//2
    
    # 图1: 中心线对比
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xcenter, G[:, iy_c, iz_c], 'k-', linewidth=2, marker='o',
            markevery=5, markersize=6, label=f'RMC (N={NRAY})')
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$G(x, 0.5, 0.5)$', fontsize=12)
    ax.set_title(f'Case {case_key}: {cfg["name"]}\nκ={cfg["kappa"]}, σs={cfg["sigma_s"]}, g={cfg["g"]}',
                 fontsize=13)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'RMC_Case{case_key}_Centerline.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, f'RMC_Case{case_key}_Centerline.pdf'))
    plt.close()
    
    # 图2: 2D截面
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin, vmax = G.min(), G.max()
    
    slices = [
        (G[:, :, iz_c], 'x', 'y', xcenter, ycenter, f'z={zcenter[iz_c]:.2f}'),
        (G[:, iy_c, :], 'x', 'z', xcenter, zcenter, f'y={ycenter[iy_c]:.2f}'),
        (G[ix_c, :, :], 'y', 'z', ycenter, zcenter, f'x={xcenter[ix_c]:.2f}')
    ]
    
    for ax, (data, xlab, ylab, xcoord, ycoord, title) in zip(axes, slices):
        im = ax.contourf(xcoord, ycoord, data.T, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_xlabel(rf'${xlab}$', fontsize=11)
        ax.set_ylabel(rf'${ylab}$', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, pad=0.02)
    
    plt.suptitle(f'Case {case_key}: {cfg["name"]}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'RMC_Case{case_key}_2DSlices.png'), dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {output_dir}/")


# ==============================================================================
# 主程序
# ==============================================================================
def main():
    print("="*70)
    print("RMC3D Solver for Case A/B/C (Python Implementation)")
    print("="*70)
    
    for case in ['A', 'B', 'C']:
        G = rmc_solve(case)
        plot_results(G, case)
    
    print("\n" + "="*70)
    print("All cases completed!")
    print("="*70)


if __name__ == "__main__":
    main()
