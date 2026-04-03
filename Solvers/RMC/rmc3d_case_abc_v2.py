#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RMC3D Case A/B/C Solver - Version 2
====================================
修正版：Case A使用解析解，Case B/C使用RMC

物理参数：
- Case A: κ=5.0, σs=0.0, g=0.0 (纯吸收) → 使用Beer-Lambert解析解
- Case B: κ=0.5, σs=4.5, g=0.0 (各向同性散射) → RMC
- Case C: κ=0.5, σs=4.5, g=0.8 (前向散射) → RMC
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from numba import jit
from scipy.special import roots_legendre
import os
import time

# 绘图设置
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = False
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10

# ==============================================================================
# 参数配置
# ==============================================================================
CASES = {
    'A': {'kappa': 5.0, 'sigma_s': 0.0, 'g': 0.0, 'name': 'Pure Absorption (Analytical)'},
    'B': {'kappa': 0.5, 'sigma_s': 4.5, 'g': 0.0, 'name': 'Isotropic Scattering (RMC)'},
    'C': {'kappa': 0.5, 'sigma_s': 4.5, 'g': 0.8, 'name': 'Forward Scattering (RMC)'}
}

# 计算参数
NRAY = 100000       # RMC射线数（用于B/C）
NX, NY, NZ = 51, 51, 51
XLEN, YLEN, ZLEN = 1.0, 1.0, 1.0
CENTER = np.array([0.5, 0.5, 0.5])

dx, dy, dz = XLEN/NX, YLEN/NY, ZLEN/NZ
xcenter = np.linspace(0.5*dx, XLEN-0.5*dx, NX)
ycenter = np.linspace(0.5*dy, YLEN-0.5*dy, NY)
zcenter = np.linspace(0.5*dz, ZLEN-0.5*dz, NZ)

# 边界
xbound = np.linspace(0, XLEN, NX+1)
ybound = np.linspace(0, YLEN, NY+1)
zbound = np.linspace(0, ZLEN, NZ+1)


# ==============================================================================
# Case A: 纯吸收解析解
# ==============================================================================
def source_term(x, y, z):
    """源项 S(r) = max(0, 1 - 5*r)"""
    r = np.sqrt((x - CENTER[0])**2 + (y - CENTER[1])**2 + (z - CENTER[2])**2)
    return np.maximum(0.0, 1.0 - 5.0 * r)


def compute_exact_intensity(x, y, z, theta, phi, kappa, n_points=100):
    """
    计算纯吸收介质中某点某方向的精确强度
    I(x,s) = ∫_0^L S(x - l*s) * exp(-κ*l) dl
    """
    # 方向向量
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    s_vec = np.array([st * cp, st * sp, ct])
    
    pos = np.array([x, y, z])
    
    # 计算到边界的距离L（沿-s方向）
    t_candidates = []
    for i in range(3):
        if abs(s_vec[i]) > 1e-14:
            t_to_0 = pos[i] / s_vec[i]
            t_to_1 = (pos[i] - 1.0) / s_vec[i]
            if t_to_0 > 1e-14:
                t_candidates.append(t_to_0)
            if t_to_1 > 1e-14:
                t_candidates.append(t_to_1)
    
    if len(t_candidates) == 0:
        return 0.0
    
    L = min(t_candidates)
    if L <= 1e-14:
        return 0.0
    
    # Gauss-Legendre积分
    xi, w = roots_legendre(n_points)
    l_vals = 0.5 * L * (xi + 1.0)
    weights = 0.5 * L * w
    
    integrand = np.zeros(n_points)
    for i in range(n_points):
        curr_pos = pos - l_vals[i] * s_vec
        S_val = source_term(curr_pos[0], curr_pos[1], curr_pos[2])
        integrand[i] = S_val * np.exp(-kappa * l_vals[i])
    
    return np.sum(integrand * weights)


def solve_case_a_analytical(kappa):
    """
    Case A解析解：计算整个场的G(x)
    """
    print("\n  Using analytical solution (Beer-Lambert integral)...")
    
    # 角度离散
    n_theta, n_phi = 16, 32
    mu, w_mu = roots_legendre(n_theta)
    theta_q = np.arccos(-mu)
    phi_q = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    phi_q += np.pi / n_phi
    w_phi = np.full(n_phi, 2*np.pi / n_phi)
    
    # 创建3D网格
    X, Y, Z = np.meshgrid(xcenter, ycenter, zcenter, indexing='ij')
    G = np.zeros((NX, NY, NZ))
    
    total_points = NX * NY * NZ
    start = time.time()
    
    for idx in range(total_points):
        ix = idx // (NY * NZ)
        iy = (idx % (NY * NZ)) // NZ
        iz = idx % NZ
        
        x, y, z = X[ix, iy, iz], Y[ix, iy, iz], Z[ix, iy, iz]
        
        # 对所有方向积分
        G_val = 0.0
        for i, theta in enumerate(theta_q):
            for j, phi in enumerate(phi_q):
                I_val = compute_exact_intensity(x, y, z, theta, phi, kappa, n_points=50)
                G_val += I_val * w_mu[i] * w_phi[j]
        
        G[ix, iy, iz] = G_val
        
        if idx % 5000 == 0 and idx > 0:
            elapsed = time.time() - start
            print(f"    Progress: {idx}/{total_points} ({100*idx/total_points:.1f}%), Time: {elapsed:.1f}s")
    
    return G


# ==============================================================================
# Case B/C: RMC求解器（修正版）
# ==============================================================================
@jit(nopython=True, cache=True)
def intersect_boundary(x0, y0, z0, ux, uy, uz, dx, dy, dz):
    """计算射线与边界交点"""
    # 到六个边界的距离
    tx = 1e10
    ty = 1e10  
    tz = 1e10
    
    if abs(ux) > 1e-14:
        if ux > 0:
            tx = (XLEN - x0) / ux
        else:
            tx = -x0 / ux
    
    if abs(uy) > 1e-14:
        if uy > 0:
            ty = (YLEN - y0) / uy
        else:
            ty = -y0 / uy
    
    if abs(uz) > 1e-14:
        if uz > 0:
            tz = (ZLEN - z0) / uz
        else:
            tz = -z0 / uz
    
    t_min = min(abs(tx), abs(ty), abs(tz))
    
    xb = x0 + ux * t_min
    yb = y0 + uy * t_min
    zb = z0 + uz * t_min
    
    # 边界限制
    xb = max(0.0, min(XLEN, xb))
    yb = max(0.0, min(YLEN, yb))
    zb = max(0.0, min(ZLEN, zb))
    
    dist = np.sqrt((xb-x0)**2 + (yb-y0)**2 + (zb-z0)**2)
    return xb, yb, zb, dist


@jit(nopython=True, cache=True)
def scatter_direction(ux, uy, uz, g):
    """HG散射"""
    if abs(g) < 1e-10:
        cos_theta = 1.0 - 2.0 * np.random.rand()
    else:
        s = (1.0 - g**2) / (1.0 - g + 2.0 * g * np.random.rand())
        cos_theta = (1.0 + g**2 - s**2) / (2.0 * g)
        cos_theta = max(-1.0, min(1.0, cos_theta))
    
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * np.random.rand()
    
    # 局部坐标系
    denom = np.sqrt((uz-uy)**2 + (ux-uz)**2 + (uy-ux)**2)
    if denom < 1e-14:
        ex1, ey1, ez1 = 1.0, 0.0, 0.0
    else:
        ex1 = (uz - uy) / denom
        ey1 = (ux - uz) / denom
        ez1 = (uy - ux) / denom
    
    ex2 = uy * ez1 - uz * ey1
    ey2 = uz * ex1 - ux * ez1
    ez2 = ux * ey1 - uy * ex1
    
    esx = sin_theta * (np.cos(phi) * ex1 + np.sin(phi) * ex2) + cos_theta * ux
    esy = sin_theta * (np.cos(phi) * ey1 + np.sin(phi) * ey2) + cos_theta * uy
    esz = sin_theta * (np.cos(phi) * ez1 + np.sin(phi) * ez2) + cos_theta * uz
    
    norm = np.sqrt(esx**2 + esy**2 + esz**2)
    if norm > 1e-14:
        return esx/norm, esy/norm, esz/norm
    return ux, uy, uz


def solve_case_bc_rmc(case_key, kappa, sigma_s, g):
    """
    Case B/C的RMC求解（正向蒙特卡洛：从源发射光子）
    """
    print(f"\n  Using Forward Monte Carlo (N={NRAY})...")
    
    ke_val = kappa + sigma_s
    albedo = sigma_s / ke_val if ke_val > 0 else 0.0
    
    # 统计网格
    G = np.zeros((NX, NY, NZ))
    
    # 源区域（r < 0.2）
    source_radius = 0.2
    
    start = time.time()
    
    for ray in range(NRAY):
        if ray % 20000 == 0 and ray > 0:
            elapsed = time.time() - start
            print(f"    Progress: {ray}/{NRAY} ({100*ray/NRAY:.1f}%), Time: {elapsed:.1f}s")
        
        # 在源区域内随机采样起点（球内均匀分布）
        r = source_radius * np.random.rand()**(1/3)  # 球内均匀
        theta_s = np.arccos(1.0 - 2.0 * np.random.rand())
        phi_s = 2.0 * np.pi * np.random.rand()
        
        x = CENTER[0] + r * np.sin(theta_s) * np.cos(phi_s)
        y = CENTER[1] + r * np.sin(theta_s) * np.sin(phi_s)
        z = CENTER[2] + r * np.cos(theta_s)
        
        # 各向同性发射
        theta = np.arccos(1.0 - 2.0 * np.random.rand())
        phi = 2.0 * np.pi * np.random.rand()
        ux = np.sin(theta) * np.cos(phi)
        uy = np.sin(theta) * np.sin(phi)
        uz = np.cos(theta)
        
        # 初始权重
        weight = source_term(x, y, z) / NRAY
        
        # 追踪光子
        while weight > 1e-10:
            # 计算到边界的距离
            xb, yb, zb, dis = intersect_boundary(x, y, z, ux, uy, uz, dx, dy, dz)
            
            # 计算光学厚度
            opt_thick = ke_val * dis
            
            # 权重衰减（Beer-Lambert）
            weight_new = weight * np.exp(-opt_thick)
            
            # 记录贡献（简化：只记录最终位置）
            if weight_new > 0:
                ix = int(np.clip(np.floor(xb / dx), 0, NX-1))
                iy = int(np.clip(np.floor(yb / dy), 0, NY-1))
                iz = int(np.clip(np.floor(zb / dz), 0, NZ-1))
                G[ix, iy, iz] += weight_new
            
            # 判断是否散射
            if sigma_s > 0:
                # 采样散射位置
                xi = np.random.rand()
                if xi < (1.0 - np.exp(-opt_thick)):  # 发生相互作用
                    # 散射位置
                    l_scatter = -np.log(1.0 - xi * (1.0 - np.exp(-opt_thick))) / ke_val
                    x = x + l_scatter * ux
                    y = y + l_scatter * uy
                    z = z + l_scatter * uz
                    
                    # 更新权重
                    weight = weight * (1.0 - np.exp(-opt_thick)) * albedo
                    
                    # 更新方向
                    ux, uy, uz = scatter_direction(ux, uy, uz, g)
                else:
                    break  # 直接穿透
            else:
                break  # 无散射，终止
    
    # 归一化
    G = G * 4.0 * np.pi
    
    return G


# ==============================================================================
# 主求解函数
# ==============================================================================
def solve_case(case_key, output_dir='RMC3D_Results'):
    """求解单个案例"""
    cfg = CASES[case_key]
    kappa = cfg['kappa']
    sigma_s = cfg['sigma_s']
    g = cfg['g']
    
    print(f"\n{'='*70}")
    print(f"Case {case_key}: {cfg['name']}")
    print(f"κ={kappa}, σs={sigma_s}, g={g}")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    start = time.time()
    
    if case_key == 'A':
        # Case A: 纯吸收，使用解析解
        G = solve_case_a_analytical(kappa)
    else:
        # Case B/C: 使用RMC
        G = solve_case_bc_rmc(case_key, kappa, sigma_s, g)
    
    elapsed = time.time() - start
    
    print(f"\n  Completed in {elapsed:.1f}s")
    ix_c, iy_c, iz_c = NX//2, NY//2, NZ//2
    print(f"  G range: [{G.min():.4f}, {G.max():.4f}]")
    print(f"  Center G: {G[ix_c, iy_c, iz_c]:.4f}")
    
    # 保存
    np.savez(os.path.join(output_dir, f'RMC_Case{case_key}_results.npz'),
             G=G, xcenter=xcenter, ycenter=ycenter, zcenter=zcenter,
             kappa=kappa, sigma_s=sigma_s, g=g)
    
    return G


# ==============================================================================
# 可视化
# ==============================================================================
def plot_results(G, case_key, output_dir='RMC3D_Results'):
    """绘制结果"""
    cfg = CASES[case_key]
    print(f"\n  Plotting...")
    
    ix_c, iy_c, iz_c = NX//2, NY//2, NZ//2
    
    # 中心线
    fig, ax = plt.subplots(figsize=(8, 6))
    method = "Analytical" if case_key == 'A' else f"RMC (N={NRAY})"
    ax.plot(xcenter, G[:, iy_c, iz_c], 'k-', linewidth=2, marker='o',
            markevery=5, markersize=6, label=method)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$G(x, 0.5, 0.5)$', fontsize=12)
    ax.set_title(f'Case {case_key}: {cfg["name"]}', fontsize=13)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'RMC_Case{case_key}_Centerline.png'), dpi=600)
    plt.close()
    
    # 2D截面
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
    print("RMC3D Solver v2 - Case A (Analytical) / Case B/C (RMC)")
    print("="*70)
    
    for case in ['A', 'B', 'C']:
        G = solve_case(case)
        plot_results(G, case)
    
    print("\n" + "="*70)
    print("All cases completed!")
    print("="*70)


if __name__ == "__main__":
    main()
