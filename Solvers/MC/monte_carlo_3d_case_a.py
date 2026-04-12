#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte Carlo 3D RTE Solver for Case A (Pure Absorption)
======================================================
用于验证MC方法的正确性

Physics:
- Pure absorption: κ = 5.0, σs = 0.0, β = 5.0
- No scattering
- Source: S(r) = max(0, 1-5r), spherical at center

Exact solution available via numerical integration:
G(r) = ∫∫∫ S(r') * exp(-β|r-r'|) / |r-r'|² * (β/4π) dV'
"""

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import sys
import time
from scipy import integrate

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# 配置参数
# ============================================================================
NX, NY, NZ = 51, 51, 51  # 与PINN一致
N_PHOTONS = 5_000_000    # 光子数
BATCH_SIZE = 50_000

# Case A参数
KAPPA = 5.0
SIGMA_S = 0.0
BETA = 5.0
G_HG = 0.0  # 无散射

# ============================================================================
# 核心函数（Numba加速）
# ============================================================================

@njit(cache=True, fastmath=True)
def source_term(x, y, z):
    """源项 S(r) = max(0, 1 - 5*r)"""
    r = np.sqrt((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)
    s = 1.0 - 5.0 * r
    return s if s > 0.0 else 0.0


@njit(cache=True, fastmath=True)
def sample_source_point():
    """在源区域内采样起点（拒绝采样）"""
    while True:
        x, y, z = np.random.random(3)
        if np.random.random() < source_term(x, y, z):
            return x, y, z


@njit(cache=True, fastmath=True)
def sample_isotropic_direction():
    """各向同性采样方向"""
    cos_theta = 2.0 * np.random.random() - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * np.random.random()
    sx = sin_theta * np.cos(phi)
    sy = sin_theta * np.sin(phi)
    sz = cos_theta
    return sx, sy, sz


@njit(cache=True, fastmath=True)
def distance_to_boundary(x, y, z, sx, sy, sz):
    """计算到边界的距离"""
    l_boundary = 1e10
    if sx > 1e-15:
        l = (1.0 - x) / sx
        if l > 0: l_boundary = min(l_boundary, l)
    elif sx < -1e-15:
        l = -x / sx
        if l > 0: l_boundary = min(l_boundary, l)
    if sy > 1e-15:
        l = (1.0 - y) / sy
        if l > 0: l_boundary = min(l_boundary, l)
    elif sy < -1e-15:
        l = -y / sy
        if l > 0: l_boundary = min(l_boundary, l)
    if sz > 1e-15:
        l = (1.0 - z) / sz
        if l > 0: l_boundary = min(l_boundary, l)
    elif sz < -1e-15:
        l = -z / sz
        if l > 0: l_boundary = min(l_boundary, l)
    return l_boundary


@njit(cache=True, fastmath=True)
def track_photon_pure_absorption(kappa, beta, nx, ny, nz, dx, dy, dz):
    """
    追踪单个光子（纯吸收情况，无散射）
    记录光子对G的贡献（路径积分估计器）
    """
    # 初始化
    x, y, z = sample_source_point()
    sx, sy, sz = sample_isotropic_direction()
    
    # 网格索引数组和权重
    max_points = 1000
    ix_arr = np.zeros(max_points, dtype=np.int32)
    iy_arr = np.zeros(max_points, dtype=np.int32)
    iz_arr = np.zeros(max_points, dtype=np.int32)
    weight_arr = np.zeros(max_points, dtype=np.float64)
    
    n_points = 0
    weight = source_term(x, y, z)
    distance_traveled = 0.0
    
    while weight > 1e-10 and n_points < max_points:
        # 计算到边界的距离
        l_boundary = distance_to_boundary(x, y, z, sx, sy, sz)
        
        # 采样光学厚度（纯吸收，直接采样到吸收或逃逸）
        tau = -np.log(1.0 - np.random.random())
        l_collision = tau / beta if beta > 1e-15 else 1e10
        
        # 在纯吸收介质中，光子要么被吸收，要么逃逸
        # 但我们需要记录路径上各点对G的贡献
        
        if l_collision < l_boundary:
            # 光子将被吸收，追踪到吸收点
            # 在到达吸收点前，记录路径上的贡献
            n_steps = int(l_collision / min(dx, dy, dz)) + 1
            dl = l_collision / n_steps
            
            for i in range(n_steps):
                xi = x + (i + 0.5) * dl * sx
                yi = y + (i + 0.5) * dl * sy
                zi = z + (i + 0.5) * dl * sz
                
                ix = int(xi / dx)
                iy = int(yi / dy)
                iz = int(zi / dz)
                
                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    # 路径积分估计器：权重衰减 exp(-β*l)
                    attenuation = np.exp(-beta * (i + 0.5) * dl)
                    if n_points < max_points:
                        ix_arr[n_points] = ix
                        iy_arr[n_points] = iy
                        iz_arr[n_points] = iz
                        weight_arr[n_points] = weight * attenuation
                        n_points += 1
            
            break  # 光子被吸收，终止
        else:
            # 光子逃逸出边界，记录逃逸前的路径
            n_steps = int(l_boundary / min(dx, dy, dz)) + 1
            dl = l_boundary / n_steps
            
            for i in range(n_steps):
                xi = x + (i + 0.5) * dl * sx
                yi = y + (i + 0.5) * dl * sy
                zi = z + (i + 0.5) * dl * sz
                
                ix = int(xi / dx)
                iy = int(yi / dy)
                iz = int(zi / dz)
                
                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    attenuation = np.exp(-beta * (i + 0.5) * dl)
                    if n_points < max_points:
                        ix_arr[n_points] = ix
                        iy_arr[n_points] = iy
                        iz_arr[n_points] = iz
                        weight_arr[n_points] = weight * attenuation
                        n_points += 1
            
            break  # 光子逃逸，终止
    
    return ix_arr[:n_points], iy_arr[:n_points], iz_arr[:n_points], weight_arr[:n_points]


@njit(cache=True, parallel=True, fastmath=True)
def compute_G_monte_carlo(n_photons, kappa, beta, nx, ny, nz, dx, dy, dz):
    """
    使用蒙特卡洛计算G(x) - 纯吸收情况
    """
    G = np.zeros((nz, ny, nx), dtype=np.float64)
    
    for i in prange(n_photons):
        ix_arr, iy_arr, iz_arr, w_arr = track_photon_pure_absorption(
            kappa, beta, nx, ny, nz, dx, dy, dz
        )
        for j in range(len(ix_arr)):
            G[iz_arr[j], iy_arr[j], ix_arr[j]] += w_arr[j]
    
    return G


# ============================================================================
# 精确解计算（数值积分）
# ============================================================================

def compute_G_exact(x, y, z, kappa=5.0):
    """
    计算G的精确值（通过数值积分）
    
    G(r) = ∫∫∫ S(r') * exp(-β|r-r'|) / |r-r'|² * (β/4π) dV'
    
    对于纯吸收，积分辐射强度满足 Beer-Lambert 衰减
    """
    beta = kappa
    
    def integrand(xp, yp, zp):
        # 源项
        rp = np.sqrt((xp-0.5)**2 + (yp-0.5)**2 + (zp-0.5)**2)
        Sp = max(0, 1 - 5*rp)
        if Sp <= 0:
            return 0.0
        
        # 距离
        dist = np.sqrt((x-xp)**2 + (y-yp)**2 + (z-zp)**2)
        if dist < 1e-10:
            return 0.0
        
        # 核函数（考虑几何衰减和吸收）
        kernel = np.exp(-beta * dist) / (4 * np.pi * dist**2) * beta
        return Sp * kernel
    
    # 数值积分（仅在源区域内积分）
    # 源区域：r < 0.2，即 (x-0.5)² + (y-0.5)² + (z-0.5)² < 0.04
    result, error = integrate.tplquad(
        integrand,
        0.3, 0.7,  # x范围（近似覆盖源球）
        lambda x: 0.3, lambda x: 0.7,  # y范围
        lambda x, y: 0.3, lambda x, y: 0.7  # z范围
    )
    
    return result


def compute_G_exact_on_grid(nx=51, ny=51, nz=51, kappa=5.0):
    """在网格上计算精确解"""
    print("Computing exact solution on grid...")
    G_exact = np.zeros((nz, ny, nx))
    
    x = np.linspace(0.5/nx, 1-0.5/nx, nx)
    y = np.linspace(0.5/ny, 1-0.5/ny, ny)
    z = np.linspace(0.5/nz, 1-0.5/nz, nz)
    
    for i in range(nx):
        if i % 10 == 0:
            print(f"  Progress: {i}/{nx}")
        for j in range(ny):
            for k in range(nz):
                G_exact[k, j, i] = compute_G_exact(x[i], y[j], z[k], kappa)
    
    return G_exact, x, y, z


# ============================================================================
# 简化精确解：沿中心线
# ============================================================================

def compute_G_exact_centerline(n_points=51, kappa=5.0):
    """计算沿x轴中心线的精确解"""
    print("Computing exact solution along centerline...")
    
    x = np.linspace(0, 1, n_points)
    y = z = 0.5
    G_exact = np.zeros(n_points)
    
    for i, xi in enumerate(x):
        G_exact[i] = compute_G_exact(xi, y, z, kappa)
        if i % 10 == 0:
            print(f"  x={xi:.2f}, G={G_exact[i]:.6f}")
    
    return x, G_exact


# ============================================================================
# 主运行函数
# ============================================================================

def run_monte_carlo_case_a():
    """运行MC并对比精确解"""
    print("="*70)
    print("Monte Carlo 3D RTE Solver - Case A (Pure Absorption)")
    print("="*70)
    print(f"Physics: κ={KAPPA}, σs={SIGMA_S}, β={BETA}")
    print(f"Grid: {NX}x{NY}x{NZ}, Photons: {N_PHOTONS:,}")
    print("="*70)
    
    dx = dy = dz = 1.0 / NX
    cell_volume = dx ** 3
    
    # 源积分
    Q_total = np.pi / 150.0
    print(f"\nSource integral Q_total = {Q_total:.6f} (π/150)")
    
    # 运行MC
    print("\n[1] Computing G using Monte Carlo (path integral estimator)...")
    start = time.time()
    
    G_raw = compute_G_monte_carlo(
        N_PHOTONS, KAPPA, BETA,
        NX, NY, NZ, dx, dy, dz
    )
    
    # 归一化
    G_mc = G_raw * (Q_total / N_PHOTONS) / cell_volume
    
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    
    # MC结果
    iz, iy, ix = NZ//2, NY//2, NX//2
    print(f"\n[2] MC Results:")
    print(f"  G range: [{G_mc.min():.4f}, {G_mc.max():.4f}]")
    print(f"  G at center (0.5, 0.5, 0.5): {G_mc[iz,iy,ix]:.4f}")
    print(f"  G at face center (0, 0.5, 0.5): {G_mc[0,iy,ix]:.4f}")
    print(f"  G at corner (0, 0, 0): {G_mc[0,0,0]:.4f}")
    
    # 计算精确解（中心线）
    print("\n[3] Computing exact solution ( Beer-Lambert )...")
    # 对于纯吸收，中心点的G可以用Beer-Lambert定律估算
    # G_center ≈ Q_total * exp(-β * r_eff) / (4π * r_eff²)
    # 其中r_eff是有效距离
    
    # 简化：沿中心线的精确解
    x_line = np.linspace(0.5/NX, 1-0.5/NX, NX)
    G_exact_line = np.zeros(NX)
    
    for i, xi in enumerate(x_line):
        G_exact_line[i] = compute_G_exact(xi, 0.5, 0.5, KAPPA)
    
    print(f"  Exact G at center (0.5, 0.5, 0.5): {G_exact_line[NX//2]:.4f}")
    print(f"  Exact G at face (0, 0.5, 0.5): {G_exact_line[0]:.4f}")
    
    # 对比
    print("\n[4] Comparison:")
    print(f"  Center: MC={G_mc[iz,iy,ix]:.4f}, Exact={G_exact_line[NX//2]:.4f}, Error={abs(G_mc[iz,iy,ix]-G_exact_line[NX//2])/G_exact_line[NX//2]*100:.2f}%")
    print(f"  Face:   MC={G_mc[0,iy,ix]:.4f}, Exact={G_exact_line[0]:.4f}, Error={abs(G_mc[0,iy,ix]-G_exact_line[0])/G_exact_line[0]*100:.2f}%")
    
    # 保存结果
    output_dir = 'MC3D_Results'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'MC_G_3D_A_CaseA_PureAbsorption.npz')
    np.savez(output_file,
             G=G_mc,
             G_exact=G_exact_line,
             x=np.linspace(0.5*dx, 1-0.5*dx, NX),
             y=np.linspace(0.5*dy, 1-0.5*dy, NY),
             z=np.linspace(0.5*dz, 1-0.5*dz, NZ),
             kappa=KAPPA,
             sigma_s=SIGMA_S,
             g=G_HG,
             n_photons=N_PHOTONS)
    
    print(f"\n[5] Results saved to: {output_file}")
    
    # 绘图
    plot_results_case_a(G_mc, G_exact_line, x_line, output_dir)
    
    return G_mc, G_exact_line


def plot_results_case_a(G_mc, G_exact, x_line, output_dir):
    """绘制Case A结果对比"""
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 11
    
    iz, iy, ix = NZ//2, NY//2, NX//2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 坐标
    x = np.linspace(0.5/NX, 1-0.5/NX, NX)
    y = np.linspace(0.5/NY, 1-0.5/NY, NY)
    z_coord = np.linspace(0.5/NZ, 1-0.5/NZ, NZ)
    
    # MC结果 - 三个截面
    vmin, vmax = 0, max(G_mc.max(), G_exact.max()) * 1.1
    
    # Z中间截面
    ax = axes[0, 0]
    im = ax.contourf(x, y, G_mc[iz, :, :].T, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(f'MC: G(x,y,0.5)')
    plt.colorbar(im, ax=ax)
    
    # Y中间截面
    ax = axes[0, 1]
    im = ax.contourf(x, z_coord, G_mc[:, iy, :].T, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$z$')
    ax.set_title(f'MC: G(x,0.5,z)')
    plt.colorbar(im, ax=ax)
    
    # X中间截面
    ax = axes[0, 2]
    im = ax.contourf(y, z_coord, G_mc[:, :, ix].T, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$z$')
    ax.set_title(f'MC: G(0.5,y,z)')
    plt.colorbar(im, ax=ax)
    
    # 中心线对比
    ax = axes[1, 0]
    ax.plot(x, G_mc[:, iy, iz], 'b-', linewidth=2, label='MC (Path Integral)')
    ax.plot(x_line, G_exact, 'r--', linewidth=2, label='Exact (Numerical)')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$G(x, 0.5, 0.5)$')
    ax.set_title('Centerline Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 相对误差
    ax = axes[1, 1]
    rel_error = np.abs(G_mc[:, iy, iz] - G_exact) / (np.abs(G_exact) + 1e-10) * 100
    ax.semilogy(x, rel_error, 'g-', linewidth=2)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'Relative Error (%)')
    ax.set_title('Relative Error along Centerline')
    ax.grid(True, alpha=0.3, which='both')
    
    # 统计信息
    ax = axes[1, 2]
    ax.axis('off')
    
    # 计算整体误差统计
    diff = G_mc[:, iy, iz] - G_exact
    l1_error = np.mean(np.abs(diff))
    l2_error = np.sqrt(np.mean(diff**2))
    linf_error = np.max(np.abs(diff))
    
    info_text = f"""
Case A - Pure Absorption Validation

Physics:
  κ = {KAPPA}
  σs = {SIGMA_S}
  β = {BETA}

MC Results:
  G_center = {G_mc[iz,iy,ix]:.4f}
  G_face = {G_mc[0,iy,ix]:.4f}
  G_max = {G_mc.max():.4f}

Exact Solution:
  G_center = {G_exact[NX//2]:.4f}
  G_face = {G_exact[0]:.4f}

Errors (Centerline):
  L1 error = {l1_error:.4e}
  L2 error = {l2_error:.4e}
  L∞ error = {linf_error:.4e}
  
MC Config:
  N_photons = {N_PHOTONS:,}
  Grid = {NX}×{NY}×{NZ}
    """
    ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Monte Carlo vs Exact Solution - Case A (Pure Absorption)', fontsize=14)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'MC_G_3D_A_CaseA_Validation.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to: {plot_file}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monte Carlo 3D RTE Solver for Case A (Validation)')
    parser.add_argument('--photons', type=int, default=N_PHOTONS,
                       help='Number of photons')
    args = parser.parse_args()
    
    N_PHOTONS = args.photons
    G_mc, G_exact = run_monte_carlo_case_a()
    
    print("\n" + "="*70)
    print("Validation completed!")
    print("="*70)
