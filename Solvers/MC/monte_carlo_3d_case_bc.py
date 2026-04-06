#!/usr/bin/env python
"""
Monte Carlo 3D RTE Solver for Case B/C
======================================
匹配PINN参数：
- Case B: κ=0.5, σs=4.5, g=0.0 (各向同性散射)
- Case C: κ=0.5, σs=4.5, g=0.8 (前向散射)

光源: S(r) = max(0, 1-5r), 仅在r<0.2区域
"""

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import sys
import time

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# 配置参数
# ============================================================================
CASE = 'B'  # 改为'C'运行Case C
NX, NY, NZ = 51, 51, 51  # 与PINN一致
N_PHOTONS = 5_000_000    # 光子数（可根据精度需求调整）
BATCH_SIZE = 50_000

# 物理参数（与PINN Case B/C一致）
PHYSICS = {
    'B': {'kappa': 0.5, 'sigma_s': 4.5, 'g': 0.0, 'name': 'CaseB_Isotropic'},
    'C': {'kappa': 0.5, 'sigma_s': 4.5, 'g': 0.8, 'name': 'CaseC_Forward'}
}

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
def sample_hg_direction(sx, sy, sz, g):
    """Henyey-Greenstein散射方向采样"""
    if abs(g) < 1e-10:
        return sample_isotropic_direction()
    
    # HG相函数采样
    if g > 0.999:
        cos_theta = 1.0 - 1e-6 * np.random.random()
    else:
        s = (1.0 - g * g) / (1.0 - g + 2.0 * g * np.random.random())
        cos_theta = (1.0 + g * g - s * s) / (2.0 * g)
        cos_theta = max(-1.0, min(1.0, cos_theta))
    
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
    phi = 2.0 * np.pi * np.random.random()
    
    # 建立局部坐标系
    if abs(sz) < 0.999:
        ux, uy = -sy, sx
        norm = np.sqrt(ux*ux + uy*uy)
        ux, uy = ux/norm, uy/norm
        uz = 0.0
    else:
        ux, uy, uz = 1.0, 0.0, 0.0
    
    vx = sy * uz - sz * uy
    vy = sz * ux - sx * uz
    vz = sx * uy - sy * ux
    
    sx_new = sin_theta * np.cos(phi) * ux + sin_theta * np.sin(phi) * vx + cos_theta * sx
    sy_new = sin_theta * np.cos(phi) * uy + sin_theta * np.sin(phi) * vy + cos_theta * sy
    sz_new = sin_theta * np.cos(phi) * uz + sin_theta * np.sin(phi) * vz + cos_theta * sz
    
    norm = np.sqrt(sx_new**2 + sy_new**2 + sz_new**2)
    return sx_new/norm, sy_new/norm, sz_new/norm


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
def track_photon(kappa, sigma_s, beta, g_hg, nx, ny, nz, dx, dy, dz):
    """
    追踪单个光子，使用碰撞估计器
    """
    # 初始化
    x, y, z = sample_source_point()
    sx, sy, sz = sample_isotropic_direction()
    
    # 网格索引数组和权重
    max_collisions = 1000
    ix_arr = np.zeros(max_collisions, dtype=np.int32)
    iy_arr = np.zeros(max_collisions, dtype=np.int32)
    iz_arr = np.zeros(max_collisions, dtype=np.int32)
    weight_arr = np.zeros(max_collisions, dtype=np.float64)
    
    n_coll = 0
    weight = source_term(x, y, z)
    
    while weight > 1e-10 and n_coll < max_collisions:
        # 计算到边界的距离
        l_boundary = distance_to_boundary(x, y, z, sx, sy, sz)
        
        # 采样光学厚度
        tau = -np.log(1.0 - np.random.random())
        l_collision = tau / beta if beta > 1e-15 else 1e10
        
        if l_collision < l_boundary:
            # 发生碰撞
            x += l_collision * sx
            y += l_collision * sy
            z += l_collision * sz
            
            # 记录碰撞（用于计算G）
            ix = int(x / dx)
            iy = int(y / dy)
            iz = int(z / dz)
            
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                ix_arr[n_coll] = ix
                iy_arr[n_coll] = iy
                iz_arr[n_coll] = iz
                # 权重：1/kappa 表示对G的贡献
                weight_arr[n_coll] = weight / kappa if kappa > 1e-15 else 0.0
                n_coll += 1
            
            # 判断吸收或散射
            if np.random.random() < (sigma_s / beta):
                # 散射，继续追踪
                sx, sy, sz = sample_hg_direction(sx, sy, sz, g_hg)
                # 权重保持不变（散射不损失能量）
            else:
                # 吸收，终止
                break
        else:
            # 逃逸出边界
            break
    
    return ix_arr[:n_coll], iy_arr[:n_coll], iz_arr[:n_coll], weight_arr[:n_coll]


@njit(cache=True, parallel=True, fastmath=True)
def compute_G_monte_carlo(n_photons, kappa, sigma_s, beta, g_hg, nx, ny, nz, dx, dy, dz):
    """
    使用蒙特卡洛计算G(x)
    """
    G = np.zeros((nz, ny, nx), dtype=np.float64)
    
    for i in prange(n_photons):
        ix_arr, iy_arr, iz_arr, w_arr = track_photon(
            kappa, sigma_s, beta, g_hg, nx, ny, nz, dx, dy, dz
        )
        for j in range(len(ix_arr)):
            G[iz_arr[j], iy_arr[j], ix_arr[j]] += w_arr[j]
    
    return G


def run_monte_carlo(case='B'):
    """主运行函数"""
    if case not in PHYSICS:
        raise ValueError(f"Unknown case: {case}. Available: {list(PHYSICS.keys())}")
    
    phys = PHYSICS[case]
    kappa = phys['kappa']
    sigma_s = phys['sigma_s']
    g_hg = phys['g']
    name = phys['name']
    beta = kappa + sigma_s
    
    print("="*70)
    print(f"Monte Carlo 3D RTE Solver - {name}")
    print("="*70)
    print(f"Physics: κ={kappa}, σs={sigma_s}, β={beta}, g={g_hg}")
    print(f"Grid: {NX}x{NY}x{NZ}, Photons: {N_PHOTONS:,}")
    print(f"Case: {case}")
    print("="*70)
    
    dx = dy = dz = 1.0 / NX
    cell_volume = dx ** 3
    
    # 源积分（归一化因子）
    # 对于源项 S(r) = max(0, 1-5r)，球内积分 = 4π * ∫[0,0.2] (1-5r) * r² dr = π/150
    Q_total = np.pi / 150.0
    
    print("\n[1] Computing G using Monte Carlo collision estimator...")
    start = time.time()
    
    G_raw = compute_G_monte_carlo(
        N_PHOTONS, kappa, sigma_s, beta, g_hg,
        NX, NY, NZ, dx, dy, dz
    )
    
    # 归一化: G = (Q_total / N_photons) * (1/V_cell) * G_raw
    G_normalized = G_raw * (Q_total / N_PHOTONS) / cell_volume
    
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  G range: [{G_normalized.min():.4f}, {G_normalized.max():.4f}]")
    
    # 中心点值
    iz, iy, ix = NZ//2, NY//2, NX//2
    print(f"  G at center: {G_normalized[iz,iy,ix]:.4f}")
    print(f"  G at face center: {G_normalized[0,iy,ix]:.4f}")
    
    # 保存结果
    output_dir = 'MC3D_Results'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'MC_G_3D_{case}_{name}.npz')
    np.savez(output_file,
             G=G_normalized,
             x=np.linspace(0.5*dx, 1-0.5*dx, NX),
             y=np.linspace(0.5*dy, 1-0.5*dy, NY),
             z=np.linspace(0.5*dz, 1-0.5*dz, NZ),
             kappa=kappa,
             sigma_s=sigma_s,
             g=g_hg,
             n_photons=N_PHOTONS)
    
    print(f"\n[2] Results saved to: {output_file}")
    
    # 绘图
    plot_results(G_normalized, case, name, output_dir)
    
    return G_normalized


def plot_results(G, case, name, output_dir):
    """绘制结果"""
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 11
    
    iz, iy, ix = NZ//2, NY//2, NX//2
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 坐标
    x = np.linspace(0.5/NX, 1-0.5/NX, NX)
    y = np.linspace(0.5/NY, 1-0.5/NY, NY)
    z = np.linspace(0.5/NZ, 1-0.5/NZ, NZ)
    
    # 三个正交截面
    slices = [
        (G[:, :, iz], 'x', 'y', x, y, f'z={z[iz]:.2f}'),
        (G[:, iy, :], 'x', 'z', x, z, f'y={y[iy]:.2f}'),
        (G[ix, :, :], 'y', 'z', y, z, f'x={x[ix]:.2f}')
    ]
    
    for idx, (data, xlab, ylab, xcoord, ycoord, title) in enumerate(slices):
        ax = axes[0, idx]
        im = ax.contourf(xcoord, ycoord, data.T, levels=20, cmap='hot')
        ax.set_xlabel(rf'${xlab}$')
        ax.set_ylabel(rf'${ylab}$')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    # 中心线
    ax = axes[1, 0]
    ax.plot(x, G[:, iy, iz], 'k-', linewidth=2, label='MC (Collision)')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$G(x, 0.5, 0.5)$')
    ax.set_title('Centerline along x')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 最大强度投影
    ax = axes[1, 1]
    max_proj = np.max(G, axis=2)
    im = ax.contourf(x, y, max_proj.T, levels=20, cmap='hot')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('Max Intensity Projection (along z)')
    plt.colorbar(im, ax=ax)
    
    # 统计信息
    ax = axes[1, 2]
    ax.axis('off')
    info_text = f"""
    Case: {case} - {name}
    
    Physics:
      κ = {PHYSICS[case]['kappa']}
      σs = {PHYSICS[case]['sigma_s']}
      g = {PHYSICS[case]['g']}
    
    Results:
      G_center = {G[iz,iy,ix]:.4f}
      G_max = {G.max():.4f}
      G_min = {G.min():.4f}
    
    MC Config:
      N_photons = {N_PHOTONS:,}
      Grid = {NX}×{NY}×{NZ}
    """
    ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.suptitle(f'Monte Carlo Results - 3D {name}', fontsize=14)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f'MC_G_3D_{case}_{name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to: {plot_file}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monte Carlo 3D RTE Solver for Case B/C')
    parser.add_argument('--case', type=str, default='B', choices=['B', 'C'],
                       help='Case to run: B (isotropic) or C (forward scattering)')
    parser.add_argument('--photons', type=int, default=N_PHOTONS,
                       help='Number of photons')
    args = parser.parse_args()
    
    N_PHOTONS = args.photons
    G = run_monte_carlo(args.case)
    
    print("\n" + "="*70)
    print("Monte Carlo completed!")
    print("="*70)
