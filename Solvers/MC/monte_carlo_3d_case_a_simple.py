#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte Carlo 3D RTE Solver for Case A (Pure Absorption) - Simplified
===================================================================
验证MC方法正确性，使用Beer-Lambert解析近似

Physics:
- Pure absorption: κ = 5.0, σs = 0.0, β = 5.0
- Source: S(r) = max(0, 1-5r), r < 0.2

Expected (from Beer-Lambert law and previous PINN results):
- G_center ≈ 0.92 (from PINN documentation)
"""

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import os
import sys
import time

# 路径设置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# 配置
# ============================================================================
NX, NY, NZ = 51, 51, 51
KAPPA = 5.0
BETA = 5.0

# ============================================================================
# 核心MC函数（Numba加速）
# ============================================================================

@njit(cache=True, fastmath=True)
def source_term(x, y, z):
    """源项 S(r) = max(0, 1-5r)"""
    r = np.sqrt((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)
    s = 1.0 - 5.0 * r
    return s if s > 0.0 else 0.0


@njit(cache=True, fastmath=True)
def sample_source_point():
    """拒绝采样源点"""
    while True:
        x, y, z = np.random.random(3)
        if np.random.random() < source_term(x, y, z):
            return x, y, z


@njit(cache=True, fastmath=True)
def sample_isotropic_direction():
    """各向同性方向"""
    cos_theta = 2.0 * np.random.random() - 1.0
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
    phi = 2.0 * np.pi * np.random.random()
    return sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta


@njit(cache=True, fastmath=True)
def distance_to_boundary(x, y, z, sx, sy, sz):
    """到边界距离"""
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
def track_photon_collision_estimator(beta, nx, ny, nz, dx, dy, dz):
    """
    碰撞估计器：纯吸收介质中，光子被吸收的地点对G有贡献
    """
    x, y, z = sample_source_point()
    sx, sy, sz = sample_isotropic_direction()
    
    # 到边界距离
    l_boundary = distance_to_boundary(x, y, z, sx, sy, sz)
    
    # 采样吸收位置（纯吸收，直接采样光学厚度）
    tau = -np.log(1.0 - np.random.random())
    l_absorb = tau / beta
    
    if l_absorb < l_boundary:
        # 在介质内被吸收
        x += l_absorb * sx
        y += l_absorb * sy
        z += l_absorb * sz
        
        ix = int(x / dx)
        iy = int(y / dy)
        iz = int(z / dz)
        
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            # 权重 = 初始源强度 / kappa
            weight = source_term(x - l_absorb*sx, y - l_absorb*sy, z - l_absorb*sz) / KAPPA
            return ix, iy, iz, weight
    
    # 逃逸或无贡献
    return -1, -1, -1, 0.0


@njit(cache=True, parallel=True, fastmath=True)
def compute_G_mc(n_photons, nx, ny, nz, dx, dy, dz):
    """MC计算G"""
    G = np.zeros((nz, ny, nx), dtype=np.float64)
    
    for i in prange(n_photons):
        ix, iy, iz, w = track_photon_collision_estimator(BETA, nx, ny, nz, dx, dy, dz)
        if ix >= 0:
            G[iz, iy, ix] += w
    
    return G


def run_mc(n_photons=2000000):
    """主函数"""
    print("="*60)
    print("Case A - Pure Absorption (MC Validation)")
    print("="*60)
    print(f"Photons: {n_photons:,}")
    print(f"Grid: {NX}x{NY}x{NZ}, κ={KAPPA}")
    
    dx = dy = dz = 1.0 / NX
    cell_volume = dx ** 3
    
    # 源积分 = π/150
    Q_total = np.pi / 150.0
    print(f"Source integral Q = {Q_total:.6f}")
    
    # 运行MC
    print("\nRunning MC...")
    start = time.time()
    G_raw = compute_G_mc(n_photons, NX, NY, NZ, dx, dy, dz)
    
    # 归一化
    G = G_raw * (Q_total / n_photons) / cell_volume
    
    print(f"Completed in {time.time()-start:.1f}s")
    
    # 结果
    iz, iy, ix = NZ//2, NY//2, NX//2
    print(f"\nResults:")
    print(f"  G_center = {G[iz,iy,ix]:.4f}")
    print(f"  G_face = {G[0,iy,ix]:.4f}")
    print(f"  G_max = {G.max():.4f}")
    print(f"  G_min = {G.min():.4f}")
    
    # 与预期值对比（来自PINN文档）
    G_expected_center = 0.92
    error = abs(G[iz,iy,ix] - G_expected_center) / G_expected_center * 100
    print(f"\nValidation:")
    print(f"  Expected G_center ≈ {G_expected_center:.2f} (from PINN docs)")
    print(f"  MC G_center = {G[iz,iy,ix]:.4f}")
    print(f"  Relative error = {error:.2f}%")
    
    # 保存
    os.makedirs('MC3D_Results', exist_ok=True)
    np.savez('MC3D_Results/MC_G_3D_A_CaseA.npz',
             G=G,
             x=np.linspace(0.5*dx, 1-0.5*dx, NX),
             y=np.linspace(0.5*dy, 1-0.5*dy, NY),
             z=np.linspace(0.5*dz, 1-0.5*dz, NZ),
             kappa=KAPPA, sigma_s=0.0, g=0.0)
    
    print("\nSaved to: MC3D_Results/MC_G_3D_A_CaseA.npz")
    
    # 简单绘图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.contourf(G[iz, :, :], levels=20, cmap='hot')
    plt.colorbar(label='G')
    plt.title('G(x,y,0.5)')
    
    plt.subplot(132)
    x = np.linspace(0, 1, NX)
    plt.plot(x, G[:, iy, iz], 'b-', label='MC')
    plt.axhline(y=G_expected_center, color='r', linestyle='--', label=f'Expected ≈{G_expected_center}')
    plt.xlabel('x')
    plt.ylabel('G(x,0.5,0.5)')
    plt.legend()
    plt.title('Centerline')
    
    plt.subplot(133)
    plt.text(0.1, 0.5, f"Case A - Pure Absorption\n\n"
                       f"κ = {KAPPA}, σs = 0\n\n"
                       f"G_center = {G[iz,iy,ix]:.4f}\n"
                       f"Expected = {G_expected_center:.2f}\n"
                       f"Error = {error:.2f}%", 
             fontsize=12, family='monospace', verticalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('MC3D_Results/MC_G_3D_A_CaseA.png', dpi=200)
    print("Plot saved to: MC3D_Results/MC_G_3D_A_CaseA.png")
    
    return G


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--photons', type=int, default=2000000)
    args = parser.parse_args()
    
    run_mc(args.photons)
