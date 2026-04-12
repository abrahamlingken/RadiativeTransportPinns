#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte Carlo 3D RTE Solver for Case A (Pure Absorption) - Fixed Version
======================================================================
验证MC方法正确性

关键修正：
- G = ∫ I dΩ (积分辐射强度)
- 对于纯吸收，使用发射密度估计器或路径积分估计器
"""

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# 配置
# ============================================================================
NX, NY, NZ = 51, 51, 51
KAPPA = 5.0
BETA = 5.0

# ============================================================================
# 核心MC函数
# ============================================================================

@njit(cache=True, fastmath=True)
def source_term(x, y, z):
    """源项 S(r) = max(0, 1-5r)"""
    r = np.sqrt((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)
    s = 1.0 - 5.0 * r
    return s if s > 0.0 else 0.0


@njit(cache=True, fastmath=True)
def sample_source_point():
    """在源区域内采样"""
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
def track_photon_path_integral(beta, nx, ny, nz, dx, dy, dz):
    """
    路径积分估计器：记录光子对路径上所有格点的贡献
    
    对于纯吸收，光子从源出发沿直线传播，强度按Beer-Lambert衰减
    G的贡献 = ∫ I(s) ds / (某个归一化)
    """
    x, y, z = sample_source_point()
    sx, sy, sz = sample_isotropic_direction()
    
    # 初始强度
    I0 = source_term(x, y, z)
    
    # 到边界距离
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
    
    # 沿路径记录贡献
    max_points = 100
    ix_arr = np.zeros(max_points, dtype=np.int32)
    iy_arr = np.zeros(max_points, dtype=np.int32)
    iz_arr = np.zeros(max_points, dtype=np.int32)
    weight_arr = np.zeros(max_points, dtype=np.float64)
    n_points = 0
    
    # 步进参数
    dl = min(dx, dy, dz) * 0.5  # 半步长确保覆盖
    n_steps = min(int(l_boundary / dl) + 1, 200)
    
    for i in range(n_steps):
        s = (i + 0.5) * dl
        if s >= l_boundary:
            break
            
        xi = x + s * sx
        yi = y + s * sy
        zi = z + s * sz
        
        ix = int(xi / dx)
        iy = int(yi / dy)
        iz = int(zi / dz)
        
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            # Beer-Lambert衰减
            attenuation = np.exp(-beta * s)
            # 权重 = 初始强度 * 衰减 * 步长
            weight = I0 * attenuation * dl
            
            if n_points < max_points:
                ix_arr[n_points] = ix
                iy_arr[n_points] = iy
                iz_arr[n_points] = iz
                weight_arr[n_points] = weight
                n_points += 1
    
    return ix_arr[:n_points], iy_arr[:n_points], iz_arr[:n_points], weight_arr[:n_points]


@njit(cache=True, parallel=True, fastmath=True)
def compute_G_path_integral(n_photons, nx, ny, nz, dx, dy, dz):
    """路径积分估计器计算G"""
    G = np.zeros((nz, ny, nx), dtype=np.float64)
    
    for i in prange(n_photons):
        ix_arr, iy_arr, iz_arr, w_arr = track_photon_path_integral(
            BETA, nx, ny, nz, dx, dy, dz
        )
        for j in range(len(ix_arr)):
            G[iz_arr[j], iy_arr[j], ix_arr[j]] += w_arr[j]
    
    return G


def compute_G_emission_density(n_photons=5000000):
    """
    发射密度估计器（另一种方法）
    
    对于纯吸收，G(r) = S(r) / kappa
    这是局部平衡近似
    """
    print("\n[Method 2] Emission Density Estimator")
    
    dx = dy = dz = 1.0 / NX
    G = np.zeros((NZ, NY, NX))
    
    # 采样大量点估计源分布
    n_samples = n_photons
    for _ in range(n_samples):
        x, y, z = np.random.random(3)
        s = source_term(x, y, z)
        if s > 0:
            ix = int(x / dx)
            iy = int(y / dy)
            iz = int(z / dz)
            if 0 <= ix < NX and 0 <= iy < NY and 0 <= iz < NZ:
                G[iz, iy, ix] += s
    
    # 归一化并除以kappa
    cell_volume = dx ** 3
    G = G / n_samples / cell_volume / KAPPA
    
    return G


def run_mc(n_photons=5000000):
    """主函数"""
    print("="*60)
    print("Case A - Pure Absorption (MC Validation - Fixed)")
    print("="*60)
    print(f"Photons: {n_photons:,}")
    print(f"Grid: {NX}x{NY}x{NZ}, κ={KAPPA}")
    
    dx = dy = dz = 1.0 / NX
    
    # 方法1：路径积分估计器
    print("\n[Method 1] Path Integral Estimator")
    print("Running MC...")
    start = time.time()
    G_path = compute_G_path_integral(n_photons, NX, NY, NZ, dx, dy, dz)
    
    # 归一化：需要4π因子，因为各向同性辐射
    # G = 4π * I，而我们在追踪单个方向
    G_path = G_path * 4.0 * np.pi / n_photons
    
    print(f"Completed in {time.time()-start:.1f}s")
    
    # 方法2：发射密度估计器
    G_emission = compute_G_emission_density(n_photons)
    
    # 结果对比
    iz, iy, ix = NZ//2, NY//2, NX//2
    
    print(f"\nResults:")
    print(f"  Path Integral:")
    print(f"    G_center = {G_path[iz,iy,ix]:.4f}")
    print(f"    G_face = {G_path[0,iy,ix]:.4f}")
    
    print(f"  Emission Density:")
    print(f"    G_center = {G_emission[iz,iy,ix]:.4f}")
    print(f"    G_face = {G_emission[0,iy,ix]:.4f}")
    
    # 理论估算
    # 对于中心点， Beer-Lambert近似：
    # G ≈ S_avg * exp(-β*r_avg) / (κ * V_eff)
    S_total = np.pi / 150.0  # 源积分
    r_avg = 0.1  # 平均距离
    attenuation = np.exp(-BETA * r_avg)
    G_theory = S_total * attenuation * 4 * np.pi  # 4π来自角度积分
    
    print(f"\nTheoretical Estimate:")
    print(f"  G_center ≈ {G_theory:.4f}")
    
    # 与参考值对比
    G_expected = 0.92
    error_path = abs(G_path[iz,iy,ix] - G_expected) / G_expected * 100
    error_emission = abs(G_emission[iz,iy,ix] - G_expected) / G_expected * 100
    
    print(f"\nValidation (Expected G_center ≈ {G_expected}):")
    print(f"  Path Integral Error = {error_path:.2f}%")
    print(f"  Emission Density Error = {error_emission:.2f}%")
    
    # 保存结果
    os.makedirs('MC3D_Results', exist_ok=True)
    np.savez('MC3D_Results/MC_G_3D_A_CaseA_Fixed.npz',
             G_path=G_path,
             G_emission=G_emission,
             x=np.linspace(0.5*dx, 1-0.5*dx, NX),
             y=np.linspace(0.5*dy, 1-0.5*dy, NY),
             z=np.linspace(0.5*dz, 1-0.5*dz, NZ),
             kappa=KAPPA, sigma_s=0.0, g=0.0)
    
    # 绘图
    plt.figure(figsize=(15, 4))
    
    plt.subplot(141)
    plt.contourf(G_path[iz, :, :], levels=20, cmap='hot')
    plt.colorbar(label='G')
    plt.title('Path Integral')
    
    plt.subplot(142)
    plt.contourf(G_emission[iz, :, :], levels=20, cmap='hot')
    plt.colorbar(label='G')
    plt.title('Emission Density')
    
    plt.subplot(143)
    x = np.linspace(0, 1, NX)
    plt.plot(x, G_path[:, iy, iz], 'b-', label='Path')
    plt.plot(x, G_emission[:, iy, iz], 'g--', label='Emission')
    plt.axhline(y=G_expected, color='r', linestyle=':', label=f'Expected={G_expected}')
    plt.xlabel('x')
    plt.ylabel('G')
    plt.legend()
    plt.title('Centerline')
    
    plt.subplot(144)
    plt.text(0.1, 0.5, f"Case A - Pure Absorption\n\n"
                       f"Path: G={G_path[iz,iy,ix]:.3f}, err={error_path:.1f}%\n"
                       f"Emission: G={G_emission[iz,iy,ix]:.3f}, err={error_emission:.1f}%\n"
                       f"Expected: {G_expected}", 
             fontsize=11, family='monospace', verticalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('MC3D_Results/MC_G_3D_A_CaseA_Fixed.png', dpi=200)
    print("\nPlot saved to: MC3D_Results/MC_G_3D_A_CaseA_Fixed.png")
    
    return G_path, G_emission


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--photons', type=int, default=5000000)
    args = parser.parse_args()
    
    run_mc(args.photons)
