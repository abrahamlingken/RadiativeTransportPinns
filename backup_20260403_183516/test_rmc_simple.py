#!/usr/bin/env python
"""
简化版RMC测试 - 验证算法正确性
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

# 简化参数
N_RAY = 10000
NX, NY, NZ = 21, 21, 21
XLEN, YLEN, ZLEN = 1.0, 1.0, 1.0
KAPPA = 5.0
SIGMA_S = 0.0  # 纯吸收
CENTER = np.array([0.5, 0.5, 0.5])


def source_term(x, y, z):
    r = np.sqrt((x - CENTER[0])**2 + (y - CENTER[1])**2 + (z - CENTER[2])**2)
    return np.maximum(0.0, 1.0 - 5.0 * r)


@jit(nopython=True, cache=True)
def intersect_boundary(x0, y0, z0, ux, uy, uz, xbound, ybound, zbound):
    """简化版边界求交"""
    n = len(x0)
    xyzb = np.empty((n, 3))
    distL = np.empty(n)
    
    for i in range(n):
        # 计算到六个边界的距离
        tx = 1e10
        ty = 1e10
        tz = 1e10
        
        if abs(ux[i]) > 1e-14:
            if ux[i] > 0:
                tx = (XLEN - x0[i]) / ux[i]
            else:
                tx = -x0[i] / ux[i]
        
        if abs(uy[i]) > 1e-14:
            if uy[i] > 0:
                ty = (YLEN - y0[i]) / uy[i]
            else:
                ty = -y0[i] / uy[i]
        
        if abs(uz[i]) > 1e-14:
            if uz[i] > 0:
                tz = (ZLEN - z0[i]) / uz[i]
            else:
                tz = -z0[i] / uz[i]
        
        t_min = min(abs(tx), abs(ty), abs(tz))
        
        xyzb[i, 0] = x0[i] + ux[i] * t_min
        xyzb[i, 1] = y0[i] + uy[i] * t_min
        xyzb[i, 2] = z0[i] + uz[i] * t_min
        distL[i] = t_min
        
    return xyzb, distL


def simple_mc():
    """简化版蒙特卡洛 - 纯吸收介质"""
    print("Running simplified MC test...")
    
    ke = KAPPA  # 纯吸收，消光系数=吸收系数
    
    # 探测器位置
    xyz0 = np.array([0.5, 0.5, 0.5])
    
    # 统计网格
    dx, dy, dz = XLEN/NX, YLEN/NY, ZLEN/NZ
    energy = np.zeros((NX, NY, NZ))
    
    start = time.time()
    
    for ray in range(N_RAY):
        if ray % 2000 == 0:
            print(f"  Ray {ray}/{N_RAY}")
        
        # 随机方向（各向同性）
        theta = np.arccos(1.0 - 2.0 * np.random.rand())
        phi = 2.0 * np.pi * np.random.rand()
        ux = np.sin(theta) * np.cos(phi)
        uy = np.sin(theta) * np.sin(phi)
        uz = np.cos(theta)
        
        # 当前位置
        x, y, z = xyz0[0], xyz0[1], xyz0[2]
        
        # 追踪射线直到边界
        while True:
            # 计算到边界的距离
            xyzb, dis = intersect_boundary(
                np.array([x]), np.array([y]), np.array([z]),
                np.array([ux]), np.array([uy]), np.array([uz]),
                None, None, None
            )
            
            # 采样随机光学厚度
            xi = np.random.rand()
            rand_opt_thi = -np.log(1.0 - xi)
            
            # 到边界的光学厚度
            opt_thi = ke * dis[0]
            
            if rand_opt_thi > opt_thi:
                # 到达边界 - 记录出射位置
                x_b, y_b, z_b = xyzb[0]
                ix = min(int(x_b / dx), NX-1)
                iy = min(int(y_b / dy), NY-1)
                iz = min(int(z_b / dz), NZ-1)
                energy[ix, iy, iz] += 1
                break
            else:
                # 在介质内被吸收（纯吸收情况）
                # 计算吸收位置
                x_abs = x + rand_opt_thi / ke * ux
                y_abs = y + rand_opt_thi / ke * uy
                z_abs = z + rand_opt_thi / ke * uz
                
                # 检查是否在源区域内
                if source_term(x_abs, y_abs, z_abs) > 0:
                    ix = min(int(x_abs / dx), NX-1)
                    iy = min(int(y_abs / dy), NY-1)
                    iz = min(int(z_abs / dz), NZ-1)
                    energy[ix, iy, iz] += 1
                
                # 纯吸收介质，射线终止
                break
    
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f}s")
    
    return energy / N_RAY


def plot_result(G):
    """绘制结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ix, iy, iz = NX//2, NY//2, NZ//2
    
    # x-y截面
    im1 = axes[0].imshow(G[:, :, iz].T, origin='lower', cmap='hot')
    axes[0].set_title(f'z={iz} slice')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # x-z截面
    im2 = axes[1].imshow(G[:, iy, :].T, origin='lower', cmap='hot')
    axes[1].set_title(f'y={iy} slice')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    plt.colorbar(im2, ax=axes[1])
    
    # y-z截面
    im3 = axes[2].imshow(G[ix, :, :].T, origin='lower', cmap='hot')
    axes[2].set_title(f'x={ix} slice')
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('z')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('test_rmc_result.png', dpi=150)
    print("Saved: test_rmc_result.png")
    print(f"G max: {G.max():.4f}, G min: {G.min():.4f}")
    print(f"Center G: {G[ix, iy, iz]:.4f}")


if __name__ == "__main__":
    G = simple_mc()
    plot_result(G)
