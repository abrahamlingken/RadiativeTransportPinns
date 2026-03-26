#!/usr/bin/env python
"""Monte Carlo 3D RTE Benchmark Solver v3 - Consistent with PINN"""

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter
import os
import sys
import time

# Configuration
CASE = 'B'
NX, NY, NZ = 40, 40, 40
N_PHOTONS = 10_000_000
BATCH_SIZE = 100_000
APPLY_SMOOTHING = True
SMOOTHING_SIGMA = 0.8

@njit(cache=True, fastmath=True)
def source_term(x, y, z):
    d = np.sqrt((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)
    val = 1.0 - 2.0 * d
    return val if val > 0.0 else 0.0

@njit(cache=True, fastmath=True)
def sample_source_point():
    while True:
        x, y, z = np.random.random(), np.random.random(), np.random.random()
        if np.random.random() < source_term(x, y, z):
            return x, y, z

@njit(cache=True, fastmath=True)
def sample_isotropic_direction():
    cos_theta = 2.0 * np.random.random() - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * np.random.random()
    sx = sin_theta * np.cos(phi)
    sy = sin_theta * np.sin(phi)
    sz = cos_theta
    return sx, sy, sz

@njit(cache=True, fastmath=True)
def sample_hg_direction(sx, sy, sz, g):
    if abs(g) < 1e-10:
        return sample_isotropic_direction()
    if g > 0.999:
        cos_theta = 1.0 - 1e-6 * np.random.random()
    else:
        s = (1.0 - g * g) / (1.0 - g + 2.0 * g * np.random.random())
        cos_theta = (1.0 + g * g - s * s) / (2.0 * g)
        cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
    phi = 2.0 * np.pi * np.random.random()
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

# ============================================================================
# KEY FIX: Use track length estimator with correct normalization
# ============================================================================

@njit(cache=True, fastmath=True)
def track_photon_for_G(kappa, sigma_s, beta, g_hg, nx, ny, nz, dx, dy, dz):
    """
    Track photon and accumulate G using track length estimator.
    
    For RTE: G(x) = ∫ I(x,s) dΩ
    
    Track length estimator: 
    - Each photon with weight W contributes W × (optical_path_length) to G
    - Optical path = physical_path × beta
    - G = Σ W × path × beta / V_cell
    
    But we need to be careful about the normalization. The factor 4π comes from
    the angular integration, not the track length itself.
    
    Correct normalization:
    G = (Q_total/N) × (1/V_cell) × Σ(path_length × beta)
    
    WITHOUT the extra 4π factor that was in v2!
    """
    x, y, z = sample_source_point()
    sx, sy, sz = sample_isotropic_direction()
    
    max_segments = 100
    ix_arr = np.zeros(max_segments, dtype=np.int32)
    iy_arr = np.zeros(max_segments, dtype=np.int32)
    iz_arr = np.zeros(max_segments, dtype=np.int32)
    optical_path_arr = np.zeros(max_segments, dtype=np.float64)
    n_seg = 0
    
    while n_seg < max_segments:
        l_boundary = distance_to_boundary(x, y, z, sx, sy, sz)
        
        if beta < 1e-15:
            break
        
        xi = max(np.random.random(), 1e-15)
        l_mfp = -np.log(xi) / beta
        
        if l_mfp >= l_boundary:
            # Escaped
            x_new = max(0.0, min(1.0, x + l_boundary*sx))
            y_new = max(0.0, min(1.0, y + l_boundary*sy))
            z_new = max(0.0, min(1.0, z + l_boundary*sz))
            
            xm, ym, zm = 0.5*(x+x_new), 0.5*(y+y_new), 0.5*(z+z_new)
            ix, iy, iz = int(xm/dx), int(ym/dy), int(zm/dz)
            ix = max(0, min(nx-1, ix))
            iy = max(0, min(ny-1, iy))
            iz = max(0, min(nz-1, iz))
            
            ix_arr[n_seg] = ix
            iy_arr[n_seg] = iy
            iz_arr[n_seg] = iz
            optical_path_arr[n_seg] = l_boundary * beta  # optical path
            n_seg += 1
            break
        
        # Move to collision
        x_new = x + l_mfp*sx
        y_new = y + l_mfp*sy
        z_new = z + l_mfp*sz
        
        xm, ym, zm = 0.5*(x+x_new), 0.5*(y+y_new), 0.5*(z+z_new)
        ix, iy, iz = int(xm/dx), int(ym/dy), int(zm/dz)
        ix = max(0, min(nx-1, ix))
        iy = max(0, min(ny-1, iy))
        iz = max(0, min(nz-1, iz))
        
        ix_arr[n_seg] = ix
        iy_arr[n_seg] = iy
        iz_arr[n_seg] = iz
        optical_path_arr[n_seg] = l_mfp * beta  # optical path
        n_seg += 1
        
        # Check interaction type
        if np.random.random() < (kappa / beta):
            break  # Absorbed
        else:
            sx, sy, sz = sample_hg_direction(sx, sy, sz, g_hg)
            x, y, z = x_new, y_new, z_new
    
    return ix_arr[:n_seg], iy_arr[:n_seg], iz_arr[:n_seg], optical_path_arr[:n_seg]

@njit(cache=True, parallel=True, fastmath=True)
def compute_G_monte_carlo(n_photons, kappa, sigma_s, beta, g_hg, nx, ny, nz, dx, dy, dz):
    """Compute G using track length estimator."""
    G = np.zeros((nz, ny, nx), dtype=np.float64)
    
    for i in prange(n_photons):
        ix_arr, iy_arr, iz_arr, path_arr = track_photon_for_G(
            kappa, sigma_s, beta, g_hg, nx, ny, nz, dx, dy, dz
        )
        for j in range(len(ix_arr)):
            G[iz_arr[j], iy_arr[j], ix_arr[j]] += path_arr[j]
    
    return G

def run_monte_carlo_v3(case='B'):
    if case == 'B':
        kappa, sigma_s, g_hg, suffix = 0.5, 0.5, 0.0, 'CaseB'
    elif case == 'C':
        kappa, sigma_s, g_hg, suffix = 0.1, 0.9, 0.6, 'CaseC'
    else:
        raise ValueError(f"Unknown case: {case}")
    
    beta = kappa + sigma_s
    
    print("="*70)
    print(f"Monte Carlo v3 - Track Length Estimator - {suffix}")
    print("="*70)
    print(f"Physics: kappa={kappa}, sigma_s={sigma_s}, beta={beta}, g={g_hg}")
    print(f"Grid: {NX}x{NY}x{NZ}, Photons: {N_PHOTONS:,}")
    print("="*70)
    
    dx = dy = dz = 1.0 / NX
    cell_volume = dx ** 3
    
    # Source integral
    Q_total = np.pi / 24
    
    print("\nComputing G using track length estimator...")
    start = time.time()
    
    G_raw = compute_G_monte_carlo(N_PHOTONS, kappa, sigma_s, beta, g_hg, 
                                   NX, NY, NZ, dx, dy, dz)
    
    # CORRECT NORMALIZATION (v3 fix):
    # G = (Q_total/N_photons) × (1/V_cell) × track_length_sum
    # WITHOUT the extra 4π factor!
    G_normalized = G_raw * (Q_total / N_PHOTONS) / cell_volume
    
    print(f"  Time: {time.time()-start:.1f}s")
    print(f"  Raw G range: [{G_normalized.min():.4f}, {G_normalized.max():.4f}]")
    
    # Apply smoothing
    if APPLY_SMOOTHING:
        print(f"\n  Applying Gaussian smoothing (sigma={SMOOTHING_SIGMA})...")
        G_smoothed = gaussian_filter(G_normalized, sigma=SMOOTHING_SIGMA, mode='nearest')
        print(f"  Smoothed G range: [{G_smoothed.min():.4f}, {G_smoothed.max():.4f}]")
    else:
        G_smoothed = G_normalized
    
    # Center values
    iz, iy, ix = NZ//2, NY//2, NX//2
    print(f"  G at center (raw): {G_normalized[iz,iy,ix]:.4f}")
    print(f"  G at center (smooth): {G_smoothed[iz,iy,ix]:.4f}")
    print(f"  G at face: {G_smoothed[0,iy,ix]:.4f}")
    
    # Save
    output_dir = 'Benchmark_Data'
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'Benchmark_G_{suffix}_v3.npy'), G_smoothed)
    np.save(os.path.join(output_dir, f'Benchmark_G_{suffix}_v3_raw.npy'), G_normalized)
    
    # Plot
    plot_results(G_normalized, G_smoothed, suffix, output_dir)
    
    return G_smoothed

def plot_results(G_raw, G_smooth, suffix, output_dir):
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 11
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    iz = NZ // 2
    iy = NY // 2
    x_center = np.linspace(0, 1, NX)
    
    # Raw G slice
    im1 = axes[0,0].imshow(G_raw[iz, :, :], origin='lower', extent=[0,1,0,1], cmap='hot')
    axes[0,0].set_title(f'{suffix}: G Raw (z=0.5)')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Smoothed G slice
    im2 = axes[0,1].imshow(G_smooth[iz, :, :], origin='lower', extent=[0,1,0,1], cmap='hot')
    axes[0,1].set_title(f'{suffix}: G Smoothed (z=0.5)')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Centerline comparison
    axes[1,0].plot(x_center, G_raw[iz, iy, :], 'k-', linewidth=1.5, alpha=0.5, label='Raw')
    axes[1,0].plot(x_center, G_smooth[iz, iy, :], 'r-', linewidth=2.5, label='Smoothed')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('G')
    axes[1,0].set_title(f'{suffix}: G(x,0.5,0.5) Comparison')
    axes[1,0].legend(frameon=True)
    axes[1,0].grid(True, alpha=0.3)
    
    # Radial profile
    center = np.array([0.5, 0.5, 0.5])
    dr = 0.5 / 20
    r_bins = np.arange(0, 0.5, dr)
    G_radial_raw = []
    G_radial_smooth = []
    
    for r in r_bins:
        mask = np.zeros((NZ, NY, NX), dtype=bool)
        for iz in range(NZ):
            for iy in range(NY):
                for ix in range(NX):
                    x = (ix + 0.5) / NX
                    y = (iy + 0.5) / NY
                    z = (iz + 0.5) / NZ
                    d = np.sqrt((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)
                    if r <= d < r + dr:
                        mask[iz, iy, ix] = True
        if mask.sum() > 0:
            G_radial_raw.append(G_raw[mask].mean())
            G_radial_smooth.append(G_smooth[mask].mean())
        else:
            G_radial_raw.append(np.nan)
            G_radial_smooth.append(np.nan)
    
    axes[1,1].plot(r_bins, G_radial_raw, 'k-', linewidth=1.5, alpha=0.5, marker='o', markersize=4, label='Raw')
    axes[1,1].plot(r_bins, G_radial_smooth, 'r-', linewidth=2.5, marker='s', markersize=4, label='Smoothed')
    axes[1,1].set_xlabel('r (distance from center)')
    axes[1,1].set_ylabel('G (radial average)')
    axes[1,1].set_title(f'{suffix}: Radial Profile')
    axes[1,1].legend(frameon=True)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Benchmark_{suffix}_v3.png'), dpi=600)
    plt.close()
    print(f"Saved: {output_dir}/Benchmark_{suffix}_v3.png")

if __name__ == "__main__":
    case = sys.argv[1].upper() if len(sys.argv) > 1 else CASE
    run_monte_carlo_v3(case)
