#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte Carlo 3D RTE Benchmark Solver

High-fidelity reference data generator for PINN validation using
Monte Carlo photon tracking with Numba acceleration.

Physics:
- 3D Radiative Transfer Equation (RTE) in unit cube [0,1]^3
- Absorption (κ), Scattering (σs), Emission (Ib)
- Henyey-Greenstein phase function for anisotropic scattering
- Cold black boundaries (zero incident radiation)

Method:
- Monte Carlo photon bundle tracking
- Numba JIT compilation with parallelization
- Energy deposition tally in uniform grid

Output:
- G(x,y,z): Incident radiation (3D array)
- I(x,0.5,0.5,s): Directional intensity along centerline (1D array)

Usage:
    python monte_carlo_3d_rte_benchmark.py [B|C]
    
    B: Case 3D_B (Isotropic Scattering, κ=0.5, σs=0.5, g=0.0)
    C: Case 3D_C (Forward Anisotropic, κ=0.1, σs=0.9, g=0.6)
"""

import numpy as np
import numba
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import sys
import time
from scipy.special import roots_legendre

# ==============================================================================
# CONFIGURATION - Modify these parameters as needed
# ==============================================================================

# Default case (can be overridden by command line argument)
CASE = 'B'  # Options: 'B' or 'C'

# Grid resolution
NX, NY, NZ = 40, 40, 40  # Spatial grid (can increase to 50 for higher resolution)

# Monte Carlo parameters
N_PHOTONS = 5_000_000    # Total number of photon bundles
BATCH_SIZE = 100_000     # Process in batches to show progress

# Directional intensity evaluation
N_LINE_POINTS = 50       # Number of points along centerline
THETA_VIEW = np.pi / 2   # View direction: polar angle
PHI_VIEW = 0.0           # View direction: azimuthal angle (+x direction)

# Angular tolerance for line evaluation (cosine threshold)
COS_ANGLE_TOL = 0.995    # ~5.7 degrees

# ==============================================================================
# NUMBA-ACCELERATED CORE FUNCTIONS
# ==============================================================================

@njit(cache=True, fastmath=True)
def source_term(x, y, z):
    """Blackbody emission: Ib = max(0, 1.0 - 2.0 * d)"""
    d = np.sqrt((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)
    val = 1.0 - 2.0 * d
    return val if val > 0.0 else 0.0

@njit(cache=True, fastmath=True)
def sample_source_point():
    """Sample emission point using rejection sampling."""
    while True:
        x = np.random.random()
        y = np.random.random()
        z = np.random.random()
        if np.random.random() < source_term(x, y, z):
            return x, y, z

@njit(cache=True, fastmath=True)
def sample_isotropic_direction():
    """Sample isotropic direction on unit sphere."""
    cos_theta = 2.0 * np.random.random() - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    phi = 2.0 * np.pi * np.random.random()
    
    sx = sin_theta * np.cos(phi)
    sy = sin_theta * np.sin(phi)
    sz = cos_theta
    return sx, sy, sz

@njit(cache=True, fastmath=True)
def sample_hg_direction(sx, sy, sz, g):
    """Sample new direction using Henyey-Greenstein phase function."""
    if abs(g) < 1e-10:
        return sample_isotropic_direction()
    
    # Sample cos(theta) in scattering frame
    if g > 0.999:
        cos_theta = 1.0 - 2e-6 * np.random.random()  # Forward peaked
    else:
        s = (1.0 - g * g) / (1.0 - g + 2.0 * g * np.random.random())
        cos_theta = (1.0 + g * g - s * s) / (2.0 * g)
        cos_theta = max(-1.0, min(1.0, cos_theta))
    
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * np.pi * np.random.random()
    
    # Build rotation matrix
    if abs(sz) < 0.999:
        ux = -sy
        uy = sx
        norm = np.sqrt(ux*ux + uy*uy)
        ux /= norm
        uy /= norm
        uz = 0.0
    else:
        ux, uy, uz = 1.0, 0.0, 0.0
    
    vx = sy * uz - sz * uy
    vy = sz * ux - sx * uz
    vz = sx * uy - sy * ux
    
    sx_new = sin_theta * np.cos(phi) * ux + sin_theta * np.sin(phi) * vx + cos_theta * sx
    sy_new = sin_theta * np.cos(phi) * uy + sin_theta * np.sin(phi) * vy + cos_theta * sy
    sz_new = sin_theta * np.cos(phi) * uz + sin_theta * np.sin(phi) * vz + cos_theta * sz
    
    norm = np.sqrt(sx_new*sx_new + sy_new*sy_new + sz_new*sz_new)
    return sx_new/norm, sy_new/norm, sz_new/norm

@njit(cache=True, fastmath=True)
def compute_boundary_distance(x, y, z, sx, sy, sz):
    """
    Compute distance to nearest boundary in direction of travel.
    Returns positive distance or large value if no intersection.
    """
    # Initialize with large value
    l_boundary = 1e10
    
    # Check x boundaries
    if sx > 1e-15:
        l = (1.0 - x) / sx
        if l > 0 and l < l_boundary:
            l_boundary = l
    elif sx < -1e-15:
        l = -x / sx
        if l > 0 and l < l_boundary:
            l_boundary = l
    
    # Check y boundaries
    if sy > 1e-15:
        l = (1.0 - y) / sy
        if l > 0 and l < l_boundary:
            l_boundary = l
    elif sy < -1e-15:
        l = -y / sy
        if l > 0 and l < l_boundary:
            l_boundary = l
    
    # Check z boundaries
    if sz > 1e-15:
        l = (1.0 - z) / sz
        if l > 0 and l < l_boundary:
            l_boundary = l
    elif sz < -1e-15:
        l = -z / sz
        if l > 0 and l < l_boundary:
            l_boundary = l
    
    return l_boundary

@njit(cache=True, fastmath=True)
def track_single_photon(kappa, sigma_s, beta, g_hg, nx, ny, nz, dx, dy, dz,
                         sx_view, sy_view, sz_view, cos_angle_tol, x_line_coords):
    """
    Track a single photon and return its contribution to G and I_line.
    
    Returns:
        (ix_list, iy_list, iz_list, weights, i_line_idx, i_line_weights)
    """
    # Initialize
    x, y, z = sample_source_point()
    sx, sy, sz = sample_isotropic_direction()
    weight = 1.0
    
    # Lists to accumulate path segments (pre-allocate with reasonable size)
    max_segments = 100
    ix_arr = np.zeros(max_segments, dtype=np.int32)
    iy_arr = np.zeros(max_segments, dtype=np.int32)
    iz_arr = np.zeros(max_segments, dtype=np.int32)
    w_arr = np.zeros(max_segments, dtype=np.float64)
    i_line_idx_arr = np.zeros(max_segments, dtype=np.int32)
    i_line_w_arr = np.zeros(max_segments, dtype=np.float64)
    
    n_segments = 0
    n_line_contrib = 0
    
    while weight > 1e-10:  # Russian roulette could be added here
        # Distance to boundary
        l_boundary = compute_boundary_distance(x, y, z, sx, sy, sz)
        
        # Sample free path
        if beta < 1e-15:
            l_mfp = l_boundary + 1.0  # No interaction, will escape
        else:
            xi = np.random.random()
            if xi < 1e-15:
                xi = 1e-15
            l_mfp = -np.log(xi) / beta
        
        # Check escape
        if l_mfp >= l_boundary:
            # Escaped domain
            break
        
        # Move to interaction point
        x += l_mfp * sx
        y += l_mfp * sy
        z += l_mfp * sz
        
        # Grid indices
        ix = int(x / dx)
        iy = int(y / dy)
        iz = int(z / dz)
        
        # Clamp
        if ix < 0: ix = 0
        if ix >= nx: ix = nx - 1
        if iy < 0: iy = 0
        if iy >= ny: iy = ny - 1
        if iz < 0: iz = 0
        if iz >= nz: iz = nz - 1
        
        # Store segment
        if n_segments < max_segments:
            ix_arr[n_segments] = ix
            iy_arr[n_segments] = iy
            iz_arr[n_segments] = iz
            w_arr[n_segments] = weight
            n_segments += 1
        
        # Check for line evaluation (y≈0.5, z≈0.5)
        dy_half = dy / 2.0
        dz_half = dz / 2.0
        if abs(y - 0.5) < dy_half and abs(z - 0.5) < dz_half:
            cos_angle = sx * sx_view + sy * sy_view + sz * sz_view
            if cos_angle > cos_angle_tol:
                ix_line = int(x / dx)
                if 0 <= ix_line < len(x_line_coords):
                    if n_line_contrib < max_segments:
                        i_line_idx_arr[n_line_contrib] = ix_line
                        i_line_w_arr[n_line_contrib] = weight
                        n_line_contrib += 1
        
        # Interaction type
        if beta > 1e-15 and np.random.random() < (kappa / beta):
            # Absorbed
            break
        else:
            # Scattered
            sx, sy, sz = sample_hg_direction(sx, sy, sz, g_hg)
    
    return (ix_arr[:n_segments], iy_arr[:n_segments], iz_arr[:n_segments], w_arr[:n_segments],
            i_line_idx_arr[:n_line_contrib], i_line_w_arr[:n_line_contrib])

# ==============================================================================
# PARALLEL BATCH PROCESSING
# ==============================================================================

@njit(cache=True, parallel=True, fastmath=True)
def process_photon_batch(n_photons, kappa, sigma_s, beta, g_hg, nx, ny, nz, 
                         dx, dy, dz, sx_view, sy_view, sz_view, cos_angle_tol, x_line_coords):
    """Process a batch of photons in parallel."""
    # Initialize output arrays
    G_batch = np.zeros((nz, ny, nx), dtype=np.float64)
    I_batch = np.zeros(len(x_line_coords), dtype=np.float64)
    
    # Track each photon
    for i in prange(n_photons):
        result = track_single_photon(kappa, sigma_s, beta, g_hg, nx, ny, nz, 
                                      dx, dy, dz, sx_view, sy_view, sz_view, 
                                      cos_angle_tol, x_line_coords)
        
        ix_arr, iy_arr, iz_arr, w_arr, i_line_idx, i_line_w = result
        
        # Accumulate to G
        for j in range(len(ix_arr)):
            G_batch[iz_arr[j], iy_arr[j], ix_arr[j]] += w_arr[j]
        
        # Accumulate to I_line
        for j in range(len(i_line_idx)):
            I_batch[i_line_idx[j]] += i_line_w[j]
    
    return G_batch, I_batch

# ==============================================================================
# MAIN SOLVER
# ==============================================================================

def run_monte_carlo(case='B'):
    """Run Monte Carlo simulation."""
    
    # Set physical parameters
    if case == 'B':
        kappa = 0.5
        sigma_s = 0.5
        g_hg = 0.0
        suffix = 'CaseB'
    elif case == 'C':
        kappa = 0.1
        sigma_s = 0.9
        g_hg = 0.6
        suffix = 'CaseC'
    else:
        raise ValueError(f"Unknown case: {case}")
    
    beta = kappa + sigma_s
    
    print("="*70)
    print(f"Monte Carlo 3D RTE Benchmark Solver - {suffix}")
    print("="*70)
    print(f"Physics: κ={kappa}, σs={sigma_s}, β={beta}, g={g_hg}")
    print(f"Grid: {NX}×{NY}×{NZ}, Photons: {N_PHOTONS:,}")
    print(f"CPU cores: {os.cpu_count()}")
    print("="*70)
    
    # Grid spacing
    dx, dy, dz = 1.0/NX, 1.0/NY, 1.0/NZ
    
    # View direction components
    sx_view = np.sin(THETA_VIEW) * np.cos(PHI_VIEW)
    sy_view = np.sin(THETA_VIEW) * np.sin(PHI_VIEW)
    sz_view = np.cos(THETA_VIEW)
    
    # Line evaluation coordinates
    x_line_coords = np.linspace(0, 1, N_LINE_POINTS)
    
    # Initialize tally arrays
    G_total = np.zeros((NZ, NY, NX), dtype=np.float64)
    I_total = np.zeros(N_LINE_POINTS, dtype=np.float64)
    
    # Process in batches
    n_batches = (N_PHOTONS + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()
    
    for batch in range(n_batches):
        batch_start = batch * BATCH_SIZE
        batch_end = min((batch + 1) * BATCH_SIZE, N_PHOTONS)
        n_photons_batch = batch_end - batch_start
        
        print(f"\n[Batch {batch+1}/{n_batches}] Tracking {n_photons_batch:,} photons...")
        batch_start_time = time.time()
        
        # Process batch (parallel)
        G_batch, I_batch = process_photon_batch(
            n_photons_batch, kappa, sigma_s, beta, g_hg, 
            NX, NY, NZ, dx, dy, dz, 
            sx_view, sy_view, sz_view, COS_ANGLE_TOL, x_line_coords
        )
        
        # Accumulate
        G_total += G_batch
        I_total += I_batch
        
        batch_time = time.time() - batch_start_time
        rate = n_photons_batch / max(batch_time, 0.001)
        print(f"  Time: {batch_time:.2f}s, Rate: {rate:.0f} photons/s")
        
        # Progress
        elapsed = time.time() - start_time
        progress = (batch + 1) / n_batches
        eta = elapsed / progress - elapsed
        print(f"  Progress: {100*progress:.1f}%, ETA: {eta/60:.1f} min")
    
    # Normalization
    print("\n" + "="*70)
    print("Normalizing results...")
    
    # Source integral (numerical integration)
    source_integral = 0.0
    for i in range(100):
        for j in range(100):
            for k in range(100):
                x = (i + 0.5) / 100.0
                y = (j + 0.5) / 100.0
                z = (k + 0.5) / 100.0
                source_integral += source_term(x, y, z)
    source_integral /= 100**3
    
    # Photon weight = total source / number of photons
    photon_weight = source_integral / N_PHOTONS
    
    # G normalization: each photon visit contributes to G
    # G = (4π / V_cell) × (energy deposited in cell)
    cell_volume = dx * dy * dz
    G_normalized = G_total * photon_weight / cell_volume
    
    # I normalization
    dx_line = 1.0 / (N_LINE_POINTS - 1) if N_LINE_POINTS > 1 else 1.0
    I_normalized = I_total * photon_weight / dx_line
    
    print(f"G range: [{G_normalized.min():.6f}, {G_normalized.max():.6f}]")
    print(f"I range: [{I_normalized.min():.6f}, {I_normalized.max():.6f}]")
    print(f"Total time: {(time.time()-start_time)/60:.1f} minutes")
    print("="*70)
    
    # Save outputs
    output_dir = 'Benchmark_Data'
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f'Benchmark_G_{suffix}.npy'), G_normalized)
    np.save(os.path.join(output_dir, f'Benchmark_I_Line_{suffix}.npy'), I_normalized)
    np.save(os.path.join(output_dir, f'Benchmark_x_Line_{suffix}.npy'), x_line_coords)
    
    # Save parameters
    params = {
        'case': suffix,
        'kappa': kappa,
        'sigma_s': sigma_s,
        'beta': beta,
        'g_hg': g_hg,
        'nx': NX, 'ny': NY, 'nz': NZ,
        'n_photons': N_PHOTONS,
        'theta_view': THETA_VIEW,
        'phi_view': PHI_VIEW
    }
    np.save(os.path.join(output_dir, f'Benchmark_Params_{suffix}.npy'), params)
    
    print(f"\nOutputs saved to {output_dir}/")
    
    # Generate visualization
    plot_results(G_normalized, I_normalized, x_line_coords, suffix, output_dir)
    
    return G_normalized, I_normalized, x_line_coords

def plot_results(G, I_line, x_line, suffix, output_dir):
    """Generate publication-quality plots."""
    
    # Set journal-quality style
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 11
    rcParams['axes.linewidth'] = 1.0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Center slice (z=0.5)
    iz_center = NZ // 2
    G_slice = G[iz_center, :, :]
    
    im1 = axes[0].imshow(G_slice, origin='lower', extent=[0,1,0,1], 
                         cmap='hot', aspect='equal')
    axes[0].set_title(f'{suffix}: G(x,y,z=0.5)', fontsize=12)
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('y', fontsize=11)
    plt.colorbar(im1, ax=axes[0], label='G')
    
    # 2. Centerline G
    iy_center = NY // 2
    G_centerline = G[iz_center, iy_center, :]
    x_center = np.linspace(0, 1, NX)
    
    axes[1].plot(x_center, G_centerline, 'k-', linewidth=2, marker='o', 
                 markersize=6, markevery=5, label='MC Benchmark')
    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('G', fontsize=11)
    axes[1].set_title(f'{suffix}: G(x,0.5,0.5)', fontsize=12)
    axes[1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Directional intensity
    axes[2].plot(x_line, I_line, 'b-', linewidth=2, marker='s', 
                 markersize=6, markevery=5)
    axes[2].set_xlabel('x', fontsize=11)
    axes[2].set_ylabel('I', fontsize=11)
    axes[2].set_title(f'{suffix}: I(x,0.5,0.5, θ={THETA_VIEW:.2f})', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Benchmark_{suffix}.png'), 
                dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/Benchmark_{suffix}.png")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        CASE = sys.argv[1].upper()
    
    # Compile Numba functions (warm-up)
    print("Compiling Numba functions...")
    _ = sample_source_point()
    _ = sample_isotropic_direction()
    _ = sample_hg_direction(1.0, 0.0, 0.0, 0.0)
    print("Compilation complete.\n")
    
    # Run simulation
    run_monte_carlo(CASE)
