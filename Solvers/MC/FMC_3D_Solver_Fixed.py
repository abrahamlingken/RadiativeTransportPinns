#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FMC_3D_Solver_Fixed.py - Forward Monte Carlo Solver for 3D RTE
===============================================================

BUG FIXES APPLIED:
1. Track-Length Estimator: Energy deposited continuously along path using micro-steps
2. Data Race: Removed parallel=True, using single-threaded accumulation
3. Source Integral: Fixed calculus error (π/375 instead of π/150)

HIGH STATISTICS VERSION:
- 200 Million photons per case
- 100 batch progress tracking
- HighStats suffix for output files

Calculates the scalar incident radiation field G(x,y,z).

Author: Computational Physics Team
Date: 2024
"""

import numpy as np
from numba import njit
import os
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

NX, NY, NZ = 50, 50, 50
DOMAIN = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
SOURCE_CENTER = np.array([0.5, 0.5, 0.5])
SOURCE_RADIUS = 0.2

# Russian Roulette settings
RR_THRESHOLD = 1e-4
RR_SURVIVAL_PROB = 0.1
RR_WEIGHT_BOOST = 10.0

# HIGH STATISTICS: 200 Million photons per case
CASE_CONFIGS = {
    'B': {
        'name': 'Isotropic',
        'kappa': 0.5,
        'sigma_s': 4.5,
        'g': 0.0,
        'n_photons': 200_000_000  # UPDATED: 200 Million
    },
    'C': {
        'name': 'Anisotropic',
        'kappa': 0.1,
        'sigma_s': 4.9,
        'g': 0.8,
        'n_photons': 200_000_000  # UPDATED: 200 Million
    }
}


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def source_term(x, y, z):
    """Source term S(r) = max(0, 1 - 5*r)"""
    dx = x - SOURCE_CENTER[0]
    dy = y - SOURCE_CENTER[1]
    dz = z - SOURCE_CENTER[2]
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    s = 1.0 - 5.0 * r
    return s if s > 0.0 else 0.0


@njit(cache=True, fastmath=True)
def sample_source_point():
    """Sample emission point within source sphere using rejection sampling."""
    while True:
        x = 0.3 + 0.4 * np.random.random()
        y = 0.3 + 0.4 * np.random.random()
        z = 0.3 + 0.4 * np.random.random()
        
        dx = x - SOURCE_CENTER[0]
        dy = y - SOURCE_CENTER[1]
        dz = z - SOURCE_CENTER[2]
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if r < SOURCE_RADIUS:
            if np.random.random() < source_term(x, y, z):
                return x, y, z


@njit(cache=True, fastmath=True)
def sample_isotropic_direction():
    """Sample isotropic direction vector."""
    cos_theta = 2.0 * np.random.random() - 1.0
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
    phi = 2.0 * np.pi * np.random.random()
    
    ux = sin_theta * np.cos(phi)
    uy = sin_theta * np.sin(phi)
    uz = cos_theta
    return ux, uy, uz


@njit(cache=True, fastmath=True)
def sample_hg_direction(ux, uy, uz, g):
    """Sample new direction using Henyey-Greenstein phase function."""
    if abs(g) < 1e-10:
        return sample_isotropic_direction()
    
    if abs(g) > 0.999:
        cos_theta = 1.0 - 1e-6 * np.random.random()
    else:
        s = (1.0 - g*g) / (1.0 - g + 2.0 * g * np.random.random())
        cos_theta = (1.0 + g*g - s*s) / (2.0 * g)
        cos_theta = max(-1.0, min(1.0, cos_theta))
    
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
    phi = 2.0 * np.pi * np.random.random()
    
    # Build local coordinate system
    if abs(uz) < 0.999:
        perp_x = -uy
        perp_y = ux
        perp_z = 0.0
        norm = np.sqrt(perp_x*perp_x + perp_y*perp_y)
        perp_x /= norm
        perp_y /= norm
    else:
        perp_x = 0.0
        perp_y = 1.0
        perp_z = 0.0
    
    perp2_x = uy * perp_z - uz * perp_y
    perp2_y = uz * perp_x - ux * perp_z
    perp2_z = ux * perp_y - uy * perp_x
    
    nx = sin_theta * (np.cos(phi) * perp_x + np.sin(phi) * perp2_x) + cos_theta * ux
    ny = sin_theta * (np.cos(phi) * perp_y + np.sin(phi) * perp2_y) + cos_theta * uy
    nz = sin_theta * (np.cos(phi) * perp_z + np.sin(phi) * perp2_z) + cos_theta * uz
    
    norm = np.sqrt(nx*nx + ny*ny + nz*nz)
    return nx/norm, ny/norm, nz/norm


@njit(cache=True, fastmath=True)
def distance_to_boundary(x, y, z, ux, uy, uz, xmin, xmax, ymin, ymax, zmin, zmax):
    """Calculate distance to domain boundary."""
    t_min = 1e15
    
    if ux > 1e-15:
        t = (xmax - x) / ux
        if t < t_min:
            t_min = t
    elif ux < -1e-15:
        t = (xmin - x) / ux
        if t < t_min:
            t_min = t
    
    if uy > 1e-15:
        t = (ymax - y) / uy
        if t < t_min:
            t_min = t
    elif uy < -1e-15:
        t = (ymin - y) / uy
        if t < t_min:
            t_min = t
    
    if uz > 1e-15:
        t = (zmax - z) / uz
        if t < t_min:
            t_min = t
    elif uz < -1e-15:
        t = (zmin - z) / uz
        if t < t_min:
            t_min = t
    
    return t_min


@njit(cache=True, fastmath=True)
def get_cell_indices(x, y, z, xmin, ymin, zmin, dx, dy, dz):
    """Get cell indices for current position."""
    ix = int((x - xmin) / dx)
    iy = int((y - ymin) / dy)
    iz = int((z - zmin) / dz)
    
    if ix < 0:
        ix = 0
    elif ix >= NX:
        ix = NX - 1
    
    if iy < 0:
        iy = 0
    elif iy >= NY:
        iy = NY - 1
    
    if iz < 0:
        iz = 0
    elif iz >= NZ:
        iz = NZ - 1
    
    return ix, iy, iz


@njit(cache=True, fastmath=True)
def trace_photon(G_field, beta, albedo, g, xmin, xmax, ymin, ymax, zmin, zmax, 
                 dx, dy, dz, cell_volume, initial_weight):
    """
    BUG FIX 1: Correct Track-Length Estimator.
    
    Energy deposited continuously along path using micro-steps.
    Formula: G_field += weight * ds / cell_volume (NO /beta)
    """
    x, y, z = sample_source_point()
    ux, uy, uz = sample_isotropic_direction()
    weight = initial_weight
    
    collision_count = 0
    max_collisions = 1000
    
    # Micro-step size
    ds_micro = min(dx, dy, dz) / 5.0
    
    while collision_count < max_collisions and weight > 1e-10:
        dist_boundary = distance_to_boundary(x, y, z, ux, uy, uz, 
                                            xmin, xmax, ymin, ymax, zmin, zmax)
        
        xi = np.random.random()
        if xi < 1e-15:
            xi = 1e-15
        dist_collision = -np.log(1.0 - xi) / beta
        
        dist_travel = dist_collision if dist_collision < dist_boundary else dist_boundary
        
        # === CORRECT TLE: Micro-step deposition ===
        n_micro_steps = int(dist_travel / ds_micro) + 1
        ds_actual = dist_travel / n_micro_steps
        
        for _ in range(n_micro_steps):
            x_mid = x + ux * ds_actual * 0.5
            y_mid = y + uy * ds_actual * 0.5
            z_mid = z + uz * ds_actual * 0.5
            
            ix, iy, iz = get_cell_indices(x_mid, y_mid, z_mid, xmin, ymin, zmin, dx, dy, dz)
            
            # CORRECT: weight * ds / V_cell (NO division by beta)
            G_field[ix, iy, iz] += weight * ds_actual / cell_volume
            
            x += ux * ds_actual
            y += uy * ds_actual
            z += uz * ds_actual
        
        if dist_collision >= dist_boundary:
            break
        
        # Survival biasing
        weight *= albedo
        
        # Russian Roulette
        if weight < RR_THRESHOLD:
            if np.random.random() < RR_SURVIVAL_PROB:
                weight *= RR_WEIGHT_BOOST
            else:
                break
        
        ux, uy, uz = sample_hg_direction(ux, uy, uz, g)
        collision_count += 1
    
    return collision_count


# BUG FIX 2: Removed parallel=True and prange
@njit(cache=True, fastmath=False)
def run_monte_carlo(G_field, n_photons, beta, albedo, g, 
                   xmin, xmax, ymin, ymax, zmin, zmax,
                   dx, dy, dz, cell_volume, initial_weight):
    """
    Single-threaded accumulation to prevent data race.
    """
    for i in range(n_photons):  # Using range instead of prange
        trace_photon(G_field, beta, albedo, g, 
                    xmin, xmax, ymin, ymax, zmin, zmax,
                    dx, dy, dz, cell_volume, initial_weight)
    
    return n_photons


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_case(case_key, output_dir='MC3D_Results'):
    """Solve a single case using Forward Monte Carlo."""
    if case_key not in CASE_CONFIGS:
        raise ValueError(f"Unknown case: {case_key}")
    
    config = CASE_CONFIGS[case_key]
    
    print("="*70)
    print(f"FMC 3D Solver (FIXED) - Case {case_key} ({config['name']})")
    print("HIGH STATISTICS VERSION - 200 Million Photons")
    print("="*70)
    
    kappa = config['kappa']
    sigma_s = config['sigma_s']
    g = config['g']
    n_photons = config['n_photons']
    
    beta = kappa + sigma_s
    albedo = sigma_s / beta if beta > 0 else 0.0
    
    print(f"Parameters:")
    print(f"  κ (absorption):    {kappa:.4f}")
    print(f"  σs (scattering):   {sigma_s:.4f}")
    print(f"  β (extinction):    {beta:.4f}")
    print(f"  ω (albedo):        {albedo:.4f}")
    print(f"  g (HG factor):     {g:.4f}")
    print(f"  Photons:           {n_photons:,}")
    print(f"  Grid:              {NX}x{NY}x{NZ}")
    
    xmin, xmax = DOMAIN[0], DOMAIN[1]
    ymin, ymax = DOMAIN[2], DOMAIN[3]
    zmin, zmax = DOMAIN[4], DOMAIN[5]
    
    dx = (xmax - xmin) / NX
    dy = (ymax - ymin) / NY
    dz = (zmax - zmin) / NZ
    cell_volume = dx * dy * dz
    
    print(f"  Cell volume:       {cell_volume:.6e}")
    
    # BUG FIX 3: Correct source integral
    # Q_volume = ∫_0^0.2 (1-5r) * 4πr² dr = π/375
    Q_volume = np.pi / 375.0  # FIXED: was np.pi / 150.0
    
    # Total power = 4π * Q_volume (emission into 4π steradians)
    total_power = 4.0 * np.pi * Q_volume  # FIXED: proper 4π factor
    
    print(f"  Source integral:   {Q_volume:.6e} (volume, π/375)")
    print(f"  Total power:       {total_power:.6e} (4π emission)")
    
    initial_weight = total_power / n_photons
    print(f"  Initial weight:    {initial_weight:.6e}")
    
    G_field = np.zeros((NX, NY, NZ), dtype=np.float64)
    
    print("\nRunning Monte Carlo simulation...")
    print("Progress updates every 1% (2 million photons)")
    start_time = time.time()
    
    # HIGH STATS: Finer progress tracking with 100 batches
    n_batches = 100  # UPDATED: 100 batches for 1% progress updates
    batch_size = n_photons // n_batches
    
    for batch in range(n_batches):
        batch_start = time.time()
        
        # Create temporary field for this batch
        G_batch = np.zeros((NX, NY, NZ), dtype=np.float64)
        
        # Run batch
        run_monte_carlo(
            G_batch, batch_size, beta, albedo, g,
            xmin, xmax, ymin, ymax, zmin, zmax,
            dx, dy, dz, cell_volume, initial_weight
        )
        
        # Accumulate to main field
        G_field += G_batch
        
        batch_time = time.time() - batch_start
        progress = (batch + 1) / n_batches * 100
        elapsed = time.time() - start_time
        eta = elapsed / (batch + 1) * (n_batches - batch - 1)
        
        print(f"  Batch {batch+1:3d}/{n_batches} ({progress:5.1f}%) | "
              f"{batch_time:6.2f}s | ETA: {eta/60:5.1f}m")
    
    total_elapsed = time.time() - start_time
    print(f"\nCompleted in {total_elapsed:.2f}s ({n_photons/total_elapsed:,.0f} photons/s)")
    
    ix_center = NX // 2
    iy_center = NY // 2
    iz_center = NZ // 2
    
    print(f"\nResults:")
    print(f"  G_center:          {G_field[ix_center, iy_center, iz_center]:.6f}")
    print(f"  G_max:             {G_field.max():.6f}")
    print(f"  G_min:             {G_field.min():.6e}")
    print(f"  G_mean:            {G_field.mean():.6e}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # HIGH STATS: Append _HighStats to filenames
    output_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}_FIXED_HighStats.npy')
    np.save(output_file, G_field)
    print(f"\nSaved to: {output_file}")
    
    metadata = {
        'case': case_key,
        'kappa': kappa,
        'sigma_s': sigma_s,
        'beta': beta,
        'g': g,
        'n_photons': n_photons,
        'Q_volume': float(Q_volume),
        'total_power': float(total_power),
        'G_center': float(G_field[ix_center, iy_center, iz_center]),
        'G_max': float(G_field.max()),
        'G_min': float(G_field.min())
    }
    
    # HIGH STATS: Append _HighStats to metadata filename
    meta_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}_FIXED_HighStats_meta.npz')
    np.savez(meta_file, **metadata)
    print(f"Metadata saved to: {meta_file}")
    
    return G_field


def plot_results(G_field, case_key, output_dir='MC3D_Results'):
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
        
        ix_center = NX // 2
        iy_center = NY // 2
        iz_center = NZ // 2
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        ax = axes[0, 0]
        im = ax.imshow(G_field[:, :, iz_center].T, origin='lower', cmap='hot',
                       extent=[0, 1, 0, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Case {case_key}: G(x,y,0.5)')
        plt.colorbar(im, ax=ax)
        
        ax = axes[0, 1]
        im = ax.imshow(G_field[:, iy_center, :].T, origin='lower', cmap='hot',
                       extent=[0, 1, 0, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(f'Case {case_key}: G(x,0.5,z)')
        plt.colorbar(im, ax=ax)
        
        ax = axes[1, 0]
        x_vals = np.linspace(0, 1, NX)
        ax.plot(x_vals, G_field[:, iy_center, iz_center], 'b-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('G(x, 0.5, 0.5)')
        ax.set_title('Centerline Profile')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.axis('off')
        
        info_text = (
            f"Case {case_key} Statistics (HIGH STATS)\n"
            f"\n"
            f"G_center:  {G_field[ix_center, iy_center, iz_center]:.4f}\n"
            f"G_max:     {G_field.max():.4f}\n"
            f"G_min:     {G_field.min():.4e}\n"
            f"\n"
            f"Parameters:\n"
            f"κ = {CASE_CONFIGS[case_key]['kappa']}\n"
            f"σs = {CASE_CONFIGS[case_key]['sigma_s']}\n"
            f"g = {CASE_CONFIGS[case_key]['g']}\n"
            f"\n"
            f"High Statistics:\n"
            f"200 Million photons\n"
            f"Publication-ready smoothness"
        )
        ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        # HIGH STATS: Append _HighStats to plot filename
        plot_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}_FIXED_HighStats.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping plots")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FMC 3D RTE Solver (HIGH STATS)')
    parser.add_argument('--case', type=str, choices=['B', 'C', 'all'], default='all')
    parser.add_argument('--output-dir', type=str, default='MC3D_Results')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Forward Monte Carlo 3D RTE Solver - HIGH STATISTICS")
    print("="*70)
    print("\nConfiguration:")
    print("  Photon count:  200,000,000 per case")
    print("  Progress:      Every 1% (100 batches)")
    print("  Output:        _HighStats suffix")
    print("="*70 + "\n")
    
    if args.case == 'all':
        cases = ['B', 'C']
    else:
        cases = [args.case]
    
    results = {}
    for case_key in cases:
        G_field = solve_case(case_key, args.output_dir)
        results[case_key] = G_field
        plot_results(G_field, case_key, args.output_dir)
        print("\n" + "="*70 + "\n")
    
    print("All cases completed!")
    print(f"Results saved in: {args.output_dir}/")
    
    return results


if __name__ == "__main__":
    main()
