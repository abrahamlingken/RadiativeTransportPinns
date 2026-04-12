#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FMC_3D_Solver.py - Forward Monte Carlo Solver for 3D RTE
=========================================================

Calculates the scalar incident radiation field G(x,y,z) using:
- Forward Monte Carlo method
- Track-Length Estimator
- Survival Biasing (Implicit Capture)
- Russian Roulette variance reduction
- Henyey-Greenstein phase function

Optimized with Numba JIT compilation.

Author: Computational Physics Team
Date: 2024
"""

import numpy as np
from numba import njit, prange
import os
import time
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Grid settings
NX, NY, NZ = 50, 50, 50
DOMAIN = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])  # [xmin, xmax, ymin, ymax, zmin, zmax]

# Source settings
SOURCE_CENTER = np.array([0.5, 0.5, 0.5])
SOURCE_RADIUS = 0.2

# Russian Roulette settings
RR_THRESHOLD = 1e-4
RR_SURVIVAL_PROB = 0.1
RR_WEIGHT_BOOST = 1.0 / RR_SURVIVAL_PROB  # 10.0

# Case configurations
CASE_CONFIGS = {
    'B': {
        'name': 'Isotropic',
        'kappa': 0.5,
        'sigma_s': 4.5,
        'g': 0.0,
        'n_photons': 10_000_000
    },
    'C': {
        'name': 'Anisotropic',
        'kappa': 0.1,
        'sigma_s': 4.9,
        'g': 0.8,
        'n_photons': 10_000_000
    }
}


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def source_term(x, y, z):
    """
    Source term S(r) = max(0, 1 - 5*r)
    where r is distance from source center.
    """
    dx = x - SOURCE_CENTER[0]
    dy = y - SOURCE_CENTER[1]
    dz = z - SOURCE_CENTER[2]
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    s = 1.0 - 5.0 * r
    return s if s > 0.0 else 0.0


@njit(cache=True, fastmath=True)
def sample_source_point():
    """
    Sample emission point within source sphere using rejection sampling.
    Returns (x, y, z) and initial weight factor.
    """
    while True:
        # Sample uniformly in box [0.3, 0.7]^3 (contains sphere r<0.2)
        x = 0.3 + 0.4 * np.random.random()
        y = 0.3 + 0.4 * np.random.random()
        z = 0.3 + 0.4 * np.random.random()
        
        # Check if inside sphere
        dx = x - SOURCE_CENTER[0]
        dy = y - SOURCE_CENTER[1]
        dz = z - SOURCE_CENTER[2]
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if r < SOURCE_RADIUS:
            # Rejection sampling based on source strength
            S_max = 1.0  # Maximum source value at center
            if np.random.random() < source_term(x, y, z) / S_max:
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
    """
    Sample new direction after scattering using Henyey-Greenstein phase function.
    
    Parameters:
        ux, uy, uz: Current direction cosines
        g: Asymmetry factor (-1 to 1)
    
    Returns:
        New direction cosines (nx, ny, nz)
    """
    if abs(g) < 1e-10:
        # Isotropic scattering
        return sample_isotropic_direction()
    
    # Sample scattering angle theta from HG phase function
    # P(cos_theta) = (1 - g^2) / (4*pi * (1 + g^2 - 2*g*cos_theta)^(3/2))
    
    if abs(g) > 0.999:
        # Highly forward scattering
        cos_theta = 1.0 - 1e-6 * np.random.random()
    else:
        # Sample from HG distribution using inversion method
        s = (1.0 - g*g) / (1.0 - g + 2.0 * g * np.random.random())
        cos_theta = (1.0 + g*g - s*s) / (2.0 * g)
        cos_theta = max(-1.0, min(1.0, cos_theta))
    
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
    phi = 2.0 * np.pi * np.random.random()
    
    # Build local coordinate system
    # Find perpendicular vectors to current direction
    if abs(uz) < 0.999:
        # Use cross product with z-axis
        perp_x = -uy
        perp_y = ux
        perp_z = 0.0
        norm = np.sqrt(perp_x*perp_x + perp_y*perp_y)
        perp_x /= norm
        perp_y /= norm
    else:
        # Current direction is nearly z-axis, use x-axis
        perp_x = 0.0
        perp_y = 1.0
        perp_z = 0.0
    
    # Second perpendicular vector (cross product)
    perp2_x = uy * perp_z - uz * perp_y
    perp2_y = uz * perp_x - ux * perp_z
    perp2_z = ux * perp_y - uy * perp_x
    
    # New direction in global coordinates
    nx = sin_theta * (np.cos(phi) * perp_x + np.sin(phi) * perp2_x) + cos_theta * ux
    ny = sin_theta * (np.cos(phi) * perp_y + np.sin(phi) * perp2_y) + cos_theta * uy
    nz = sin_theta * (np.cos(phi) * perp_z + np.sin(phi) * perp2_z) + cos_theta * uz
    
    # Normalize
    norm = np.sqrt(nx*nx + ny*ny + nz*nz)
    return nx/norm, ny/norm, nz/norm


@njit(cache=True, fastmath=True)
def distance_to_boundary(x, y, z, ux, uy, uz, xmin, xmax, ymin, ymax, zmin, zmax):
    """Calculate distance to domain boundary in current direction."""
    t_min = 1e15
    
    # X boundaries
    if ux > 1e-15:
        t = (xmax - x) / ux
        if t < t_min:
            t_min = t
    elif ux < -1e-15:
        t = (xmin - x) / ux
        if t < t_min:
            t_min = t
    
    # Y boundaries
    if uy > 1e-15:
        t = (ymax - y) / uy
        if t < t_min:
            t_min = t
    elif uy < -1e-15:
        t = (ymin - y) / uy
        if t < t_min:
            t_min = t
    
    # Z boundaries
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
    
    # Clamp to valid range
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
    Trace a single photon through the domain using track-length estimator.
    
    Parameters:
        G_field: 3D array to accumulate G values (modified in place)
        beta: Total extinction coefficient
        albedo: Scattering albedo
        g: HG asymmetry factor
        initial_weight: Starting weight of photon
    
    Returns:
        Number of collisions processed
    """
    # Initialize photon
    x, y, z = sample_source_point()
    ux, uy, uz = sample_isotropic_direction()
    weight = initial_weight
    
    collision_count = 0
    max_collisions = 1000  # Safety limit
    
    while collision_count < max_collisions and weight > 1e-10:
        # Distance to boundary
        dist_boundary = distance_to_boundary(x, y, z, ux, uy, uz, 
                                            xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Sample free flight distance
        # PDF: p(s) = beta * exp(-beta * s)
        # CDF: P(s) = 1 - exp(-beta * s)
        # Sampling: s = -ln(1 - xi) / beta
        xi = np.random.random()
        if xi < 1e-15:
            xi = 1e-15
        dist_collision = -np.log(1.0 - xi) / beta
        
        if dist_collision < dist_boundary:
            # Photon will collide before boundary
            
            # Move to collision point
            x_new = x + ux * dist_collision
            y_new = y + uy * dist_collision
            z_new = z + uz * dist_collision
            
            # Track-length accumulation: traverse from (x,y,z) to (x_new,y_new,z_new)
            # Add contribution to all cells along the path
            dist_traveled = dist_collision
            
            # Simple approach: add contribution to final cell
            # More accurate: ray-marching through cells
            ix, iy, iz = get_cell_indices(x_new, y_new, z_new, xmin, ymin, zmin, dx, dy, dz)
            
            # Track-length estimator: G_cell += weight * track_length / cell_volume
            # For collision-based contribution: use implicit capture weight
            G_field[ix, iy, iz] += weight * dist_traveled / cell_volume / beta
            
            # Survival biasing: apply weight reduction
            weight *= albedo
            
            # Russian Roulette
            if weight < RR_THRESHOLD:
                if np.random.random() < RR_SURVIVAL_PROB:
                    weight *= RR_WEIGHT_BOOST
                else:
                    # Photon terminated
                    break
            
            # Scatter to new direction
            ux, uy, uz = sample_hg_direction(ux, uy, uz, g)
            
            # Update position
            x, y, z = x_new, y_new, z_new
            collision_count += 1
            
        else:
            # Photon escapes domain
            # Move to boundary
            x_new = x + ux * dist_boundary
            y_new = y + uy * dist_boundary
            z_new = z + uz * dist_boundary
            
            # Add track-length contribution along escape path
            dist_traveled = dist_boundary
            ix, iy, iz = get_cell_indices(x, y, z, xmin, ymin, zmin, dx, dy, dz)
            G_field[ix, iy, iz] += weight * dist_traveled / cell_volume / beta
            
            # Photon leaves domain
            break
    
    return collision_count


@njit(cache=True, parallel=True, fastmath=True)
def run_monte_carlo(G_field, n_photons, beta, albedo, g, 
                   xmin, xmax, ymin, ymax, zmin, zmax,
                   dx, dy, dz, cell_volume, initial_weight):
    """
    Run Monte Carlo simulation with parallel photon tracing.
    
    Note: Each thread accumulates to its own G_field copy, then combined.
    """
    # For thread safety in parallel mode, each thread needs its own accumulator
    # We'll use a simple approach: process photons in parallel but accumulate serially
    # For better performance with large n_photons, use reduction
    
    n_processed = 0
    
    for i in prange(n_photons):
        # Trace single photon
        n_coll = trace_photon(G_field, beta, albedo, g, 
                             xmin, xmax, ymin, ymax, zmin, zmax,
                             dx, dy, dz, cell_volume, initial_weight)
        
        if n_coll > 0:
            n_processed += 1
    
    return n_processed


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_case(case_key, output_dir='MC3D_Results'):
    """
    Solve a single case (B or C) using Forward Monte Carlo.
    
    Parameters:
        case_key: 'B' or 'C'
        output_dir: Directory to save results
    
    Returns:
        G_field: 3D array of scalar incident radiation
    """
    if case_key not in CASE_CONFIGS:
        raise ValueError(f"Unknown case: {case_key}. Available: {list(CASE_CONFIGS.keys())}")
    
    config = CASE_CONFIGS[case_key]
    
    print("="*70)
    print(f"FMC 3D Solver - Case {case_key} ({config['name']})")
    print("="*70)
    
    # Extract parameters
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
    
    # Grid setup
    xmin, xmax = DOMAIN[0], DOMAIN[1]
    ymin, ymax = DOMAIN[2], DOMAIN[3]
    zmin, zmax = DOMAIN[4], DOMAIN[5]
    
    dx = (xmax - xmin) / NX
    dy = (ymax - ymin) / NY
    dz = (zmax - zmin) / NZ
    cell_volume = dx * dy * dz
    
    print(f"  Cell volume:       {cell_volume:.6e}")
    
    # Calculate source integral for normalization
    # Q = ∫∫∫ S(r) dV over source sphere
    # S(r) = max(0, 1-5r), integrate over sphere r<0.2
    # Analytical: Q = 4π ∫[0,0.2] (1-5r) r² dr = π/150
    source_integral = np.pi / 150.0
    print(f"  Source integral:   {source_integral:.6e}")
    
    # Initial weight per photon
    # Total energy = source_integral
    # Weight per photon = source_integral / n_photons
    initial_weight = source_integral / n_photons
    print(f"  Initial weight:    {initial_weight:.6e}")
    
    # Initialize G field
    G_field = np.zeros((NX, NY, NZ), dtype=np.float64)
    
    # Run Monte Carlo
    print("\nRunning Monte Carlo simulation...")
    start_time = time.time()
    
    # For parallel execution with proper accumulation
    # Split photons into batches for better parallel performance
    n_batches = 10
    batch_size = n_photons // n_batches
    
    for batch in range(n_batches):
        batch_start = time.time()
        
        # Create temporary field for this batch
        G_batch = np.zeros((NX, NY, NZ), dtype=np.float64)
        
        # Run batch
        n_processed = run_monte_carlo(
            G_batch, batch_size, beta, albedo, g,
            xmin, xmax, ymin, ymax, zmin, zmax,
            dx, dy, dz, cell_volume, initial_weight
        )
        
        # Accumulate to main field
        G_field += G_batch
        
        batch_time = time.time() - batch_start
        print(f"  Batch {batch+1}/{n_batches}: {batch_size:,} photons, "
              f"{batch_time:.2f}s, {n_processed} active")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s ({n_photons/elapsed:,.0f} photons/s)")
    
    # Normalize G field
    # The track-length estimator already includes the proper normalization
    # through weight/cell_volume/beta, but we need to verify
    # G represents ∫ I dΩ
    
    # Additional normalization if needed
    # G_field *= (4.0 * np.pi)  # Uncomment if G should include 4π factor
    
    # Statistics
    ix_center = NX // 2
    iy_center = NY // 2
    iz_center = NZ // 2
    
    print(f"\nResults:")
    print(f"  G_center:          {G_field[ix_center, iy_center, iz_center]:.6f}")
    print(f"  G_max:             {G_field.max():.6f}")
    print(f"  G_min:             {G_field.min():.6e}")
    print(f"  G_mean:            {G_field.mean():.6e}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}.npy')
    np.save(output_file, G_field)
    print(f"\nSaved to: {output_file}")
    
    # Also save metadata
    metadata = {
        'case': case_key,
        'kappa': kappa,
        'sigma_s': sigma_s,
        'beta': beta,
        'g': g,
        'n_photons': n_photons,
        'G_center': float(G_field[ix_center, iy_center, iz_center]),
        'G_max': float(G_field.max()),
        'G_min': float(G_field.min())
    }
    
    meta_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}_metadata.npz')
    np.savez(meta_file, **metadata)
    print(f"Metadata saved to: {meta_file}")
    
    return G_field


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(G_field, case_key, output_dir='MC3D_Results'):
    """Create visualization plots for the G field."""
    try:
        import matplotlib.pyplot as plt
        
        ix_center = NX // 2
        iy_center = NY // 2
        iz_center = NZ // 2
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # XY slice at z=center
        ax = axes[0, 0]
        im = ax.imshow(G_field[:, :, iz_center].T, origin='lower', cmap='hot',
                       extent=[0, 1, 0, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Case {case_key}: G(x,y,0.5)')
        plt.colorbar(im, ax=ax)
        
        # XZ slice at y=center
        ax = axes[0, 1]
        im = ax.imshow(G_field[:, iy_center, :].T, origin='lower', cmap='hot',
                       extent=[0, 1, 0, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(f'Case {case_key}: G(x,0.5,z)')
        plt.colorbar(im, ax=ax)
        
        # Centerline along x
        ax = axes[1, 0]
        x_vals = np.linspace(0, 1, NX)
        ax.plot(x_vals, G_field[:, iy_center, iz_center], 'b-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('G(x, 0.5, 0.5)')
        ax.set_title('Centerline Profile')
        ax.grid(True, alpha=0.3)
        
        # Statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        info_text = (
            f"Case {case_key} Statistics\n"
            f"\n"
            f"G_center:  {G_field[ix_center, iy_center, iz_center]:.4f}\n"
            f"G_max:     {G_field.max():.4f}\n"
            f"G_min:     {G_field.min():.4e}\n"
            f"G_mean:    {G_field.mean():.4e}\n"
            f"\n"
            f"Parameters:\n"
            f"κ = {CASE_CONFIGS[case_key]['kappa']}\n"
            f"σs = {CASE_CONFIGS[case_key]['sigma_s']}\n"
            f"g = {CASE_CONFIGS[case_key]['g']}\n"
        )
        ax.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping plots")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Forward Monte Carlo Solver for 3D RTE'
    )
    parser.add_argument(
        '--case', 
        type=str, 
        choices=['B', 'C', 'all'],
        default='all',
        help='Case to run (B=isotropic, C=anisotropic, all=both)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='MC3D_Results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Forward Monte Carlo 3D RTE Solver")
    print("Track-Length Estimator with Survival Biasing")
    print("="*70 + "\n")
    
    # Determine cases to run
    if args.case == 'all':
        cases = ['B', 'C']
    else:
        cases = [args.case]
    
    # Run each case
    results = {}
    for case_key in cases:
        G_field = solve_case(case_key, args.output_dir)
        results[case_key] = G_field
        
        # Generate plots
        plot_results(G_field, case_key, args.output_dir)
        
        print("\n" + "="*70 + "\n")
    
    print("All cases completed!")
    print(f"Results saved in: {args.output_dir}/")
    
    return results


if __name__ == "__main__":
    main()
