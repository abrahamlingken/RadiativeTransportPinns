#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_pinn_vs_fmc.py - Final PINN Validation against FMC Benchmark
=====================================================================

Quantitative error analysis and publication-quality overlay plot for
comparing PINN predictions against Forward Monte Carlo benchmarks.

Processing Pipeline:
1. FMC: Symmetry folding + Gaussian smoothing (sigma=0.8)
2. PINN: Interpolation to 50 points if needed
3. Errors: L2 Relative (%) and Maximum Absolute
4. Plot: SCI journal standard with hollow markers and solid lines

Output:
- Console: Quantitative error metrics
- Figure: High-resolution validation plot (600 DPI)

Author: Computational Physics Team
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import argparse

# =============================================================================
# PUBLICATION STYLE CONFIGURATION
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
})


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def process_mc_benchmark(npy_file, nx=50):
    """
    Process FMC benchmark data with symmetry folding and Gaussian smoothing.
    
    Steps:
    1. Extract centerline (y=nx//2, z=nx//2)
    2. Symmetry folding around x=0.5 (double sample size)
    3. Light Gaussian filter (sigma=0.8)
    
    Parameters:
        npy_file: Path to 3D FMC .npy file (nx, nx, nx)
        nx: Grid resolution (default 50)
    
    Returns:
        G_smooth: Processed 1D centerline array (nx points)
    """
    # Load 3D field
    G_field = np.load(npy_file)
    
    # Extract centerline: y=nx//2, z=nx//2
    iy_center, iz_center = nx // 2, nx // 2
    G_line_raw = G_field[:, iy_center, iz_center]
    
    # Symmetry folding around center (x=0.5)
    mid = nx // 2
    left_half = G_line_raw[:mid]
    right_half = G_line_raw[mid:][::-1]
    symmetric_half = (left_half + right_half) / 2.0
    G_symmetric = np.concatenate([symmetric_half, symmetric_half[::-1]])
    
    # Light Gaussian smoothing to remove micro-noise
    G_smooth = gaussian_filter1d(G_symmetric, sigma=0.8)
    
    return G_smooth


def align_pinn_data(G_pinn, nx_target=50):
    """
    Interpolate PINN data to match FMC grid points if needed.
    
    Parameters:
        G_pinn: PINN centerline array (may be different length)
        nx_target: Target number of points (default 50 for FMC)
    
    Returns:
        G_aligned: Interpolated array with nx_target points
    """
    n_pinn = len(G_pinn)
    
    if n_pinn == nx_target:
        # Already aligned
        return G_pinn
    else:
        # Interpolate to target grid
        x_pinn = np.linspace(0, 1, n_pinn)
        x_target = np.linspace(0, 1, nx_target)
        G_aligned = np.interp(x_target, x_pinn, G_pinn)
        return G_aligned


def calculate_errors(pred, exact):
    """
    Calculate quantitative error metrics.
    
    Parameters:
        pred: Predicted values (PINN)
        exact: Exact/reference values (FMC)
    
    Returns:
        l2_error: L2 relative error (%)
        max_abs_error: Maximum absolute error
    """
    # L2 Relative Error (%)
    l2_error = np.linalg.norm(pred - exact) / np.linalg.norm(exact) * 100.0
    
    # Maximum Absolute Error
    max_abs_error = np.max(np.abs(pred - exact))
    
    return l2_error, max_abs_error


# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def plot_validation(G_MC_B, G_PINN_B, G_MC_C, G_PINN_C, 
                   l2_B, max_B, l2_C, max_C,
                   output_file='Validation_PINN_vs_FMC.png'):
    """
    Generate publication-quality validation plot.
    
    Style:
    - FMC: Hollow markers (circles for B, squares for C)
    - PINN: Solid/dashed lines (blue for B, red for C)
    - Lines drawn OVER markers
    - Frameless legend
    """
    # Spatial coordinate
    x_vals = np.linspace(0, 1, 50)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    # Case B: Isotropic (g=0.0)
    # FMC Benchmark - hollow blue circles
    ax.plot(x_vals, G_MC_B, 'bo', 
            markerfacecolor='none', 
            markersize=6, 
            markeredgewidth=1.2,
            label='FMC Benchmark (Case B, $g=0.0$)')
    
    # PINN Prediction - solid blue line
    ax.plot(x_vals, G_PINN_B, 'b-', 
            linewidth=2.0, 
            label='PINN Prediction (Case B)')
    
    # Case C: Anisotropic (g=0.8)
    # FMC Benchmark - hollow red squares
    ax.plot(x_vals, G_MC_C, 'rs', 
            markerfacecolor='none', 
            markersize=6, 
            markeredgewidth=1.2,
            label='FMC Benchmark (Case C, $g=0.8$)')
    
    # PINN Prediction - dashed red line
    ax.plot(x_vals, G_PINN_C, 'r--', 
            linewidth=2.0, 
            label='PINN Prediction (Case C)')
    
    # Labels with LaTeX formatting
    ax.set_xlabel('Spatial Position $x$', fontsize=12)
    ax.set_ylabel('Incident Radiation $G(x, 0.5, 0.5)$', fontsize=12)
    ax.set_title('Validation: 3D PINN vs FMC Benchmark', 
                 fontsize=13, fontweight='bold')
    
    # Legend (frameless, best location)
    ax.legend(frameon=False, loc='best', fontsize=10)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    
    # Axis limits with padding
    ax.set_xlim(-0.02, 1.02)
    y_min = min(G_MC_B.min(), G_MC_C.min(), G_PINN_B.min(), G_PINN_C.min())
    y_max = max(G_MC_B.max(), G_MC_C.max(), G_PINN_B.max(), G_PINN_C.max())
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Tight layout
    plt.tight_layout()
    
    # Save high-resolution figure
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\nHigh-resolution figure saved to: {output_file}")
    
    # Also save PDF for vector graphics
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    print(f"Vector PDF saved to: {pdf_file}")
    
    plt.show()


# =============================================================================
# MAIN VALIDATION PIPELINE
# =============================================================================

def main():
    """Main validation pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Validate PINN predictions against FMC benchmarks'
    )
    parser.add_argument(
        '--fmc-b',
        type=str,
        default='../Solvers/MC/MC3D_Results/FMC_G_3D_CaseB_FIXED_HighStats.npy',
        help='Path to FMC Case B .npy file'
    )
    parser.add_argument(
        '--fmc-c',
        type=str,
        default='../Solvers/MC/MC3D_Results/FMC_G_3D_CaseC_FIXED_HighStats.npy',
        help='Path to FMC Case C .npy file'
    )
    parser.add_argument(
        '--pinn',
        type=str,
        default='G_centerline_data.npz',
        help='Path to PINN predictions .npz file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Validation_PINN_vs_FMC.png',
        help='Output figure filename'
    )
    args = parser.parse_args()
    
    # Header
    print("="*70)
    print("PINN Validation against FMC Benchmark")
    print("="*70)
    print("\nData Processing:")
    print("  - FMC: Symmetry folding + Gaussian smoothing (σ=0.8)")
    print("  - PINN: Interpolation to 50 points if needed")
    print("\nError Metrics:")
    print("  - L2 Relative Error (%)")
    print("  - Maximum Absolute Error")
    print("="*70 + "\n")
    
    # Step 1: Load and process FMC benchmark data
    print("Loading FMC benchmark data...")
    
    if not os.path.exists(args.fmc_b):
        # Try alternative paths
        alt_path_b = '../Solvers/MC/MC3D_Results/FMC_G_3D_CaseB_FIXED.npy'
        if os.path.exists(alt_path_b):
            args.fmc_b = alt_path_b
        else:
            raise FileNotFoundError(f"FMC Case B file not found: {args.fmc_b}")
    
    if not os.path.exists(args.fmc_c):
        alt_path_c = '../Solvers/MC/MC3D_Results/FMC_G_3D_CaseC_FIXED.npy'
        if os.path.exists(alt_path_c):
            args.fmc_c = alt_path_c
        else:
            raise FileNotFoundError(f"FMC Case C file not found: {args.fmc_c}")
    
    G_MC_B = process_mc_benchmark(args.fmc_b, nx=50)
    G_MC_C = process_mc_benchmark(args.fmc_c, nx=50)
    print(f"  ✓ FMC Case B: {args.fmc_b}")
    print(f"  ✓ FMC Case C: {args.fmc_c}")
    
    # Step 2: Load PINN predictions
    print("\nLoading PINN predictions...")
    
    if not os.path.exists(args.pinn):
        # Try alternative paths
        alt_paths = [
            '../G_centerline_data.npz',
            '../Results_3D_CaseB/G_centerline_data.npz',
            'G_centerline_data.npz'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                args.pinn = alt_path
                break
        else:
            raise FileNotFoundError(f"PINN data file not found: {args.pinn}")
    
    pinn_data = np.load(args.pinn)
    
    # Extract PINN centerlines (handle different key names)
    if 'CaseB' in pinn_data:
        G_PINN_B_raw = pinn_data['CaseB']
        G_PINN_C_raw = pinn_data['CaseC']
    elif 'caseB' in pinn_data:
        G_PINN_B_raw = pinn_data['caseB']
        G_PINN_C_raw = pinn_data['caseC']
    elif 'G_pinn_B' in pinn_data:
        G_PINN_B_raw = pinn_data['G_pinn_B']
        G_PINN_C_raw = pinn_data['G_pinn_C']
    else:
        # Assume first two arrays are Case B and C
        keys = list(pinn_data.keys())
        G_PINN_B_raw = pinn_data[keys[0]]
        G_PINN_C_raw = pinn_data[keys[1]]
    
    print(f"  ✓ PINN data: {args.pinn}")
    print(f"    Case B shape: {G_PINN_B_raw.shape}")
    print(f"    Case C shape: {G_PINN_C_raw.shape}")
    
    # Step 3: Align PINN data to FMC grid (50 points)
    print("\nAligning PINN data to FMC grid...")
    G_PINN_B = align_pinn_data(G_PINN_B_raw, nx_target=50)
    G_PINN_C = align_pinn_data(G_PINN_C_raw, nx_target=50)
    print(f"  ✓ Aligned to 50 points")
    
    # Step 4: Calculate errors
    print("\n" + "="*70)
    print("QUANTITATIVE ERROR METRICS")
    print("="*70)
    
    l2_B, max_B = calculate_errors(G_PINN_B, G_MC_B)
    l2_C, max_C = calculate_errors(G_PINN_C, G_MC_C)
    
    print(f"\nCase B (Isotropic, g=0.0):")
    print(f"  L2 Relative Error:  {l2_B:8.3f}%")
    print(f"  Maximum Abs Error:  {max_B:8.5f}")
    print(f"  FMC Center Value:   {G_MC_B[25]:8.5f}")
    print(f"  PINN Center Value:  {G_PINN_B[25]:8.5f}")
    
    print(f"\nCase C (Anisotropic, g=0.8):")
    print(f"  L2 Relative Error:  {l2_C:8.3f}%")
    print(f"  Maximum Abs Error:  {max_C:8.5f}")
    print(f"  FMC Center Value:   {G_MC_C[25]:8.5f}")
    print(f"  PINN Center Value:  {G_PINN_C[25]:8.5f}")
    
    print("="*70)
    
    # Step 5: Generate validation plot
    print("\nGenerating validation plot...")
    plot_validation(G_MC_B, G_PINN_B, G_MC_C, G_PINN_C,
                   l2_B, max_B, l2_C, max_C,
                   output_file=args.output)
    
    print("\n✓ Validation complete!")


if __name__ == "__main__":
    main()
