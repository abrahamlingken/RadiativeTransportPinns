#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_pinn_vs_fmc_fixed.py - Corrected PINN Validation against FMC Benchmark
================================================================================

BUG FIXES APPLIED:
1. Centerline Extraction: Average 4 central cells (indices 24,25) to hit physical 
   bullseye at (y=0.5, z=0.5) for cell-centered grid
2. Coordinate Alignment: Explicitly map node-centered PINN to cell-centered FMC 
   coordinates using np.interp with correct physical positions

Calculates quantitative errors (L2 Relative %, Max Absolute) and generates 
publication-quality overlay plots.

Author: Computational Physics Team
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import argparse

# =============================================================================
# SCI PUBLICATION STYLE
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
# CORRECTED DATA PROCESSING FUNCTIONS
# =============================================================================

def process_mc_benchmark(npy_file, nx=50):
    """
    Process FMC benchmark data with corrected centerline extraction.
    
    BUG FIX 1: For cell-centered grid, physical center (y=0.5, z=0.5) lies 
    between indices 24 and 25. We average the 4 central cells to hit the 
    exact physical bullseye.
    
    Parameters:
        npy_file: Path to 3D FMC .npy file (nx, nx, nx)
        nx: Grid resolution (default 50)
    
    Returns:
        G_smooth: Processed 1D centerline array (nx points)
    """
    G_field = np.load(npy_file)
    
    # BUG FIX 1: Exact Physical Bullseye (Average 4 central cells for even grid)
    # For nx=50, indices 24 and 25 straddle the physical center y=z=0.5
    iy1, iy2 = nx // 2 - 1, nx // 2  # indices 24, 25
    iz1, iz2 = nx // 2 - 1, nx // 2  # indices 24, 25
    
    # Average over the 4 central cells: [iy1:iy2+1, iz1:iz2+1] = [24:26, 24:26]
    G_line_raw = np.mean(G_field[:, iy1:iy2+1, iz1:iz2+1], axis=(1, 2))
    
    # 2. Symmetry Folding around center (x=0.5)
    mid = nx // 2
    left_half = G_line_raw[:mid]
    right_half = G_line_raw[mid:][::-1]
    symmetric_half = (left_half + right_half) / 2.0
    G_symmetric = np.concatenate([symmetric_half, symmetric_half[::-1]])
    
    # 3. Light Gaussian smoothing to remove micro-noise
    G_smooth = gaussian_filter1d(G_symmetric, sigma=0.8)
    
    return G_smooth


def align_pinn_data(G_pinn, nx_target=50):
    """
    Align PINN data to FMC cell-centered coordinates.
    
    BUG FIX 2: PINN is Node-centered (boundary inclusive: linspace(0, 1, n_pinn))
    FMC is Cell-centered (linspace(0.5/nx, 1-0.5/nx, nx))
    
    We MUST explicitly map PINN to FMC physical coordinates using np.interp
    to avoid Delta_x/2 spatial shift.
    
    Parameters:
        G_pinn: PINN centerline array (may be different length)
        nx_target: Target number of points (default 50 for FMC)
    
    Returns:
        G_aligned: Interpolated array with nx_target points at cell centers
    """
    n_pinn = len(G_pinn)
    
    if n_pinn == nx_target:
        # Same length, but still need to check if coordinates align
        # PINN node-centered coordinates
        x_pinn = np.linspace(0, 1, n_pinn)
        # FMC cell-centered coordinates
        x_target = np.linspace(0.5/nx_target, 1 - 0.5/nx_target, nx_target)
        
        # Even if same length, interpolate to ensure correct physical positions
        G_aligned = np.interp(x_target, x_pinn, G_pinn)
    else:
        # Different lengths - explicit coordinate mapping
        # PINN is Node-centered (Boundary inclusive)
        x_pinn = np.linspace(0, 1, n_pinn)
        # FMC is Cell-centered
        x_target = np.linspace(0.5/nx_target, 1 - 0.5/nx_target, nx_target)
        
        # Map PINN strictly to FMC physical coordinates
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
    Generate publication-quality validation plot with CORRECT cell-centered x-axis.
    
    Style:
    - FMC: Hollow markers (circles for B, squares for C)
    - PINN: Solid/dashed lines (blue for B, red for C)
    - Lines drawn OVER markers
    - Frameless legend
    """
    # BUG FIX: Correct X-axis MUST use cell-centered coordinates
    # NOT linspace(0, 1, 50) which would be node-centered
    nx = 50
    x_vals = np.linspace(0.5/nx, 1 - 0.5/nx, nx)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    # Case B: Isotropic (g=0.0)
    # FMC Benchmark - hollow blue circles (drawn first, underneath)
    ax.plot(x_vals, G_MC_B, 'bo', 
            markerfacecolor='none', 
            markersize=6, 
            markeredgewidth=1.2,
            label='FMC Benchmark (Case B, $g=0.0$)',
            zorder=1)
    
    # PINN Prediction - solid blue line (drawn second, on top)
    ax.plot(x_vals, G_PINN_B, 'b-', 
            linewidth=2.0, 
            label='PINN Prediction (Case B)',
            zorder=2)
    
    # Case C: Anisotropic (g=0.8)
    # FMC Benchmark - hollow red squares
    ax.plot(x_vals, G_MC_C, 'rs', 
            markerfacecolor='none', 
            markersize=6, 
            markeredgewidth=1.2,
            label='FMC Benchmark (Case C, $g=0.8$)',
            zorder=1)
    
    # PINN Prediction - dashed red line
    ax.plot(x_vals, G_PINN_C, 'r--', 
            linewidth=2.0, 
            label='PINN Prediction (Case C)',
            zorder=2)
    
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
    """Main validation pipeline with corrected physics."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Validate PINN predictions against FMC benchmarks (CORRECTED)'
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
        default='../Figures_3D/G_centerline_data.npz',
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
    print("PINN Validation against FMC Benchmark (PHYSICS-CORRECTED)")
    print("="*70)
    print("\nBug Fixes Applied:")
    print("  1. Centerline: Average 4 central cells (indices 24,25)")
    print("     to hit physical bullseye at (y=0.5, z=0.5)")
    print("  2. Coordinates: Map node-centered PINN to cell-centered FMC")
    print("     using explicit np.interp to avoid spatial shift")
    print("\nError Metrics:")
    print("  - L2 Relative Error (%)")
    print("  - Maximum Absolute Error")
    print("="*70 + "\n")
    
    # Step 1: Load and process FMC benchmark data
    print("Loading and processing FMC benchmark data...")
    
    # Try alternative paths if defaults don't exist
    if not os.path.exists(args.fmc_b):
        alt_paths_b = [
            '../Solvers/MC/MC3D_Results/FMC_G_3D_CaseB_FIXED.npy',
            '../MC3D_Results/FMC_G_3D_CaseB_FIXED.npy',
            'FMC_G_3D_CaseB_FIXED.npy'
        ]
        for alt in alt_paths_b:
            if os.path.exists(alt):
                args.fmc_b = alt
                break
        else:
            raise FileNotFoundError(f"FMC Case B file not found: {args.fmc_b}")
    
    if not os.path.exists(args.fmc_c):
        alt_paths_c = [
            '../Solvers/MC/MC3D_Results/FMC_G_3D_CaseC_FIXED.npy',
            '../MC3D_Results/FMC_G_3D_CaseC_FIXED.npy',
            'FMC_G_3D_CaseC_FIXED.npy'
        ]
        for alt in alt_paths_c:
            if os.path.exists(alt):
                args.fmc_c = alt
                break
        else:
            raise FileNotFoundError(f"FMC Case C file not found: {args.fmc_c}")
    
    G_MC_B = process_mc_benchmark(args.fmc_b, nx=50)
    G_MC_C = process_mc_benchmark(args.fmc_c, nx=50)
    print(f"  ✓ FMC Case B processed: {args.fmc_b}")
    print(f"  ✓ FMC Case C processed: {args.fmc_c}")
    
    # Step 2: Load PINN predictions
    print("\nLoading PINN predictions...")
    
    if not os.path.exists(args.pinn):
        alt_paths_pinn = [
            '../Figures_3D/G_centerline_data.npz',
            'G_centerline_data.npz',
            '../Results_3D_CaseB/G_centerline_data.npz'
        ]
        for alt in alt_paths_pinn:
            if os.path.exists(alt):
                args.pinn = alt
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
        keys = list(pinn_data.keys())
        print(f"  Available keys: {keys}")
        G_PINN_B_raw = pinn_data[keys[0]]
        G_PINN_C_raw = pinn_data[keys[1]]
    
    print(f"  ✓ PINN data loaded: {args.pinn}")
    print(f"    Case B raw shape: {G_PINN_B_raw.shape}")
    print(f"    Case C raw shape: {G_PINN_C_raw.shape}")
    
    # Step 3: Align PINN data to FMC cell-centered grid
    print("\nAligning PINN data to FMC cell-centered coordinates...")
    G_PINN_B = align_pinn_data(G_PINN_B_raw, nx_target=50)
    G_PINN_C = align_pinn_data(G_PINN_C_raw, nx_target=50)
    print(f"  ✓ Aligned to 50 cell-centered points")
    
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
