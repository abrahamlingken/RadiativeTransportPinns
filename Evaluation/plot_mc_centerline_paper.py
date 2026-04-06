#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_mc_centerline_paper.py - MC Centerline Visualization for SCI Publication
==============================================================================

Extracts, smooths, and plots the centerline profile of Monte Carlo G-field data
using physics-based symmetry folding and light Gaussian filtering.

Physics Strategy:
1. Symmetry Folding: Exploit center symmetry at (0.5, 0.5, 0.5) to double sample size
2. Light Gaussian: sigma=0.8 to preserve macro-gradients while removing micro-noise

Output: Publication-ready figure with Times New Roman font, inward ticks, hollow markers.

Author: Computational Physics Team
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

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

def symmetry_fold_centerline(G_line_raw, nx=50):
    """
    Apply physics-based symmetry folding to centerline data.
    
    The heat source is centered at (0.5, 0.5, 0.5), so the centerline
    along X must be perfectly symmetric around x=0.5.
    
    Strategy:
    - Split into left (0-24) and right (25-49) halves
    - Flip right half and average with left: symmetric_half = (left + right[::-1]) / 2
    - Reconstruct full array by mirroring
    
    This doubles effective sample size and cancels asymmetric noise.
    
    Parameters:
        G_line_raw: Raw 1D centerline array (50 points)
        nx: Number of grid points (default 50)
    
    Returns:
        G_symmetric: Symmetry-folded centerline (50 points)
    """
    mid = nx // 2  # 25
    
    # Left half: indices 0 to 24
    left_half = G_line_raw[:mid]
    
    # Right half: indices 25 to 49, reversed
    right_half = G_line_raw[mid:][::-1]
    
    # Average to enforce symmetry (doubles sample size)
    symmetric_half = (left_half + right_half) / 2.0
    
    # Reconstruct full array: [symmetric_half, symmetric_half reversed]
    G_symmetric = np.concatenate([symmetric_half, symmetric_half[::-1]])
    
    return G_symmetric


def process_and_smooth_centerline(npy_file, nx=50, sigma=0.8):
    """
    Complete processing pipeline for MC centerline data.
    
    Steps:
    1. Load 3D G-field and extract centerline (y=nx//2, z=nx//2)
    2. Apply symmetry folding to exploit physical symmetry
    3. Apply light Gaussian filter to remove residual noise
    
    Parameters:
        npy_file: Path to .npy file containing G-field (nx, nx, nx)
        nx: Grid resolution (default 50)
        sigma: Gaussian filter width (default 0.8 for light smoothing)
    
    Returns:
        G_smooth: Processed and smoothed centerline (nx points)
        G_symmetric: Intermediate symmetry-folded result (for comparison)
        G_raw: Raw extracted centerline (for comparison)
    """
    # Load data
    G_field = np.load(npy_file)
    
    # Extract centerline: y=nx//2, z=nx//2
    iy_center, iz_center = nx // 2, nx // 2
    G_raw = G_field[:, iy_center, iz_center]
    
    # Step 1: Symmetry folding
    G_symmetric = symmetry_fold_centerline(G_raw, nx)
    
    # Step 2: Light Gaussian filtering
    G_smooth = gaussian_filter1d(G_symmetric, sigma=sigma)
    
    return G_smooth, G_symmetric, G_raw


# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def plot_mc_benchmark(case_b_file, case_c_file, output_dir='MC3D_Results'):
    """
    Create publication-ready plot comparing Case B and Case C centerlines.
    
    Parameters:
        case_b_file: Path to Case B .npy file
        case_c_file: Path to Case C .npy file
        output_dir: Directory for output figure
    """
    # X-axis: normalized spatial coordinate [0, 1]
    nx = 50
    x_vals = np.linspace(0, 1, nx)
    
    # Process both cases
    print("Processing Case B (Isotropic, g=0.0)...")
    G_caseB, G_sym_B, G_raw_B = process_and_smooth_centerline(case_b_file, nx)
    
    print("Processing Case C (Anisotropic, g=0.8)...")
    G_caseC, G_sym_C, G_raw_C = process_and_smooth_centerline(case_c_file, nx)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot with hollow markers (markerfacecolor='none')
    # Case B: Blue circles
    ax.plot(x_vals, G_caseB, 'bo', markerfacecolor='none', 
            markersize=6, markeredgewidth=1.2,
            label='MC Benchmark (Isotropic, $g=0.0$)')
    
    # Case C: Red squares
    ax.plot(x_vals, G_caseC, 'rs', markerfacecolor='none', 
            markersize=6, markeredgewidth=1.2,
            label='MC Benchmark (Anisotropic, $g=0.8$)')
    
    # Labels with LaTeX formatting
    ax.set_xlabel('Spatial Position $x$', fontsize=12)
    ax.set_ylabel('Incident Radiation $G(x, 0.5, 0.5)$', fontsize=12)
    
    # Legend
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Axis limits (auto with some padding)
    ax.set_xlim(-0.02, 1.02)
    y_min = min(G_caseB.min(), G_caseC.min())
    y_max = max(G_caseB.max(), G_caseC.max())
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Tight layout
    plt.tight_layout()
    
    # Save high-resolution figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'MC_Centerline_Paper_HighRes.png')
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\nHigh-resolution figure saved to: {output_file}")
    
    # Also save as PDF for vector graphics
    pdf_file = os.path.join(output_dir, 'MC_Centerline_Paper.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    print(f"Vector PDF saved to: {pdf_file}")
    
    # Show statistics
    print("\n" + "="*60)
    print("Processing Statistics")
    print("="*60)
    print(f"Case B (Isotropic):")
    print(f"  Raw center:     {G_raw_B[nx//2]:.6f}")
    print(f"  Smoothed center:{G_caseB[nx//2]:.6f}")
    print(f"  Max:            {G_caseB.max():.6f}")
    print(f"  Min:            {G_caseB.min():.6e}")
    print(f"\nCase C (Anisotropic):")
    print(f"  Raw center:     {G_raw_C[nx//2]:.6f}")
    print(f"  Smoothed center:{G_caseC[nx//2]:.6f}")
    print(f"  Max:            {G_caseC.max():.6f}")
    print(f"  Min:            {G_caseC.min():.6e}")
    print("="*60)
    
    plt.show()
    
    return G_caseB, G_caseC


def plot_comparison_stages(case_file, nx=50, output_dir='MC3D_Results'):
    """
    Create a diagnostic plot showing all processing stages:
    Raw -> Symmetry Folded -> Gaussian Smoothed
    
    Useful for verifying the smoothing pipeline.
    """
    G_smooth, G_sym, G_raw = process_and_smooth_centerline(case_file, nx)
    x_vals = np.linspace(0, 1, nx)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Raw data (light gray, small dots)
    ax.plot(x_vals, G_raw, 'k.', alpha=0.3, markersize=3, label='Raw MC Data')
    
    # Symmetry folded (blue dashed)
    ax.plot(x_vals, G_sym, 'b--', linewidth=1.5, alpha=0.7, 
            label='After Symmetry Folding')
    
    # Final smoothed (red solid)
    ax.plot(x_vals, G_smooth, 'r-', linewidth=2, 
            label='After Gaussian Filter ($\\sigma=0.8$)')
    
    ax.set_xlabel('Spatial Position $x$', fontsize=12)
    ax.set_ylabel('Incident Radiation $G(x, 0.5, 0.5)$', fontsize=12)
    ax.set_title('MC Data Smoothing Pipeline', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.set_xlim(-0.02, 1.02)
    
    plt.tight_layout()
    
    case_name = 'CaseB' if 'CaseB' in case_file else 'CaseC'
    output_file = os.path.join(output_dir, f'MC_SmoothingStages_{case_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Smoothing stages plot saved to: {output_file}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Plot MC centerline with physics-based smoothing'
    )
    parser.add_argument(
        '--case-b',
        type=str,
        default='../Solvers/MC/MC3D_Results/FMC_G_3D_CaseB_FIXED_HighStats.npy',
        help='Path to Case B .npy file'
    )
    parser.add_argument(
        '--case-c',
        type=str,
        default='../Solvers/MC/MC3D_Results/FMC_G_3D_CaseC_FIXED_HighStats.npy',
        help='Path to Case C .npy file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='MC3D_Results',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--stages',
        action='store_true',
        help='Also generate diagnostic plots showing smoothing stages'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("MC Centerline Visualization - Publication Quality")
    print("="*70)
    print("\nSmoothing Strategy:")
    print("  1. Symmetry Folding: Exploit center symmetry at x=0.5")
    print("  2. Light Gaussian: sigma=0.8 (preserves macro-gradients)")
    print("\nPlot Style:")
    print("  - Font: Times New Roman, 12pt")
    print("  - Ticks: Inward-facing, all four sides")
    print("  - Markers: Hollow (for overlay compatibility)")
    print("  - Output: 600 DPI PNG + Vector PDF")
    print("="*70 + "\n")
    
    # Main comparison plot
    G_caseB, G_caseC = plot_mc_benchmark(args.case_b, args.case_c, args.output_dir)
    
    # Optional diagnostic plots
    if args.stages:
        print("\nGenerating diagnostic plots...")
        plot_comparison_stages(args.case_b, output_dir=args.output_dir)
        plot_comparison_stages(args.case_c, output_dir=args.output_dir)
    
    print("\nDone! Publication-ready figures generated.")


if __name__ == "__main__":
    main()
