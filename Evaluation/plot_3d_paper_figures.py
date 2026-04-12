#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_3d_paper_figures.py - PINN vs FMC Validation with Dynamic Inference
========================================================================

Publication-quality validation script that:
1. Loads FMC benchmark arrays and extracts smoothed centerline
2. Dynamically infers PINN predictions by loading model.pkl and computing G on-the-fly
3. Calculates quantitative error metrics (L2 Relative, Max Absolute)
4. Generates SCI-journal standard overlay plot

Critical Physics Constraints:
- Coordinate Alignment: FMC cell-centered (0.5/50, 1-0.5/50, 50)
- FMC Center Extraction: Average 4 central cells (indices 24,25)

Author: Computational Physics Team
Date: 2024
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# SCI Plotting Style
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
    'xtick.minor.visible': True,
    'ytick.minor.visible': True
})

# Add paths for Custom OOP Engine
# Script is in Evaluation/, need to go up one level to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up from Evaluation to project root
if PROJECT_ROOT not in sys.path: 
    sys.path.insert(0, PROJECT_ROOT)
CORE_PATH = os.path.join(PROJECT_ROOT, 'Core')
if CORE_PATH not in sys.path: 
    sys.path.insert(0, CORE_PATH)

# Also add parent of project root (for imports)
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
if PARENT_ROOT not in sys.path:
    sys.path.insert(0, PARENT_ROOT)

from EquationModels import RadTrans3D_Complex as Ec
from ModelClassTorch2 import Pinns


def process_mc_benchmark(npy_file, nx=50):
    """
    Process FMC benchmark data with corrected centerline extraction.
    
    Critical: Average 4 central cells to hit exact physical bullseye at (0.5, 0.5)
    """
    if not os.path.exists(npy_file): 
        return None
    
    G_field = np.load(npy_file)
    
    # Average the 4 central cells to hit the exact physical bullseye
    # For even grid (50x50), center lies between indices 24 and 25
    iy1, iy2 = nx // 2 - 1, nx // 2  # 24, 25
    iz1, iz2 = nx // 2 - 1, nx // 2  # 24, 25
    G_line_raw = np.mean(G_field[:, iy1:iy2+1, iz1:iz2+1], axis=(1, 2))
    
    # Symmetry folding around center (x=0.5)
    mid = nx // 2
    left_half = G_line_raw[:mid]
    right_half = G_line_raw[mid:][::-1]
    symmetric_half = (left_half + right_half) / 2.0
    G_symmetric = np.concatenate([symmetric_half, symmetric_half[::-1]])
    
    # Light Gaussian smoothing to remove micro-noise
    return gaussian_filter1d(G_symmetric, sigma=0.8)


def load_model_and_compute_G(case_folder, x_tensor, y_tensor, z_tensor, engine):
    """
    Load PINN model and compute G field dynamically.
    
    Args:
        case_folder: Path to folder containing model.pkl
        x_tensor, y_tensor, z_tensor: Coordinate tensors (already on correct device)
        engine: RadTrans3D_Physics instance
    
    Returns:
        G: numpy array of incident radiation
    """
    model_path = os.path.join(case_folder, 'model.pkl')
    if not os.path.exists(model_path): 
        return None
    
    model = torch.load(model_path, map_location=engine.dev, weights_only=False)
    model.eval()
    
    with torch.no_grad():
        G_tensor = engine.compute_incident_radiation(x_tensor, y_tensor, z_tensor, model)
    
    return G_tensor.cpu().numpy().flatten()


def calculate_errors(pred, exact):
    """
    Calculate quantitative error metrics.
    
    Returns:
        l2_error: L2 relative error (%)
        max_abs: Maximum absolute error
    """
    l2_error = np.linalg.norm(pred - exact) / np.linalg.norm(exact) * 100.0
    max_abs = np.max(np.abs(pred - exact))
    return l2_error, max_abs


def main():
    """Main validation pipeline with dynamic inference."""
    
    # Debug: Print paths
    print("="*70)
    print("PINN vs FMC Validation - Dynamic Inference")
    print("="*70)
    print(f"Script dir: {SCRIPT_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Core path: {CORE_PATH}")
    print(f"Python path: {sys.path[:3]}")  # Show first 3 paths
    
    # Setup device and physics engine
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine = Ec.RadTrans3D_Physics(dev=device)
    
    print(f"Device: {device}")
    
    # Strictly align query tensor with FMC Cell Centers (N=50)
    # CRITICAL: Cell-centered coordinates, NOT node-centered
    n_points = 50
    x_np = np.linspace(0.5/n_points, 1 - 0.5/n_points, n_points)
    # compute_incident_radiation expects 1D tensors, not 2D
    x_1d = torch.tensor(x_np, dtype=torch.float32, device=device)  # 1D: (50,)
    y_1d = torch.ones_like(x_1d) * 0.5  # 1D: (50,)
    z_1d = torch.ones_like(x_1d) * 0.5  # 1D: (50,)
    
    print(f"Coordinate grid: Cell-centered, {n_points} points")
    print(f"  x range: [{x_np[0]:.4f}, {x_np[-1]:.4f}]")
    
    # --- Robust Path Configuration ---
    # Binding model folders to PROJECT_ROOT to prevent execution-directory bugs.
    # FMC files: Try UltraStats first, then HighStats, then standard.
    
    configs = {
        'Case B (Isotropic)': {
            'model_folder': os.path.join(PROJECT_ROOT, 'Results_3D_CaseB'),
            'fmc_file': '../Solvers/MC/MC3D_Results/FMC_G_3D_CaseB_FIXED_UltraStats.npy',
            'alt_fmc_files': [
                '../Solvers/MC/MC3D_Results/FMC_G_3D_CaseB_FIXED_HighStats.npy',
                '../Solvers/MC/MC3D_Results/FMC_G_3D_CaseB_FIXED.npy'
            ],
            'c': 'blue', 
            'm': 'o',
            'linestyle': '-'
        },
        'Case C (Anisotropic)': {
            'model_folder': os.path.join(PROJECT_ROOT, 'Results_3D_CaseC'),
            'fmc_file': '../Solvers/MC/MC3D_Results/FMC_G_3D_CaseC_FIXED_UltraStats.npy',
            'alt_fmc_files': [
                '../Solvers/MC/MC3D_Results/FMC_G_3D_CaseC_FIXED_HighStats.npy',
                '../Solvers/MC/MC3D_Results/FMC_G_3D_CaseC_FIXED.npy'
            ],
            'c': 'red', 
            'm': 's',
            'linestyle': '--'
        }
    }
    
    # Storage for results
    results = {}
    
    # Loop through configs
    print("\nProcessing cases...")
    for case_name, config in configs.items():
        print(f"\n  [{case_name}]")
        
        # Try FMC files in order: UltraStats -> HighStats -> standard
        fmc_file = config['fmc_file']
        if not os.path.exists(fmc_file):
            for alt_file in config['alt_fmc_files']:
                if os.path.exists(alt_file):
                    fmc_file = alt_file
                    break
        
        # Load FMC benchmark
        G_fmc = process_mc_benchmark(fmc_file, nx=50)
        if G_fmc is None:
            print(f"    ⚠ FMC file not found: {fmc_file}")
            continue
        print(f"    ✓ FMC loaded: {fmc_file}")
        
        # Dynamically infer PINN prediction
        G_pinn = load_model_and_compute_G(
            config['model_folder'], x_1d, y_1d, z_1d, engine
        )
        if G_pinn is None:
            print(f"    ⚠ PINN model not found in: {config['model_folder']}")
            continue
        print(f"    ✓ PINN inferred from: {config['model_folder']}/model.pkl")
        
        # Calculate errors
        l2_err, max_err = calculate_errors(G_pinn, G_fmc)
        
        results[case_name] = {
            'G_fmc': G_fmc,
            'G_pinn': G_pinn,
            'l2_error': l2_err,
            'max_error': max_err,
            'config': config
        }
        
        print(f"    L2 Relative Error: {l2_err:.3f}%")
        print(f"    Max Absolute Error: {max_err:.5f}")
    
    # Print summary table
    print("\n" + "="*70)
    print("QUANTITATIVE ERROR METRICS")
    print("="*70)
    for case_name, data in results.items():
        print(f"\n{case_name}:")
        print(f"  L2 Relative Error:  {data['l2_error']:8.3f}%")
        print(f"  Maximum Abs Error:  {data['max_error']:8.5f}")
        print(f"  FMC Center Value:   {data['G_fmc'][25]:8.5f}")
        print(f"  PINN Center Value:  {data['G_pinn'][25]:8.5f}")
    print("="*70)
    
    # Generate publication-quality plot
    print("\nGenerating validation plot...")
    
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    for case_name, data in results.items():
        cfg = data['config']
        
        # FMC Benchmark - hollow markers (drawn first, underneath)
        ax.plot(x_np, data['G_fmc'], 
                marker=cfg['m'], 
                color=cfg['c'],
                markerfacecolor='none', 
                markersize=6, 
                markeredgewidth=1.2,
                linestyle='none',
                label=f'FMC Benchmark ({case_name})',
                zorder=1)
        
        # PINN Prediction - solid/dashed line (drawn second, on top)
        ax.plot(x_np, data['G_pinn'], 
                color=cfg['c'],
                linestyle=cfg['linestyle'],
                linewidth=2.0,
                label=f'PINN Prediction ({case_name})',
                zorder=2)
    
    # Labels and formatting
    ax.set_xlabel('Spatial Position $x$', fontsize=12)
    ax.set_ylabel('Incident Radiation $G(x, 0.5, 0.5)$', fontsize=12)
    ax.set_title('Validation: 3D PINN vs FMC Benchmark', 
                 fontsize=13, fontweight='bold')
    
    # Legend (frameless)
    ax.legend(frameon=False, loc='best', fontsize=9)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    
    # Axis limits
    ax.set_xlim(-0.02, 1.02)
    
    plt.tight_layout()
    
    # Save high-resolution figure
    output_file = 'Validation_PINN_vs_FMC_Final.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\n  ✓ Saved: {output_file}")
    
    # Also save PDF
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {pdf_file}")
    
    plt.show()
    
    print("\n✓ Validation complete!")


if __name__ == "__main__":
    main()
