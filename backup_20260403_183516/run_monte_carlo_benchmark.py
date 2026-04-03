#!/usr/bin/env python
"""
Quick launcher for Monte Carlo benchmark generation.
Runs both Case B and Case C with reduced photon count for testing.
For production runs, edit monte_carlo_3d_rte_benchmark.py directly.
"""

import sys
import os

# Import the main solver and override parameters
import monte_carlo_3d_rte_benchmark as mc

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        case = sys.argv[1].upper()
        if case in ['B', 'C']:
            print(f"Running Case {case}...")
            mc.run_monte_carlo(case)
        else:
            print("Usage: python run_monte_carlo_benchmark.py [B|C]")
            print("  B: Isotropic scattering (κ=0.5, σs=0.5, g=0.0)")
            print("  C: Forward anisotropic (κ=0.1, σs=0.9, g=0.6)")
    else:
        # Default: run both cases with reduced photon count for quick test
        print("Quick test mode: Running both cases with 100,000 photons each...")
        print("For production (5M photons), run: python monte_carlo_3d_rte_benchmark.py B")
        print()
        
        # Temporarily reduce photon count for testing
        original_n_photons = mc.N_PHOTONS
        mc.N_PHOTONS = 100_000
        mc.BATCH_SIZE = 10_000
        
        # Run Case B
        print("\n" + "="*70)
        print("CASE 3D_B: Isotropic Scattering")
        print("="*70)
        mc.run_monte_carlo('B')
        
        # Run Case C
        print("\n" + "="*70)
        print("CASE 3D_C: Forward Anisotropic Scattering")
        print("="*70)
        mc.run_monte_carlo('C')
        
        # Restore original values
        mc.N_PHOTONS = original_n_photons
        
        print("\n" + "="*70)
        print("Quick test complete!")
        print("Production runs: Edit N_PHOTONS in monte_carlo_3d_rte_benchmark.py")
        print("="*70)
