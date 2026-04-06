# FMC 3D Solver - User Guide

## Overview

This Forward Monte Carlo (FMC) solver calculates the scalar incident radiation field $G(x,y,z)$ for validating PINN solutions to the 3D Radiative Transfer Equation.

## Key Features

- **Track-Length Estimator**: Accumulates distance traveled in each cell
- **Survival Biasing (Implicit Capture)**: Weight reduction instead of termination
- **Russian Roulette**: Variance reduction for low-weight photons
- **Henyey-Greenstein Phase Function**: Accurate anisotropic scattering
- **Numba JIT Acceleration**: C-level performance from Python

## Test Cases

### Case B (Isotropic Scattering)
- κ = 0.5, σs = 4.5, β = 5.0
- g = 0.0 (isotropic)
- 10,000,000 photons

### Case C (Anisotropic Scattering)
- κ = 0.1, σs = 4.9, β = 5.0
- g = 0.8 (strong forward scattering)
- 10,000,000 photons

## Usage

### Run Both Cases
```bash
cd Solvers/MC
python FMC_3D_Solver.py --case all
```

### Run Single Case
```bash
# Case B only
python FMC_3D_Solver.py --case B

# Case C only
python FMC_3D_Solver.py --case C
```

### Custom Output Directory
```bash
python FMC_3D_Solver.py --case all --output-dir my_results
```

## Output Files

| File | Description |
|------|-------------|
| `FMC_G_3D_CaseB.npy` | 3D G field array for Case B (50×50×50) |
| `FMC_G_3D_CaseC.npy` | 3D G field array for Case C (50×50×50) |
| `FMC_G_3D_CaseB_metadata.npz` | Parameters and statistics |
| `FMC_G_3D_CaseB.png` | Visualization plots |

## Loading Results in Python

```python
import numpy as np

# Load G field
G_B = np.load('MC3D_Results/FMC_G_3D_CaseB.npy')

# Load metadata
meta_B = np.load('MC3D_Results/FMC_G_3D_CaseB_metadata.npz')
print(f"G_center = {meta_B['G_center']:.4f}")
print(f"κ = {meta_B['kappa']}, σs = {meta_B['sigma_s']}, g = {meta_B['g']}")

# Access specific point
center_idx = 25  # 50/2
G_center = G_B[center_idx, center_idx, center_idx]
```

## Algorithm Details

### Track-Length Estimator
For each photon traveling through the domain:
```
G_cell += (weight × track_length) / (cell_volume × beta)
```

### Survival Biasing
At each collision:
```
weight *= albedo  # weight reduction instead of termination
```

### Russian Roulette
When weight < 10⁻⁴:
- 10% chance: weight ×= 10, continue
- 90% chance: terminate photon

### Source Sampling
Photons emitted from spherical source (r < 0.2) with distribution:
```
S(r) = max(0, 1 - 5r)
```

## Performance

Typical performance on modern CPU:
- ~500,000-1,000,000 photons/second
- 10 million photons: ~10-20 seconds

## Requirements

- Python 3.8+
- NumPy
- Numba
- Matplotlib (optional, for plotting)

## Comparing with PINN

```python
import numpy as np

# Load FMC result
G_fmc = np.load('MC3D_Results/FMC_G_3D_CaseB.npy')

# Load PINN result (adjust path as needed)
# G_pinn = np.load('../../Results_3D_CaseB/G_field.npy')

# Calculate error
diff = np.abs(G_fmc - G_pinn)
rel_error = diff / (np.abs(G_fmc) + 1e-10)

print(f"Mean relative error: {rel_error.mean():.2%}")
print(f"Max relative error: {rel_error.max():.2%}")
```

## Troubleshooting

**Numba warnings about TBB**: Install TBB or ignore (parallel still works)

**Low G values**: Check normalization. The solver uses physical normalization based on source integral.

**Memory issues**: Reduce batch size in `run_monte_carlo()` if needed.
