# Directory Structure

## Core Directories
- Core/: Core modules (network models, datasets)
- EquationModels/: Physical equation models (1D/3D RTE)

## Functional Modules
- Training/: Training scripts
- Solvers/: Numerical solvers (DOM, MC, RMC)
- Evaluation/: Evaluation and validation scripts
- Tests/: Test scripts
- Docs/: Documentation and paper materials

## Usage Examples
`ash
# 3D Training
python Training/train_3d_multicase.py --case A

# DOM Solver
python Solvers/DOM/dom_1d_solver_HG.py

# RMC Solver
python Solvers/RMC/rmc3d_case_abc_v2.py

# Evaluation
python Evaluation/evaluate_pinn_vs_dom.py
`
