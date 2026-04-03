#!/usr/bin/env powershell
# Project Organization Script - English Version
# Usage: Run in PowerShell: .\organize_project_en.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Project Organization Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$ProjectRoot = Get-Location
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Yellow

# Step 1: Create directory structure
Write-Host "`n[Step 1] Creating directory structure..." -ForegroundColor Green

$NewDirs = @(
    "Training",
    "Solvers\DOM",
    "Solvers\MC", 
    "Solvers\RMC",
    "Evaluation",
    "Tests",
    "Docs",
    "Figures\1D",
    "Figures\3D",
    "Figures\Validation",
    "Figures\RMC3D"
)

foreach ($dir in $NewDirs) {
    $fullPath = Join-Path $ProjectRoot $dir
    if (!(Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    }
}

# Step 2: Move training scripts
Write-Host "`n[Step 2] Moving training scripts to Training/..." -ForegroundColor Green

$TrainingFiles = @{
    "train_3d_multicase.py" = "Training\train_3d_multicase.py"
    "Scripts\train_1d_multicase.py" = "Training\train_1d_multicase.py"
    "Scripts\train_1d_multicase_anisotropic.py" = "Training\train_1d_multicase_anisotropic.py"
}

foreach ($src in $TrainingFiles.Keys) {
    $dest = $TrainingFiles[$src]
    if (Test-Path $src) {
        Move-Item $src $dest -Force
        Write-Host "  Moved: $src -> $dest" -ForegroundColor Gray
    }
}

# Step 3: Move solvers
Write-Host "`n[Step 3] Moving solvers to Solvers/..." -ForegroundColor Green

$SolverFiles = @{
    "dom_1d_solver.py" = "Solvers\DOM\dom_1d_solver.py"
    "dom_1d_solver_HG.py" = "Solvers\DOM\dom_1d_solver_HG.py"
    "monte_carlo_3d_rte_benchmark_final.py" = "Solvers\MC\monte_carlo_3d_rte_benchmark_final.py"
    "rmc3d_case_abc_v2.py" = "Solvers\RMC\rmc3d_case_abc_v2.py"
}

foreach ($src in $SolverFiles.Keys) {
    $dest = $SolverFiles[$src]
    if (Test-Path $src) {
        Move-Item $src $dest -Force
        Write-Host "  Moved: $src -> $dest" -ForegroundColor Gray
    }
}

# Step 4: Move evaluation scripts
Write-Host "`n[Step 4] Moving evaluation scripts to Evaluation/..." -ForegroundColor Green

$EvalFiles = @(
    "evaluate_pinn_vs_dom.py",
    "evaluate_pinn_vs_exact.py",
    "evaluate_anisotropic.py",
    "validate_3d_pure_absorption.py",
    "validate_pure_absorption.py",
    "plot_3d_paper_figures.py"
)

foreach ($file in $EvalFiles) {
    if (Test-Path $file) {
        Move-Item $file "Evaluation\$file" -Force
        Write-Host "  Moved: $file" -ForegroundColor Gray
    }
}

# Step 5: Move test files
Write-Host "`n[Step 5] Moving test files to Tests/..." -ForegroundColor Green

$TestFiles = @(
    "test_pure_absorption.py",
    "check_paths.py"
)

foreach ($file in $TestFiles) {
    if (Test-Path $file) {
        Move-Item $file "Tests\$file" -Force
        Write-Host "  Moved: $file" -ForegroundColor Gray
    }
}

# Step 6: Move documents
Write-Host "`n[Step 6] Moving documents to Docs/..." -ForegroundColor Green

# Move txt files (paper pages)
Get-ChildItem -Filter "page_*.txt" | ForEach-Object {
    Move-Item $_.Name "Docs\$($_.Name)" -Force
    Write-Host "  Moved: $($_.Name)" -ForegroundColor Gray
}

# Move pdf file
if (Test-Path "paper.pdf") {
    Move-Item "paper.pdf" "Docs\paper.pdf" -Force
    Write-Host "  Moved: paper.pdf" -ForegroundColor Gray
}

# Move md files (keep README.md in root)
Get-ChildItem -Filter "*.md" | ForEach-Object {
    if ($_.Name -ne "README.md" -and $_.Name -ne "LICENSE") {
        Move-Item $_.Name "Docs\$($_.Name)" -Force
        Write-Host "  Moved: $($_.Name)" -ForegroundColor Gray
    }
}

# Step 7: Fix paths in moved files
Write-Host "`n[Step 7] Fixing path references in moved files..." -ForegroundColor Green

# Fix Training/train_3d_multicase.py
$train3dPath = "Training\train_3d_multicase.py"
if (Test-Path $train3dPath) {
    $content = Get-Content $train3dPath -Raw
    $content = $content -replace "PROJECT_ROOT = os.path.dirname\(os.path.abspath\(__file__\)\)", 
        "PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"
    Set-Content $train3dPath $content -NoNewline
    Write-Host "  Fixed path: $train3dPath" -ForegroundColor Gray
}

# Fix Evaluation scripts
$evalScripts = Get-ChildItem "Evaluation\*.py" -ErrorAction SilentlyContinue
foreach ($script in $evalScripts) {
    $content = Get-Content $script.FullName -Raw
    if ($content -match "os.path.dirname\(__file__\)" -and !($content -match "dirname.*dirname")) {
        $content = $content -replace 
            "os.path.dirname\(os.path.abspath\(__file__\)\)",
            "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"
        Set-Content $script.FullName $content -NoNewline
        Write-Host "  Fixed path: $($script.Name)" -ForegroundColor Gray
    }
}

# Step 8: Remove old files
Write-Host "`n[Step 8] Removing old version files..." -ForegroundColor Green

$FilesToRemove = @(
    "dom_1d_solver_v2.py",
    "dom_1d_solver_v3.py",
    "monte_carlo_3d_rte_benchmark.py",
    "monte_carlo_3d_rte_benchmark_v2.py",
    "monte_carlo_3d_rte_benchmark_v3.py",
    "rmc3d_case_abc.py",
    "rmc3d_solver.py",
    "test_rmc_import.py",
    "test_rmc_simple.py",
    "test_rmc_result.png",
    "chunked_loss_snippet.py",
    "training_summary_3d.json",
    "dom_boundaries.png",
    "dom_solution.png",
    "3D_Pure_Absorption_1D_Line.pdf",
    "3D_Pure_Absorption_2D_Contours.pdf",
    "3D_Pure_Absorption_Line.pdf",
    "run_monte_carlo_benchmark.py",
    "Scripts\train_1d.py",
    "Scripts\train_3d.py",
    "PINNS2.py",
    "PINNS2_3D.py"
)

foreach ($file in $FilesToRemove) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  Removed: $file" -ForegroundColor Red
    }
}

# Step 9: Organize image files
Write-Host "`n[Step 9] Organizing image files..." -ForegroundColor Green

$domPngFiles = @(
    "dom_solution_g0.5.png",
    "dom_solution_g-0.5.png", 
    "dom_solution_boundaries_g0.5.png",
    "dom_solution_boundaries_g-0.5.png",
    "dom_solution_boundaries_g0.8.png"
)

foreach ($file in $domPngFiles) {
    if (Test-Path $file) {
        Move-Item $file "Figures\1D\$file" -Force -ErrorAction SilentlyContinue
        Write-Host "  Moved: $file -> Figures\1D\" -ForegroundColor Gray
    }
}

# Step 10: Clean empty directories
Write-Host "`n[Step 10] Cleaning empty directories..." -ForegroundColor Green

if (Test-Path "Scripts") {
    $hasFiles = Get-ChildItem "Scripts" -Recurse -File | Where-Object { $_.Extension -eq ".py" }
    if (-not $hasFiles) {
        Remove-Item "Scripts" -Recurse -Force
        Write-Host "  Removed empty: Scripts" -ForegroundColor Red
    }
}

# Step 11: Create directory README
Write-Host "`n[Step 11] Creating directory structure documentation..." -ForegroundColor Green

$readmeContent = @"
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
```bash
# 3D Training
python Training/train_3d_multicase.py --case A

# DOM Solver
python Solvers/DOM/dom_1d_solver_HG.py

# RMC Solver
python Solvers/RMC/rmc3d_case_abc_v2.py

# Evaluation
python Evaluation/evaluate_pinn_vs_dom.py
```
"@

Set-Content "DIRECTORY_STRUCTURE.md" $readmeContent
Write-Host "  Created: DIRECTORY_STRUCTURE.md" -ForegroundColor Gray

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Organization Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nVerify with these commands:" -ForegroundColor Yellow
Write-Host "  python Training/train_3d_multicase.py --help" -ForegroundColor White
Write-Host "  python Solvers/DOM/dom_1d_solver_HG.py" -ForegroundColor White
Write-Host "  python Solvers/RMC/rmc3d_case_abc_v2.py" -ForegroundColor White
