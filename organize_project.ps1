#!/usr/bin/env powershell
# organize_project.ps1 - 项目整理自动化脚本
# 使用方法: 在PowerShell中运行: .\organize_project.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  项目整理脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 获取项目根目录
$ProjectRoot = Get-Location
Write-Host "项目根目录: $ProjectRoot" -ForegroundColor Yellow

# 步骤1: 创建新目录结构
Write-Host "`n[步骤1] 创建新目录结构..." -ForegroundColor Green

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
        Write-Host "  创建: $dir" -ForegroundColor Gray
    }
}

# 步骤2: 移动训练脚本
Write-Host "`n[步骤2] 移动训练脚本到 Training/..." -ForegroundColor Green

$TrainingFiles = @{
    "train_3d_multicase.py" = "Training\train_3d_multicase.py"
    "Scripts\train_1d_multicase.py" = "Training\train_1d_multicase.py"
    "Scripts\train_1d_multicase_anisotropic.py" = "Training\train_1d_multicase_anisotropic.py"
}

foreach ($src in $TrainingFiles.Keys) {
    $dest = $TrainingFiles[$src]
    if (Test-Path $src) {
        Move-Item $src $dest -Force
        Write-Host "  移动: $src -> $dest" -ForegroundColor Gray
    }
}

# 步骤3: 移动求解器
Write-Host "`n[步骤3] 移动求解器到 Solvers/..." -ForegroundColor Green

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
        Write-Host "  移动: $src -> $dest" -ForegroundColor Gray
    }
}

# 步骤4: 移动评估脚本
Write-Host "`n[步骤4] 移动评估脚本到 Evaluation/..." -ForegroundColor Green

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
        Write-Host "  移动: $file" -ForegroundColor Gray
    }
}

# 步骤5: 移动测试文件
Write-Host "`n[步骤5] 移动测试文件到 Tests/..." -ForegroundColor Green

$TestFiles = @(
    "test_pure_absorption.py",
    "check_paths.py"
)

foreach ($file in $TestFiles) {
    if (Test-Path $file) {
        Move-Item $file "Tests\$file" -Force
        Write-Host "  移动: $file" -ForegroundColor Gray
    }
}

# 步骤6: 移动文档
Write-Host "`n[步骤6] 移动文档到 Docs/..." -ForegroundColor Green

# 移动txt文件（论文页面）
Get-ChildItem -Filter "page_*.txt" | ForEach-Object {
    Move-Item $_.Name "Docs\$($_.Name)" -Force
    Write-Host "  移动: $($_.Name)" -ForegroundColor Gray
}

# 移动pdf文件
if (Test-Path "paper.pdf") {
    Move-Item "paper.pdf" "Docs\paper.pdf" -Force
    Write-Host "  移动: paper.pdf" -ForegroundColor Gray
}

# 移动md文件（保留README.md在根目录）
Get-ChildItem -Filter "*.md" | ForEach-Object {
    if ($_.Name -ne "README.md" -and $_.Name -ne "LICENSE") {
        Move-Item $_.Name "Docs\$($_.Name)" -Force
        Write-Host "  移动: $($_.Name)" -ForegroundColor Gray
    }
}

# 步骤7: 修复文件中的路径引用
Write-Host "`n[步骤7] 修复文件中的路径引用..." -ForegroundColor Green

# 修复Training/train_3d_multicase.py
$train3dPath = "Training\train_3d_multicase.py"
if (Test-Path $train3dPath) {
    $content = Get-Content $train3dPath -Raw
    # 修复PROJECT_ROOT路径（因为现在在一级子目录）
    $content = $content -replace "PROJECT_ROOT = os.path.dirname\(os.path.abspath\(__file__\)\)", 
        "PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"
    Set-Content $train3dPath $content -NoNewline
    Write-Host "  修复路径: $train3dPath" -ForegroundColor Gray
}

# 修复Evaluation中的脚本
$evalScripts = Get-ChildItem "Evaluation\*.py"
foreach ($script in $evalScripts) {
    $content = Get-Content $script.FullName -Raw
    
    # 检查是否需要添加路径修复
    if ($content -match "sys.path.insert.*__file__") {
        # 已经有路径插入，可能需要修改
        if ($content -match "os.path.dirname\(__file__\)" -and !($content -match "dirname.*dirname")) {
            # 单层dirname，需要改为双层
            $content = $content -replace 
                "os.path.dirname\(os.path.abspath\(__file__\)\)",
                "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"
            Set-Content $script.FullName $content -NoNewline
            Write-Host "  修复路径: $($script.Name)" -ForegroundColor Gray
        }
    }
}

# 步骤8: 删除旧版本文件
Write-Host "`n[步骤8] 删除旧版本和临时文件..." -ForegroundColor Green

$FilesToRemove = @(
    # DOM求解器旧版本
    "dom_1d_solver_v2.py",
    "dom_1d_solver_v3.py",
    # MC求解器旧版本
    "monte_carlo_3d_rte_benchmark.py",
    "monte_carlo_3d_rte_benchmark_v2.py",
    "monte_carlo_3d_rte_benchmark_v3.py",
    # RMC求解器旧版本
    "rmc3d_case_abc.py",
    "rmc3d_solver.py",
    # 测试临时文件
    "test_rmc_import.py",
    "test_rmc_simple.py",
    "test_rmc_result.png",
    "chunked_loss_snippet.py",
    # 临时输出文件
    "training_summary_3d.json",
    # 旧图片文件
    "dom_boundaries.png",
    "dom_solution.png",
    # 3D验证旧图
    "3D_Pure_Absorption_1D_Line.pdf",
    "3D_Pure_Absorption_2D_Contours.pdf",
    "3D_Pure_Absorption_Line.pdf",
    # 批运行脚本（可选）
    "run_monte_carlo_benchmark.py",
    # 旧训练脚本
    "Scripts\train_1d.py",
    "Scripts\train_3d.py",
    "PINNS2.py",
    "PINNS2_3D.py"
)

foreach ($file in $FilesToRemove) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  删除: $file" -ForegroundColor Red
    }
}

# 步骤9: 移动结果图片到Figures目录
Write-Host "`n[步骤9] 整理图片文件..." -ForegroundColor Green

# 移动DOM结果图
$domPngFiles = @("dom_solution_g0.5.png", "dom_solution_g-0.5.png", "dom_solution_boundaries_g0.5.png", "dom_solution_boundaries_g-0.5.png", "dom_solution_boundaries_g0.8.png")
foreach ($file in $domPngFiles) {
    if (Test-Path $file) {
        Move-Item $file "Figures\1D\$file" -Force
        Write-Host "  移动: $file -> Figures\1D\" -ForegroundColor Gray
    }
}

# 移动npy文件到Benchmark_Data
$npyFiles = @("dom_solution_g0.5.npy", "dom_solution_g-0.5.npy", "dom_solution_g0.8.npy")
foreach ($file in $npyFiles) {
    if (Test-Path $file) {
        Move-Item $file "Benchmark_Data\$file" -Force -ErrorAction SilentlyContinue
        Write-Host "  移动: $file -> Benchmark_Data\" -ForegroundColor Gray
    }
}

# 步骤10: 清理空目录
Write-Host "`n[步骤10] 清理空目录..." -ForegroundColor Green

if (Test-Path "Scripts") {
    $scriptsContent = Get-ChildItem "Scripts" -Recurse | Where-Object { !$_.PSIsContainer -or (Get-ChildItem $_.FullName -Recurse | Where-Object { !$_.PSIsContainer }).Count -gt 0 }
    if ($scriptsContent.Count -eq 0) {
        Remove-Item "Scripts" -Recurse -Force
        Write-Host "  删除空目录: Scripts" -ForegroundColor Red
    }
}

# 步骤11: 创建README文件说明新结构
Write-Host "`n[步骤11] 创建目录说明文件..." -ForegroundColor Green

$readmeContent = @"
# 项目目录结构说明

## 核心目录
- **Core/**: 核心模块（网络模型、数据集等）
- **EquationModels/**: 物理方程模型（1D/3D RTE）

## 功能模块
- **Training/**: 训练脚本
  - train_3d_multicase.py: 3D多案例训练
  - train_1d_multicase.py: 1D各向同性案例
  - train_1d_multicase_anisotropic.py: 1D各向异性案例

- **Solvers/**: 数值求解器
  - DOM/: 离散坐标法求解器
  - MC/: 正向蒙特卡洛求解器
  - RMC/: 反向蒙特卡洛求解器

- **Evaluation/**: 评估与验证脚本
  - evaluate_*.py: 各种评估脚本
  - validate_*.py: 验证脚本
  - plot_*.py: 绘图脚本

- **Tests/**: 测试脚本

- **Docs/**: 文档和论文资料

## 结果目录
- **Results/**: 训练结果
- **Figures/**: 可视化输出
- **Benchmark_Data/**: 基准数据（DOM/MC结果）

## 使用示例

```bash
# 3D训练
python Training/train_3d_multicase.py --case A

# DOM求解
python Solvers/DOM/dom_1d_solver_HG.py

# RMC求解
python Solvers/RMC/rmc3d_case_abc_v2.py

# 评估
python Evaluation/evaluate_pinn_vs_dom.py
```
"@

Set-Content "DIRECTORY_STRUCTURE.md" $readmeContent
Write-Host "  创建: DIRECTORY_STRUCTURE.md" -ForegroundColor Gray

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  整理完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`n请验证以下命令是否正常工作：" -ForegroundColor Yellow
Write-Host "  python Training/train_3d_multicase.py --help" -ForegroundColor White
Write-Host "  python Solvers/DOM/dom_1d_solver_HG.py" -ForegroundColor White
Write-Host "  python Solvers/RMC/rmc3d_case_abc_v2.py" -ForegroundColor White
