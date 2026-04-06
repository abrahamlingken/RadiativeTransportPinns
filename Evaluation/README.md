# 3D RTE PINN vs Monte Carlo 验证

本目录包含用于对比PINN预测结果与Monte Carlo基准解的代码。

## 文件说明

- `compare_pinn_mc.py` - 主对比脚本：加载PINN模型、计算G场、与MC结果对比

## 使用流程

### 1. 生成Monte Carlo验证数据

在云端服务器上运行MC求解器：

```bash
cd Solvers/MC

# Case B: 各向同性散射 (κ=0.5, σs=4.5, g=0.0)
python monte_carlo_3d_case_bc.py --case B --photons 10000000

# Case C: 前向散射 (κ=0.5, σs=4.5, g=0.8)
python monte_carlo_3d_case_bc.py --case C --photons 10000000
```

输出保存在 `Solvers/MC/MC3D_Results/` 目录。

### 2. 训练PINN模型

```bash
cd Training

# Case B: 推荐用较大网络
python train_3d_multicase.py --case 3D_B \
    --layers 10 --neurons 256 --activation swish \
    --adam-iters 2000 --lbfgs-max-iter 50000

# Case C: 前向散射，可能需要更多迭代
python train_3d_multicase.py --case 3D_C \
    --layers 10 --neurons 256 --activation swish \
    --adam-iters 2000 --lbfgs-max-iter 100000
```

输出保存在 `Results_3D_CaseB/` 或 `Results_3D_CaseC/` 目录。

### 3. 对比验证

```bash
cd Evaluation

# Case B对比
python compare_pinn_mc.py --case 3D_B

# Case C对比  
python compare_pinn_mc.py --case 3D_C

# 指定MC路径（如自动检测失败）
python compare_pinn_mc.py --case 3D_B \
    --mc-path ../Solvers/MC/MC3D_Results/MC_G_3D_B_CaseB_Isotropic.npz
```

## 输出文件

对比脚本会在 `Results_3D_CaseX/` 目录生成：

- `G_field_3D_X.npz` - PINN和MC的G场数据
- `error_report_3D_X.json` - 误差指标
- `comparison_3D_X.png` - 切片对比图（X/Y/Z中间切片）
- `centerline_3D_X.png` - 中心线对比图

## 误差指标说明

| 指标 | 说明 |
|------|------|
| L1_absolute | 平均绝对误差 |
| L2_absolute | 均方根误差 |
| Linf_absolute | 最大绝对误差 |
| L1_relative | 平均相对误差 |
| L2_relative | 均方根相对误差 |
| Linf_relative | 最大相对误差 |

## 预期结果参考

基于MC模拟（1000万光子）：

| Case | G_center | G_face_center |
|------|----------|---------------|
| 3D_B (isotropic) | ~2.15 | ~0.02 |
| 3D_C (forward) | ~1.67 | ~0.03 |

Case C的G_center比Case B低，因为前向散射(g=0.8)使光子更倾向于沿原方向传播，减少了横向扩散。
