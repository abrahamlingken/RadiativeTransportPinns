# RMC for PINN Case B/C 验证指南

## 代码文件
- `Solvers/MC/rmc_pinn_case_bc.py` - 基于RMC3D逻辑的PINN验证代码

## 物理逻辑（来自RMC3D）

### 1. 射线追踪流程
```
源点采样 → 方向采样 → 追踪直到吸收/逃逸
    ↓
计算到边界距离 (IntersectBoundary)
    ↓
采样光学厚度决定碰撞点
    ↓
判断吸收/散射 (根据albedo)
    ↓
散射时更新方向 (HG相函数)
    ↓
记录碰撞位置 → RDF统计
```

### 2. 关键参数
| 参数 | Case B | Case C | 说明 |
|------|--------|--------|------|
| κ | 0.5 | 0.5 | 吸收系数 |
| σs | 4.5 | 4.5 | 散射系数 |
| β | 5.0 | 5.0 | 总衰减系数 |
| g | 0.0 | 0.8 | HG散射不对称因子 |
| Albedo | 0.9 | 0.9 | 散射反照率 = σs/β |

### 3. 与RMC3D的主要区别

| 特性 | RMC3D (传热) | RMC_PINN (Case B/C) |
|------|-------------|---------------------|
| 源项 | 黑体辐射 σT⁴ | S(r) = max(0,1-5r) |
| 源分布 | 整个计算域 | 中心球体 r<0.2 |
| 衰减系数 | 空间变化 ke(r) | 常数 β=5.0 |
| 结果 | 辐射传热通量 | 积分辐射强度 G |
| 边界 | 温度边界 | 真空边界（逃逸） |

## 使用方法

### 云端运行
```bash
cd Solvers/MC

# Case B (isotropic)
python rmc_pinn_case_bc.py --case B --nray 5000000

# Case C (forward scattering)
python rmc_pinn_case_bc.py --case C --nray 5000000
```

### 输出文件
- `MC3D_Results/RMC_G_3D_B_CaseB.npz` - Case B结果
- `MC3D_Results/RMC_G_3D_C_CaseC.npz` - Case C结果
- `MC3D_Results/RMC_G_3D_*.png` - 可视化图

### 结果对比
```python
import numpy as np

# 加载RMC结果
rmc_data = np.load('MC3D_Results/RMC_G_3D_B_CaseB.npz')
G_rmc = rmc_data['G']

# 加载PINN结果（假设已生成）
# pinn_data = np.load('Results_3D_CaseB/G_field_3D_B.npz')
# G_pinn = pinn_data['G_pinn']

# 对比
print(f"RMC G_center: {G_rmc[25,25,25]:.4f}")
# print(f"PINN G_center: {G_pinn[25,25,25]:.4f}")
```

## 归一化说明

RMC输出的是**辐射密度函数 (RDF)**，需要转换为**积分辐射强度 G**：

```
G = RDF × (Q_total / V_cell) × N_cells

其中：
- Q_total = π/150 ≈ 0.0209 (源积分)
- V_cell = dx³ (网格体积)
- N_cells = 51³ (总网格数)
```

**注意**：归一化因子可能需要根据PINN的定义进行校准。

## 预期结果

基于物理分析：

| Case | G_center | G_face | 说明 |
|------|----------|--------|------|
| B (g=0.0) | ~2.0-2.5 | ~0.01-0.05 | 各向同性散射 |
| C (g=0.8) | ~1.5-2.0 | ~0.02-0.06 | 前向散射，穿透更深 |

Case C的G_center应该比Case B低（前向散射减少横向扩散）。

## 与PINN对比步骤

1. **运行RMC**获取基准解
2. **运行PINN训练**（如果尚未完成）
3. **使用`Evaluation/compare_pinn_mc.py`对比**

```bash
cd Evaluation
python compare_pinn_mc.py --case 3D_B --mc-path ../Solvers/MC/MC3D_Results/RMC_G_3D_B_CaseB.npz
```

## 常见问题

### Q: RMC结果与PINN差距大？
**可能原因：**
- 归一化因子不匹配
- PINN训练未收敛
- 源项定义差异（S(r) vs κ·S(r)）

### Q: 如何提高RMC精度？
**增加射线数：**
```bash
python rmc_pinn_case_bc.py --case B --nray 10000000
```

### Q: 如何验证RMC正确性？
**先运行Case A（纯吸收）验证：**
- 使用之前提供的`monte_carlo_3d_case_a_v2.py`
- 预期G_center ≈ 0.92
