# 评估脚本使用指南

## 各向异性散射评估 (Cases D/E/F)

### 前提条件

确保以下文件已存在：

1. **DOM 结果文件** (运行 `dom_1d_solver_HG.py` 生成):
   - `dom_solution_g0.5.npy` (Case D: 前向散射)
   - `dom_solution_g-0.5.npy` (Case E: 后向散射)
   - `dom_solution_g0.8.npy` (Case F: 强前向散射)

2. **PINN 模型** (运行 `train_1d_multicase_anisotropic.py` 生成):
   - `Results_1D_CaseD/TrainedModel/model.pkl`
   - `Results_1D_CaseE/TrainedModel/model.pkl`
   - `Results_1D_CaseF/TrainedModel/model.pkl`

### 运行评估

```bash
python evaluate_anisotropic.py
```

### 输出文件

运行后将生成以下文件在 `Evaluation_Results_Anisotropic/` 目录：

| 文件 | 说明 |
|------|------|
| `Anisotropic_CaseD_Eval.png` | Case D 1x3 组合评估图 |
| `Anisotropic_CaseE_Eval.png` | Case E 1x3 组合评估图 |
| `Anisotropic_CaseF_Eval.png` | Case F 1x3 组合评估图 |
| `evaluation_anisotropic_summary.txt` | 误差统计汇总 |

### 图表说明

每个案例的 1x3 组合图包含：

1. **左图 (a)**: PINN vs DOM 等高线对比
   - 上半部分: DOM 解 $I_{\mathrm{dom}}(x, \mu)$
   - 下半部分: PINN 解 $I_{\mathrm{pinn}}(x, \mu)$
   - 使用统一的 jet colormap 和色标

2. **中图 (b)**: 绝对误差场
   - $|I_{\mathrm{pinn}} - I_{\mathrm{dom}}|$ 的等高线图
   - 使用 coolwarm colormap
   - 显示全局相对 L2 误差

3. **右图 (c)**: 宏观入射辐射 $G(x)$ 对比
   - DOM: 黑色实线 (Ground Truth)
   - PINN: 红色散点
   - $G(x) = \int_{-1}^{1} I(x, \mu) \, \mathrm{d}\mu$

### 误差指标

脚本计算的定量指标：

- **Relative L2 Error**: $\|u_{\mathrm{pinn}} - u_{\mathrm{dom}}\|_2 / \|u_{\mathrm{dom}}\|_2$
- **Absolute L2 Error**: $\|u_{\mathrm{pinn}} - u_{\mathrm{dom}}\|_2$
- **Maximum Error**: $\max |u_{\mathrm{pinn}} - u_{\mathrm{dom}}|$
- **G(x) Relative Error**: $\|G_{\mathrm{pinn}} - G_{\mathrm{dom}}\|_2 / \|G_{\mathrm{dom}}\|_2$

## 各向同性散射评估 (Cases A/B/C)

对于各向同性散射案例，使用：

```bash
python evaluate_pinn_vs_dom.py
```

输出保存在 `Evaluation_Results/` 目录。
