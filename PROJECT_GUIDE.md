# RadiativeTransportPinns 项目完全指南

## 项目概述

本项目使用 **物理信息神经网络 (Physics-Informed Neural Networks, PINN)** 求解辐射传输方程 (Radiative Transfer Equation, RTE)。项目涵盖从1D平板几何到3D立方体几何的多种介质配置，包括纯吸收、各向同性散射和各向异性散射案例。

**核心特点：**
- 纯Python/PyTorch实现
- 支持高梯度、强局部化热源问题的自适应训练
- 与DOM、Monte Carlo等经典数值方法对比验证
- 期刊级可视化输出

---

## 目录结构总览

```
RadiativeTransportPinns/
├── Core/                          # 核心模块
├── EquationModels/                # 物理方程定义
├── Training/                      # 训练脚本
├── Solvers/                       # 经典数值求解器
├── Evaluation/                    # 评估与验证
├── Tests/                         # 测试脚本
├── Docs/                          # 文档资料
├── Results_*/                     # 训练结果
├── Figures_*/                     # 可视化输出
└── Benchmark_Data/                # 基准数据
```

---

## 第一部分：核心模块 (Core/)

### 1.1 核心文件说明

| 文件 | 功能 | 关键类/函数 |
|:---|:---|:---|
| `ImportFile.py` | 统一导入模块 | 集中导入PyTorch、NumPy等所有依赖 |
| `ModelClassTorch2.py` | PINN网络定义 | `Pinns`类 - 全连接神经网络 |
| `DatasetTorch2.py` | 数据集管理 | `DefineDataset`类 - 配点采样与数据加载 |
| `ObjectClass.py` | 优化器封装 | 训练循环和优化器管理 |

### 1.2 网络架构 (ModelClassTorch2.py)

```python
class Pinns(nn.Module):
    # 全连接神经网络，支持：
    # - 可变深度 (hidden_layers)
    # - 可变宽度 (neurons)
    # - 多种激活函数 (tanh, swish, etc.)
    # - Xavier初始化
```

**关键特性：**
- 输入维度自适应（1D: 2维 (x,μ)，3D: 5维 (x,y,z,θ,φ)）
- 残差连接支持
- 可配置的正则化参数

---

## 第二部分：物理方程模块 (EquationModels/)

### 2.1 1D辐射传输方程 (RadTrans1D.py)

**控制方程：**
$$\mu \frac{\partial I}{\partial x} + (\kappa + \sigma_s) I = \frac{\sigma_s}{2} \int_{-1}^{1} \Phi(\mu, \mu') I(\mu') d\mu'$$

**关键函数：**

| 函数 | 功能 |
|:---|:---|
| `compute_res()` | 计算PDE残差 |
| `compute_scattering()` | 计算散射积分（HG相函数）|
| `apply_BC()` | 应用Marshak边界条件 |
| `add_collocation()` | 生成内部配点 |
| `add_boundary()` | 生成边界点 |

**全局参数（可通过训练脚本动态修改）：**
```python
KAPPA_CONST = 0.5      # 吸收系数 κ
SIGMA_S_CONST = 0.0    # 散射系数 σs
I_B_CONST = 0.0        # 黑体辐射强度
G_HG = 0.0             # HG不对称因子 g
```

### 2.2 3D辐射传输方程 (RadTrans3D_Complex.py)

**控制方程：**
$$\mathbf{s} \cdot \nabla I + (\kappa + \sigma_s) I = \frac{\sigma_s}{4\pi} \int_{4\pi} \Phi(\mathbf{s}, \mathbf{s}') I d\Omega' + I_b$$

**关键类：**
```python
class RadTrans3D_Physics:
    # 封装所有物理参数和计算方法
    - __init__(): 初始化κ, σs, g, 求积配置
    - compute_res(): 计算PDE残差
    - compute_scattering_3d(): 3D散射积分
    - compute_incident_radiation(): 计算G(x)
    - generate_collocation_points(): 生成配点（含重要性采样）
```

**重要性采样策略（50/50分割）：**
- 50% 均匀分布（全局覆盖）
- 50% 中心聚焦（[0.3,0.7]³区域加密）

### 2.3 其他物理模型

| 文件 | 用途 |
|:---|:---|
| `RadTrans3D_paper.py` | 论文复现版本（各向同性，固定源项）|
| `RadiativeInverseBEST.py` | 反问题求解 |
| `RadiativeFreqRan2.py` | 频域随机介质 |

---

## 第三部分：训练脚本 (Training/)

### 3.1 1D案例训练

#### 3.1.1 各向同性散射 (train_1d_multicase.py)

**案例配置：**

| 案例 | κ | σs | g | τ | ω | 物理意义 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **A** | 0.5 | 0.5 | 0.0 | 1.0 | 0.5 | 基准案例 |
| **B** | 2.0 | 2.0 | 0.0 | 4.0 | 0.5 | 光学厚介质 |
| **C** | 0.1 | 0.9 | 0.0 | 1.0 | 0.9 | 强散射介质 |

**使用方法：**
```bash
# 训练所有案例
python Training/train_1d_multicase.py

# 仅训练特定案例
python Training/train_1d_multicase.py --case A
python Training/train_1d_multicase.py --case B
python Training/train_1d_multicase.py --case C
```

**训练参数（自适应配置）：**
```python
Case A: n_coll=16384, n_u=2048    # 默认配置
Case B: n_coll=20480, n_u=2560    # 光学厚，增加配点
Case C: n_coll=16384, n_u=3072    # 强散射，增加边界点
```

**网络配置：**
- 隐藏层：8层
- 神经元：32个
- 激活函数：tanh
- 优化器：L-BFGS

#### 3.1.2 各向异性散射 (train_1d_multicase_anisotropic.py)

**案例配置：**

| 案例 | κ | σs | g | 散射类型 |
|:---:|:---:|:---:|:---:|:---|
| **D** | 0.5 | 0.5 | **+0.5** | 前向散射 |
| **E** | 0.5 | 0.5 | **-0.5** | 后向散射 |
| **F** | 0.5 | 0.5 | **+0.8** | 强前向散射 |

**使用方法：**
```bash
python Training/train_1d_multicase_anisotropic.py
```

### 3.2 3D案例训练 (train_3d_multicase.py)

**案例配置：**

| 案例 | κ | σs | g | β | 物理意义 |
|:---:|:---:|:---:|:---:|:---:|:---|
| **3D_A** | 5.0 | 0.0 | 0.0 | 5.0 | 纯吸收（光学厚）|
| **3D_B** | 0.5 | 4.5 | 0.0 | 5.0 | 各向同性散射 |
| **3D_C** | 0.5 | 4.5 | 0.8 | 5.0 | 前向散射 |

**关键改进（高κ自适应）：**
```python
if config['kappa'] >= 5.0:
    NETWORK_PROPERTIES = {
        "hidden_layers": 10,    # 更深
        "neurons": 256,         # 更宽
        "activation": "swish"   # Swish激活
    }
```

**两阶段训练策略：**
1. **Adam预热**（2000迭代）：λ_residual=0.1，优先学习边界
2. **L-BFGS精调**：λ_residual=1.0，强化PDE约束

**使用方法：**
```bash
python Training/train_3d_multicase.py --case 3D_A
python Training/train_3d_multicase.py --case 3D_B
python Training/train_3d_multicase.py --case 3D_C
python Training/train_3d_multicase.py --case all
```

---

## 第四部分：经典数值求解器 (Solvers/)

### 4.1 离散坐标法 (DOM/)

#### dom_1d_solver_HG.py
**功能：** 1D RTE DOM求解器，支持Henyey-Greenstein各向异性散射

**输出：**
- `dom_solution_g{g}.npy` - 辐射强度矩阵 I[mu, x]
- `dom_solution_g{g}.png` - 等高线图
- `dom_solution_boundaries_g{g}.png` - 边界对比图

**使用方法：**
```bash
python Solvers/DOM/dom_1d_solver_HG.py
```

**案例：** g = 0.5, -0.5, 0.8（与PINN Case D/E/F对应）

### 4.2 正向蒙特卡洛 (MC/)

#### monte_carlo_3d_rte_benchmark_final.py
**功能：** 3D RTE正向蒙特卡洛求解器

**特点：**
- 从热源发射光子
- HG相函数散射
- 统计能量沉积

**案例：** 与PINN 3D_A/B/C对应

### 4.3 反向蒙特卡洛 (RMC/)

#### rmc3d_case_abc_v2.py
**功能：** 3D RTE反向蒙特卡洛求解器

**求解策略：**
- Case A：解析解（Beer-Lambert积分）
- Case B/C：正向蒙特卡洛（从源发射）

**输出：**
- `RMC3D_Results/RMC_Case{A/B/C}_results.npz`
- 中心线对比图
- 2D截面热图

---

## 第五部分：评估与验证 (Evaluation/)

### 5.1 1D案例评估

#### evaluate_pinn_vs_dom.py
**功能：** PINN与DOM基准结果的全面比较

**评估案例：** A, B, C（各向同性）

**输出：**
- `{CaseA/B/C}_comparison.png` - 3×2组合图（精确解、PINN、误差、G(x)对比）
- `{CaseA/B/C}_G_comparison.png` - G(x)单独对比图
- `evaluation_summary.txt` - 误差统计

#### evaluate_anisotropic.py
**功能：** 各向异性散射案例评估（Case D/E/F）

**输出：**
- `Anisotropic_Case{D/E/F}_comparison.png` - 2×2组合图
- `Anisotropic_Case{D/E/F}_G_comparison.png` - G(x)对比图

#### evaluate_pinn_vs_exact.py
**功能：** 纯吸收案例与解析解对比

**参考案例：** Results_1D_Section32（κ=0.5, σs=0.0）

### 5.2 3D案例验证

#### validate_3d_pure_absorption.py
**功能：** 3D纯吸收案例的高精度验证

**方法：**
- 精确解：Beer-Lambert光线追踪积分
- 角度积分：Gauss-Legendre求积（32×64方向）
- 沿光线积分：300点高斯求积

**输出：**
- `Figures_3D_Validation/G_CaseA_Centerline_HighPrecision.png` - 中心线对比
- `Figures_3D_Validation/G_CaseA_2D_HighPrecision.png` - 2D截面热图
- `.npz`数据文件（含G_exact, G_pinn, error）

#### plot_3d_paper_figures.py
**功能：** 生成期刊级质量的3D可视化图表

**输出：**
- `Figures_3D/`目录下的高质量PNG/PDF/VTS文件
- 支持VTK格式用于ParaView后处理

---

## 第六部分：结果文件夹说明

### 6.1 1D结果 (Results_1D_*/)

**标准结构：**
```
Results_1D_Case{A~F}/
├── Images/                    # 可视化结果
│   ├── net_sol.png           # PINN解 I(x,μ)
│   ├── solution.png          # 与参考解对比
│   ├── u0.png               # 左边界角度分布
│   ├── u1.png               # 右边界角度分布
│   └── errors.txt           # 误差统计
├── TrainedModel/
│   ├── model.pkl            # 训练好的模型
│   └── Information.txt      # 模型信息
├── InfoModel.txt            # 训练参数和误差
├── training_history.json    # 损失历史
└── training_curves*.png     # 训练曲线
```

**案例对比：**

| 文件夹 | κ | σs | g | 类型 | 参考基准 |
|:---|:---:|:---:|:---:|:---:|:---|
| Results_1D_CaseA | 0.5 | 0.5 | 0.0 | 各向同性 | DOM |
| Results_1D_CaseB | 2.0 | 2.0 | 0.0 | 光学厚 | DOM |
| Results_1D_CaseC | 0.1 | 0.9 | 0.0 | 强散射 | DOM |
| Results_1D_CaseD | 0.5 | 0.5 | +0.5 | 前向散射 | DOM |
| Results_1D_CaseE | 0.5 | 0.5 | -0.5 | 后向散射 | DOM |
| Results_1D_CaseF | 0.5 | 0.5 | +0.8 | 强前向 | DOM |
| Results_1D_Section32 | 0.5 | 0.0 | 0.0 | 纯吸收 | 解析解 |

### 6.2 3D结果 (Results_3D_*/)

**标准结构：**
```
Results_3D_Case{A/B/C}/
├── model.pkl                # 训练好的模型
├── case_config.json        # 案例配置（κ, σs, g）
├── training_history.json   # 损失历史
├── training_info.json      # 训练信息（时间、迭代数等）
└── training_curves.png     # 训练曲线
```

**关键文件：**
- `case_config.json` - 包含完整的物理参数记录
- `model.pkl` - 可直接加载用于预测和评估

### 6.3 评估结果 (Evaluation_Results*/)

| 文件夹 | 内容 |
|:---|:---|
| `Evaluation_Results/` | Case A/B/C评估图 |
| `Evaluation_Results_Anisotropic/` | Case D/E/F评估图 |
| `Evaluation_Results_PureAbsorption/` | 纯吸收案例评估（旧）|

### 6.4 基准数据 (Benchmark_Data/)

**内容：**
- `dom_solution_g{0.5,-0.5,0.8}.npy` - DOM参考解
- 用于PINN与DOM的定量对比

---

## 第七部分：辅助工具与脚本

### 7.1 测试脚本 (Tests/)

| 文件 | 功能 |
|:---|:---|
| `test_pure_absorption.py` | 纯吸收案例单元测试 |
| `check_paths.py` | 检查项目路径和数据文件完整性 |

### 7.2 绘图脚本 (Scripts/)

| 文件 | 功能 |
|:---|:---|
| `plot_training.py` | 绘制期刊级训练曲线 |
| `analyze_vts.py` | 分析VTK输出文件 |

### 7.3 项目整理脚本

| 文件 | 功能 |
|:---|:---|
| `backup_before_organize.ps1` | 整理前备份 |
| `organize_project_en.ps1` | 自动整理项目结构 |
| `restore_from_backup.ps1` | 从备份恢复 |

---

## 第八部分：快速开始指南

### 8.1 环境配置

```bash
# 创建conda环境
conda env create -f environment.yml
conda activate radiative-transport-pinns

# 或pip安装
pip install -r requirements.txt
```

### 8.2 运行流程

**1D案例完整流程：**
```bash
# 1. 训练
python Training/train_1d_multicase.py --case A

# 2. DOM基准（用于对比）
python Solvers/DOM/dom_1d_solver_HG.py

# 3. 评估
python Evaluation/evaluate_pinn_vs_dom.py
```

**3D案例完整流程：**
```bash
# 1. 训练
python Training/train_3d_multicase.py --case 3D_A

# 2. 验证（与精确解对比）
python Evaluation/validate_3d_pure_absorption.py

# 3. 生成期刊级图表
python Evaluation/plot_3d_paper_figures.py
```

### 8.3 关键参数调整

**在训练脚本中修改：**
```python
# 物理参数（在CASE_CONFIGS中）
'kappa': 0.5,      # 吸收系数
'sigma_s': 0.5,    # 散射系数
g': 0.0,           # HG不对称因子

# 网络参数（在initialize_inputs_for_case中）
n_coll_ = 16384    # 内部配点数
n_u_ = 2048        # 边界点数
hidden_layers = 8  # 隐藏层数
neurons = 32       # 神经元数
```

---

## 第九部分：常见问题 (FAQ)

### Q1: Case A（Results_1D_CaseA）与 Results_1D_Section32 有何区别？
**A:** 
- Case A: κ=0.5, σs=0.5（有散射，无解析解）
- Section32: κ=0.5, σs=0.0（纯吸收，有解析解）
- 两者不可互换！

### Q2: 如何训练纯吸收案例？
**A:** 设置 `sigma_s = 0.0`，使用 `evaluate_pinn_vs_exact.py` 与解析解对比。

### Q3: 3D案例训练时间多久？
**A:** 
- Case A (κ=5.0): ~90分钟（7400迭代）
- Case B/C: ~60-120分钟（取决于收敛）

### Q4: 如何判断训练收敛？
**A:** 
- 损失值稳定（< 1e-5）
- 梯度范数 < 1e-7
- 验证误差不再下降

---

## 附录：文件命名约定

| 前缀 | 含义 | 示例 |
|:---|:---|:---|
| `train_*.py` | 训练脚本 | `train_3d_multicase.py` |
| `evaluate_*.py` | 评估脚本 | `evaluate_pinn_vs_dom.py` |
| `validate_*.py` | 验证脚本 | `validate_3d_pure_absorption.py` |
| `dom_*.py` | DOM求解器 | `dom_1d_solver_HG.py` |
| `plot_*.py` | 绘图脚本 | `plot_3d_paper_figures.py` |
| `Results_*` | 训练结果 | `Results_3D_CaseA/` |
| `Figures_*` | 可视化输出 | `Figures_3D_Validation/` |

---

**最后更新：** 2024年
**项目维护者：** abrahamlingken
**许可证：** MIT License
