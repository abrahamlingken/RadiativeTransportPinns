# Physics-Informed Neural Networks for Radiative Transfer Equation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于物理信息神经网络（PINN）求解辐射传输方程（RTE）的高精度数值模拟框架。本项目实现了从1D平板几何到3D立方体几何的多尺度、多物理场辐射传输问题求解，支持纯吸收、各向同性/各向异性散射等复杂介质配置。

**核心创新**：针对高光学厚度（τ=5.0）和强局部化热源问题，提出自适应网络架构与重要性采样策略，显著提升了PINN在陡峭梯度区域的预测精度。

---

## 📋 目录

- [项目亮点](#项目亮点)
- [物理模型与案例](#物理模型与案例)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [结果与验证](#结果与验证)
- [引用](#引用)

---

## ✨ 项目亮点

### 🔬 物理模型创新
- **1D RTE**：支持纯吸收、各向同性/各向异性散射（HG相函数）
- **3D RTE**：完整的三维辐射传输方程，支持方向依赖性散射
- **高梯度处理**：针对 κ=5.0 光学厚介质，提出10层×256神经元Swish网络架构
- **重要性采样**：50/50分层采样策略，热源区域配点密度提升64倍

### 🧠 算法优化
- **两阶段训练**：Adam预热（边界聚焦）+ L-BFGS精调（PDE约束）
- **梯度累积Chunking**：支持16K+配点，避免OOM
- **权重退火策略**：动态调整残差权重，平衡边界与PDE约束
- **源项解耦**：数学形式与物理形式分离，确保散射案例能量守恒

### 📊 验证体系
- **解析解验证**：纯吸收案例与Beer-Lambert定律对比
- **DOM基准对比**：与离散坐标法（Discrete Ordinates Method）结果对比
- **蒙特卡洛验证**：与正向/反向蒙特卡洛统计结果对比
- **误差分析**：相对L2误差 < 5%，最大误差 < 10%

---

## 🎯 物理模型与案例

### 1D 辐射传输方程

控制方程：
$$\mu \frac{\partial I}{\partial x} + (\kappa + \sigma_s) I = \frac{\sigma_s}{2} \int_{-1}^{1} \Phi(\mu, \mu') I(\mu') d\mu'$$

**案例配置**（各向同性散射，g=0）：

| 案例 | κ | σs | τ | ω | 物理意义 | 配点策略 |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **A** | 0.5 | 0.5 | 1.0 | 0.5 | 基准案例 | 16,384 coll + 2,048 boundary |
| **B** | 2.0 | 2.0 | 4.0 | 0.5 | 光学厚介质 | 20,480 coll + 2,560 boundary |
| **C** | 0.1 | 0.9 | 1.0 | 0.9 | 强散射介质 | 16,384 coll + 3,072 boundary |

**案例配置**（各向异性散射，HG相函数）：

| 案例 | κ | σs | g | 散射类型 | 物理特征 |
|:---:|:---:|:---:|:---:|:---|:---|
| **D** | 0.5 | 0.5 | +0.5 | 前向散射 | 光子主要沿原方向传播 |
| **E** | 0.5 | 0.5 | -0.5 | 后向散射 | 光子主要反向散射 |
| **F** | 0.5 | 0.5 | +0.8 | 强前向散射 | 几乎直线传播，小角度偏转 |

### 3D 辐射传输方程

控制方程：
$$\mathbf{s} \cdot \nabla I + (\kappa + \sigma_s) I = \frac{\sigma_s}{4\pi} \int_{4\pi} \Phi(\mathbf{s}, \mathbf{s}') I d\Omega' + I_b$$

其中源项为高度局部化热源：
$$I_b(\mathbf{x}) = \max(0, 1 - 5r), \quad r = |\mathbf{x} - \mathbf{x}_c|, \quad \mathbf{x}_c = (0.5, 0.5, 0.5)$$

**案例配置**（固定光学厚度 β=5.0）：

| 案例 | κ | σs | g | 网络架构 | 物理意义 |
|:---:|:---:|:---:|:---:|:---|:---|
| **3D_A** | 5.0 | 0.0 | 0.0 | 10层×256, Swish | 纯吸收，有解析解（验证基准）|
| **3D_B** | 0.5 | 4.5 | 0.0 | 8层×128, tanh | 各向同性散射 |
| **3D_C** | 0.5 | 4.5 | 0.8 | 8层×128, tanh | 强前向散射 |

---

## 📁 项目结构

```
RadiativeTransportPinns/
├── Core/                          # 核心模块
│   ├── ModelClassTorch2.py       # PINN网络定义（支持可变深度/宽度）
│   ├── DatasetTorch2.py          # 数据集管理与采样（Sobol/均匀/重要性）
│   ├── ImportFile.py             # 统一依赖导入
│   └── ObjectClass.py            # 优化器封装
│
├── EquationModels/                # 物理方程模型
│   ├── RadTrans1D.py             # 1D RTE（支持HG各向异性散射）
│   ├── RadTrans3D_Complex.py     # 3D RTE（含重要性采样）
│   └── RadTrans3D_paper.py       # 论文复现版本
│
├── Training/                      # 训练脚本
│   ├── train_1d_multicase.py     # 1D各向同性案例（A/B/C）
│   ├── train_1d_multicase_anisotropic.py  # 1D各向异性案例（D/E/F）
│   └── train_3d_multicase.py     # 3D案例（A/B/C），含高κ自适应
│
├── Solvers/                       # 经典数值求解器（验证基准）
│   ├── DOM/
│   │   └── dom_1d_solver_HG.py   # 离散坐标法（1D，HG散射）
│   ├── MC/
│   │   └── monte_carlo_3d_rte_benchmark_final.py  # 正向蒙特卡洛
│   └── RMC/
│       └── rmc3d_case_abc_v2.py  # 反向蒙特卡洛（Case A解析解）
│
├── Evaluation/                    # 评估与验证
│   ├── evaluate_pinn_vs_dom.py   # PINN vs DOM对比
│   ├── evaluate_anisotropic.py   # 各向异性案例评估
│   ├── validate_3d_pure_absorption.py  # 3D纯吸收解析解验证
│   └── plot_3d_paper_figures.py  # 期刊级可视化
│
├── Tests/                         # 测试脚本
├── Docs/                          # 文档与论文资料
├── Results_1D_{CaseA~F}/          # 1D训练结果
├── Results_3D_{CaseA~C}/          # 3D训练结果
├── Figures_3D/                    # 3D可视化输出
├── PROJECT_GUIDE.md               # 完整项目指南
└── README.md                      # 本文件
```

详细说明请参考 [PROJECT_GUIDE.md](PROJECT_GUIDE.md)。

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/abrahamlingken/RadiativeTransportPinns.git
cd RadiativeTransportPinns

# 创建conda环境
conda env create -f environment.yml
conda activate radiative-transport-pinns

# 验证安装
python -c "import torch; print(torch.__version__)"
```

### 1D案例训练

```bash
# 训练所有1D案例（A/B/C）
python Training/train_1d_multicase.py

# 仅训练特定案例
python Training/train_1d_multicase.py --case A

# 训练各向异性案例（D/E/F）
python Training/train_1d_multicase_anisotropic.py
```

### 3D案例训练

```bash
# 训练3D Case A（纯吸收，有解析解）
python Training/train_3d_multicase.py --case 3D_A

# 训练所有3D案例
python Training/train_3d_multicase.py --case all
```

### 验证与评估

```bash
# DOM基准求解（用于1D案例对比）
python Solvers/DOM/dom_1d_solver_HG.py

# PINN与DOM对比评估
python Evaluation/evaluate_pinn_vs_dom.py

# 3D纯吸收案例解析解验证
python Evaluation/validate_3d_pure_absorption.py
```

---

## 📖 使用方法

### 修改物理参数

在训练脚本的 `CASE_CONFIGS` 中修改：

```python
CASE_CONFIGS = {
    'A': {
        'kappa': 0.5,      # 吸收系数
        'sigma_s': 0.5,    # 散射系数
        'g': 0.0,          # HG不对称因子
        'folder': 'Results_1D_CaseA'
    },
    # ...
}
```

### 调整网络架构

对于高光学厚度问题（κ ≥ 5.0），自动启用增强网络：

```python
if config['kappa'] >= 5.0:
    NETWORK_PROPERTIES = {
        "hidden_layers": 10,    # 深层网络
        "neurons": 256,         # 宽层
        "activation": "swish"   # Swish激活
    }
```

### 自定义配点策略

在 `RadTrans3D_Complex.py` 中调整重要性采样：

```python
def generate_collocation_points(self, n_collocation):
    n_uniform = n_collocation // 2   # 50% 均匀分布
    n_center = n_collocation - n_uniform  # 50% 中心聚焦 [0.3,0.7]^3
```

---

## 📊 结果与验证

### 精度对比

| 案例 | PINN误差 (相对L2) | DOM/MC误差 | 备注 |
|:---:|:---:|:---:|:---|
| 1D Case A | ~2% | < 1% | 与DOM对比 |
| 1D Case B | ~3% | < 1% | 光学厚，收敛稍慢 |
| 1D Case C | ~3% | < 1% | 强散射，需要更多边界点 |
| 3D Case A | ~5% | 0% | 与解析解对比（中心G=0.92）|
| 3D Case B | ~5% | < 2% | 与MC对比 |

### 关键发现

1. **纯吸收案例（3D_A）**：PINN预测中心G值0.92，与Beer-Lambert解析解吻合
2. **散射增强效应**：Case B > Case A，Case C > Case A（符合物理预期）
3. **高κ自适应**：10层Swish网络在κ=5.0时显著优于8层tanh

### 输出文件

每个案例生成：
- `model.pkl` - 训练好的神经网络模型
- `training_history.json` - 损失历史
- `training_curves.png` - 训练曲线
- `Images/` - 预测结果可视化

---

## 📝 引用

如果您使用本项目的代码或方法，请引用：

```bibtex
@article{radiative2020,
  title={Physics-Informed Neural Networks for Simulating Radiative Transfer},
  author={[Your Name]},
  journal={arXiv preprint arXiv:2009.13291},
  year={2024}
}
```

---

## 📧 联系方式

- **项目维护者**: abrahamlingken
- **项目主页**: https://github.com/abrahamlingken/RadiativeTransportPinns

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可。

---

**致谢**: 本项目基于论文 "Physics-Informed Neural Networks for Radiative Transfer" (arXiv:2009.13291) 的方法框架，并针对高梯度问题进行了算法改进和扩展。
