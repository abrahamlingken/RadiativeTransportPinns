# Monte Carlo 3D RTE Benchmark Solver

高性能蒙特卡洛光子追踪求解器，用于生成3D辐射传输方程的基准参考数据。

## 物理问题

**控制方程**: 3D辐射传输方程 (RTE)
$$
\mathbf{s} \cdot \nabla I + (\kappa + \sigma_s) I = \kappa I_b + \frac{\sigma_s}{4\pi} \int_{4\pi} I' \Phi(\mathbf{s}' \to \mathbf{s}) d\Omega'
$$

**源项**: $I_b = \max(0, 1.0 - 2.0 \cdot d)$, 其中 $d$ 是从中心点 $(0.5, 0.5, 0.5)$ 的距离

**边界条件**: 所有6个面都是冷黑边界（零入射辐射）

**相函数**: Henyey-Greenstein 各向异性散射

## 目标案例

### Case 3D_B: 各向同性散射
- $\kappa = 0.5$ (吸收系数)
- $\sigma_s = 0.5$ (散射系数)
- $g = 0.0$ (HG各向同性)

### Case 3D_C: 前向各向异性散射
- $\kappa = 0.1$ (吸收系数)
- $\sigma_s = 0.9$ (散射系数)
- $g = 0.6$ (HG前向散射)

## 文件说明

| 文件 | 说明 |
|------|------|
| `monte_carlo_3d_rte_benchmark.py` | 主求解器（Numba加速） |
| `run_monte_carlo_benchmark.py` | 快速启动器 |
| `test_mc_quick.py` | 功能测试脚本 |
| `Benchmark_Data/` | 输出目录（自动创建） |

## 使用方法

### 快速测试（10万光子）
```bash
python run_monte_carlo_benchmark.py
```

### 运行特定案例（500万光子）
```bash
# Case B: 各向同性散射
python monte_carlo_3d_rte_benchmark.py B

# Case C: 前向各向异性散射
python monte_carlo_3d_rte_benchmark.py C
```

### 自定义参数

编辑 `monte_carlo_3d_rte_benchmark.py` 顶部的配置部分:

```python
# Grid resolution
NX, NY, NZ = 40, 40, 40  # 可增加到 50

# Monte Carlo parameters  
N_PHOTONS = 5_000_000    # 光子数量（越多精度越高）
BATCH_SIZE = 100_000     # 批处理大小
```

## 输出文件

运行后会在 `Benchmark_Data/` 目录生成:

| 文件 | 内容 |
|------|------|
| `Benchmark_G_CaseB.npy` | $G(x,y,z)$ - 入射辐射3D场 |
| `Benchmark_I_Line_CaseB.npy` | $I(x, 0.5, 0.5, \mathbf{s}_{view})$ - 中心线方向强度 |
| `Benchmark_x_Line_CaseB.npy` | $x$ 坐标数组 |
| `Benchmark_Params_CaseB.npy` | 模拟参数字典 |
| `Benchmark_CaseB.png` | 可视化结果 |

## 性能优化

1. **Numba JIT编译**: 所有核心函数使用 `@njit` 装饰器
2. **并行计算**: `process_photon_batch()` 使用 `prange` 并行化
3. **批处理**: 分批次处理光子以显示进度

**典型性能**（AMD Ryzen 7950X 16核）:
- 100万光子: ~2-3分钟
- 500万光子: ~10-15分钟

## 验证PINN结果

使用生成的基准数据验证PINN预测:

```python
import numpy as np

# Load benchmark data
G_mc = np.load('Benchmark_Data/Benchmark_G_CaseB.npy')
x_line = np.load('Benchmark_Data/Benchmark_x_Line_CaseB.npy')
I_mc = np.load('Benchmark_Data/Benchmark_I_Line_CaseB.npy')

# Compare with PINN predictions
# ... your PINN evaluation code ...
```

## 算法说明

### 1. 光子发射
- 从体积源随机采样发射点
- 使用拒绝采样，概率正比于 $I_b(x,y,z)$
- 初始方向各向同性采样

### 2. 光子追踪
- 采样自由程: $l = -\ln(\xi)/\beta$, 其中 $\xi \in [0,1]$
- 边界检测: 计算到最近边界距离
- 若逃逸则终止，否则在交互点沉积能量

### 3. 交互处理
- 吸收概率: $\kappa/\beta$
- 散射概率: $\sigma_s/\beta$
- HG相函数采样新方向

### 4. 结果归一化
- $G$ 归一化: $G = \frac{\text{总沉积能量}}{\text{网格体积}}$
- $I$ 归一化: 沿视线方向积分
