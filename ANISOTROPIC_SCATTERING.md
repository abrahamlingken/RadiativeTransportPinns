# 各向异性散射支持 (Anisotropic Scattering Support)

本文档描述辐射传输PINN项目中各向异性散射的实现。

## 概述

项目现在支持使用 **Henyey-Greenstein (HG)** 相函数的各向异性散射。HG相函数是一个单参数模型，用于描述散射光的角度分布。

## Henyey-Greenstein 相函数

### 数学定义

$$\Phi(\mu, \mu', g) = \frac{1 - g^2}{(1 + g^2 - 2g\mu\mu')^{3/2}}$$

其中：
- $\mu, \mu'$: 入射和散射方向余弦
- $g$: 不对称因子 (asymmetry factor)，取值范围 $[-1, 1]$

### 物理意义

| g 值 | 散射类型 | 说明 |
|------|----------|------|
| $g = 0$ | 各向同性 | 均匀向所有方向散射 |
| $g > 0$ | 前向散射 | 峰值在 $\mu = \mu'$ 方向 |
| $g < 0$ | 后向散射 | 峰值在 $\mu = -\mu'$ 方向 |
| $g \to 1$ | 强前向 | 近乎直线传播 |
| $g \to -1$ | 强后向 | 几乎完全反向 |

## 案例配置

### Cases D/E/F - 各向异性散射案例

在 `Scripts/train_1d_multicase_anisotropic.py` 中定义：

```python
CASE_CONFIGS = {
    'D': {
        'name': 'CaseD_Forward',
        'description': 'Forward scattering (g=0.5)',
        'kappa': 0.5,
        'sigma_s': 0.5,
        'g': 0.5,        # 前向散射
    },
    'E': {
        'name': 'CaseE_Backward',
        'description': 'Backward scattering (g=-0.5)',
        'kappa': 0.5,
        'sigma_s': 0.5,
        'g': -0.5,       # 后向散射
    },
    'F': {
        'name': 'CaseF_StrongForward',
        'description': 'Strong forward scattering (g=0.8)',
        'kappa': 0.5,
        'sigma_s': 0.5,
        'g': 0.8,        # 强前向散射
    }
}
```

## 关键代码结构

### 1. 全局参数 (EquationModels/RadTrans1D.py)

```python
G_HG = 0.0  # HG不对称因子，可由外部脚本动态修改
```

### 2. HG相函数

```python
def kernel_HG(mu, mu_prime, g=G_HG):
    """Henyey-Greenstein相函数"""
    cos_theta = mu * mu_prime
    g_sq = g ** 2
    numerator = 1.0 - g_sq
    denominator = torch.pow(1.0 + g_sq - 2.0 * g * cos_theta, 1.5)
    return numerator / denominator
```

### 3. 散射积分计算

```python
def compute_scattering(u, x, mu, network, g=G_HG):
    """计算各向异性散射积分项"""
    # ... 数值积分使用 Gauss-Legendre 求积 ...
    phi_matrix = kernel_HG(mu_current, mu_prime, g=g)
    # ... 积分计算 ...
    return scattering_term
```

### 4. 动态参数修改

```python
def set_anisotropic_params(kappa_val, sigma_s_val, g_val):
    """动态设置各向异性散射参数"""
    import EquationModels.RadTrans1D as RadTrans
    
    RadTrans.KAPPA_CONST = kappa_val
    RadTrans.SIGMA_S_CONST = sigma_s_val
    RadTrans.G_HG = g_val  # 修改全局变量
```

## 使用方法

### 运行单个案例

```bash
# Case D: 前向散射 (g=0.5)
python Scripts/train_1d_multicase_anisotropic.py --case D

# Case E: 后向散射 (g=-0.5)
python Scripts/train_1d_multicase_anisotropic.py --case E

# Case F: 强前向散射 (g=0.8)
python Scripts/train_1d_multicase_anisotropic.py --case F
```

### 运行所有案例

```bash
python Scripts/train_1d_multicase_anisotropic.py
```

## 验证

运行测试脚本验证各向异性散射机制：

```bash
python Core/test_anisotropic.py
```

测试内容：
1. 动态修改 `G_HG` 参数
2. HG相函数计算正确性
3. 模块 reload 机制

## 注意事项

1. **动态修改**: `G_HG` 是模块级全局变量，训练脚本通过直接修改该变量来切换散射类型

2. **数值积分**: 使用 32 点 Gauss-Legendre 求积计算散射积分，确保精度

3. **可视化**: 训练完成后，可以使用 `evaluate_pinn_vs_dom.py` 对比 PINN 与 DOM 结果

## 参考文献

1. Henyey, L. G., & Greenstein, J. L. (1941). Diffuse radiation in the Galaxy. *The Astrophysical Journal*, 93, 70-83.

2. Modest, M. F. (2013). *Radiative Heat Transfer* (3rd ed.). Academic Press.
