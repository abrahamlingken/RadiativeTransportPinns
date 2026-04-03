# 介质配置指南

## 快速配置

修改 `EquationModels/RadTrans1D.py` 文件顶部的参数：

```python
# 配置区域：修改以下参数切换介质类型
KAPPA_CONST = 0.5      # 吸收系数
SIGMA_S_CONST = 0.0    # 散射系数（设为0即为纯吸收）
I_B_CONST = 0.0        # 黑体辐射强度
G_HG = 0.0             # HG不对称因子（0为各向同性）
```

## 三种介质模式

### 1. 纯吸收介质
```python
KAPPA_CONST = 0.5
SIGMA_S_CONST = 0.0    # 关键：设为0
I_B_CONST = 0.0
```
方程简化为：
```
mu * du/dx + kappa * u = 0
```

### 2. 吸收-散射介质（各向同性）
```python
KAPPA_CONST = 0.5
SIGMA_S_CONST = 0.5    # 非零散射
I_B_CONST = 0.0
G_HG = 0.0             # 各向同性
```

### 3. 各向异性散射介质
```python
KAPPA_CONST = 0.5
SIGMA_S_CONST = 0.5
I_B_CONST = 0.0
G_HG = 0.5             # 前向散射（0<g<1）
# G_HG = -0.5          # 后向散射（-1<g<0）
```

## 验证配置

运行测试：
```bash
cd Core
python -c "import sys; sys.path.insert(0, '..'); from EquationModels.RadTrans1D import *; print(f'Kappa={KAPPA_CONST}, Sigma_s={SIGMA_S_CONST}, I_b={I_B_CONST}')"
```

## 当前状态

项目已配置为：**纯吸收介质** (SIGMA_S=0, I_B=0)
