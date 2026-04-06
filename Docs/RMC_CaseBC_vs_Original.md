# RMC PINN Case B/C 与原RMC3D的对比

## 主要区别

### 1. 物性参数

| 参数 | 原RMC3D (传热) | RMC PINN Case B/C |
|------|---------------|-------------------|
| 吸收系数 κ | 空间变化函数 | **常数 0.5** |
| 衰减系数 β | 空间变化函数 | **常数 5.0** |
| 散射反照率 ω | 空间变化函数 | **常数 0.9** |
| HG因子 g | 固定类型 (1/2/3) | **0.0 或 0.8** |

### 原RMC3D代码（空间变化）
```matlab
% 衰减系数-散射反照率-温度分布（空间变化）
ke = -10*((xyzcenter(:,1)-0.5).^2+...)+10;  % β从中心到边缘变化
albedo = -0.8*((xyzcenter(:,1)-0.5).^2+...)+0.8;  % albedo空间变化
Temp = -800*((xyzcenter(:,1)-0.5).^2+...)+2000;  % 温度场
```

### RMC PINN Case B/C（常数参数）
```python
# Case B: 各向同性散射
kappa = 0.5      # 常数
sigma_s = 4.5    # 常数  
beta = 5.0       # 常数
g_hg = 0.0       # HG因子
albedo = 0.9     # 常数

# Case C: 前向散射
g_hg = 0.8       # 仅HG因子不同，其他相同
```

## 2. 源项

| 特性 | 原RMC3D | RMC PINN |
|------|---------|----------|
| 源类型 | 黑体辐射 σT⁴ | 给定源项 S(r) |
| 空间分布 | 整个计算域 | 中心球体 r<0.2 |
| 分布函数 | 温度依赖 | max(0, 1-5r) |

### RMC PINN源采样
```python
def sample_source_point():
    """在源球内按S(r)分布采样"""
    while True:
        x, y, z = np.random.random(3)
        r = sqrt((x-0.5)² + (y-0.5)² + (z-0.5)²)
        if r < 0.2:  # 只在源球内
            if random() < (1 - 5*r):  # 按S(r)接受
                return x, y, z
```

## 3. 追踪逻辑差异

### 原RMC3D（空间变化ke）
```matlab
% 需要沿路径累积光学厚度（因为ke变化）
OptThi = ke(node).*dis;  % 每个网格的ke不同
```

### RMC PINN（常数ke）
```python
# 简化为常数光学厚度计算
dist_collision = rand_opt_thick / beta  # beta是常数
```

## 4. 边界条件

| 条件 | 原RMC3D | RMC PINN |
|------|---------|----------|
| 边界类型 | 温度/辐射边界 | **真空边界** |
| 射线出界 | 统计边界热流 | **逃逸终止** |
| 反射 | 可能有 | **无** |

## 5. 结果物理量

| 输出 | 原RMC3D | RMC PINN |
|------|---------|----------|
| RDF | 辐射密度函数 | 碰撞密度 |
| 结果 | 热流/温度场 | **积分辐射强度 G** |
| 单位 | W/m² | 无量纲或W/m² |

## 修正后的RMC代码特点

### 关键修改点
1. **移除了空间变化的ke和albedo**
2. **使用常数beta = 5.0**
3. **使用常数albedo = 0.9**
4. **添加了源球限制 (r<0.2)**
5. **使用S(r)分布进行源采样**

### 验证方法
```bash
# 运行修正后的代码
cd Solvers/MC
python rmc_pinn_case_bc.py --case B --nray 5000000

# 预期输出
Physics: κ=0.5, σs=4.5, β=5.0, g=0.0
Albedo (σs/β): 0.9000
NOTE: ke and albedo are CONSTANT (uniform)
```

## 如果要用原RMC3D逻辑（空间变化参数）

需要修改代码，将ke和albedo改为数组：
```python
# 空间变化的ke和albedo（类似原RMC3D）
ke = -10 * ((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2) + 10
albedo = -0.8 * ((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2) + 0.8
```

但这将**不再是PINN Case B/C**的验证，而是另一个不同的问题。

## 总结

| 需求 | 应使用的代码 |
|------|-------------|
| 验证PINN Case B/C | `rmc_pinn_case_bc.py`（常数参数）✅ |
| 复制原RMC3D结果 | 需要额外修改支持空间变化参数 |
| 对比MATLAB代码 | 注意参数定义差异 |

**重要**：请确保运行RMC代码时看到 `NOTE: ke and albedo are CONSTANT (uniform)` 消息，以确认使用的是常数参数版本。
