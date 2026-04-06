# G定义差异分析

## 问题的核心

你的质疑完全正确！当前RMC代码和PINN代码中G的计算方式有本质差异。

---

## 1. PINN中的G定义

在PINN代码（`EquationModels/RadTrans3D_Complex.py`）中：

```python
# G = ∫ I dΩ （积分辐射强度）
# 通过对离散方向求和得到
G = sum(I * w_i)  # w_i是角度积分权重
```

**关键特点：**
- G是**辐射强度在所有方向上的积分**
- 单位：W/(m²) 或无量纲（取决于归一化）
- 与源项S(r)直接相关

---

## 2. 当前RMC代码的问题

在我的RMC代码中：

```python
# 追踪射线，记录碰撞次数
nodes[n_coll] = node  # 只记录碰撞位置，权重=1

# 统计
RDFs[node_ext] += 1.0  # 每次碰撞计数+1
```

**问题：**
- 每次碰撞的**权重都是1**
- 没有考虑：
  1. **源强度** S(r₀) 的差异
  2. **Beer-Lambert衰减** exp(-βs)
  3. **散射次数**对权重的累积影响
  4. **角度覆盖**（4π因子）

---

## 3. 正确的碰撞估计器

根据RTE理论，碰撞估计器应该计算：

```
G(r) = (1/V_cell) × Σ (weight_i / κ)

其中 weight_i = S(r₀) × exp(-β×s) × (其他因子)
```

### 修正后的代码应该是：

```python
@njit(cache=True, fastmath=True)
def track_ray_with_weight(beta, kappa, sigma_s, g_hg, albedo_const, ...):
    # 采样起点和初始强度
    x0, y0, z0 = sample_source_point()
    S0 = source_term(x0, y0, z0)  # 初始源强度
    weight = S0  # 初始权重
    
    distance_traveled = 0.0
    
    while n_coll < max_collisions:
        # ... 计算碰撞点 ...
        
        if dist_collision < dist_boundary:
            # 到达碰撞点
            distance_traveled += dist_collision
            
            # Beer-Lambert衰减
            attenuation = np.exp(-beta * distance_traveled)
            
            # 碰撞权重 = 初始强度 × 衰减 / kappa
            collision_weight = weight * attenuation / kappa
            
            # 记录带权重的碰撞
            nodes[n_coll] = node
            weights[n_coll] = collision_weight
            n_coll += 1
            
            # 散射继续，但权重保持不变（散射不损失能量）
            if np.random.random() < albedo_const:
                # 散射，更新方向
                ux, uy, uz = sample_hg_direction(ux, uy, uz, g_hg)
                x0, y0, z0 = x_coll, y_coll, z_coll
                # 注意：weight不变，继续追踪
                continue
            else:
                # 吸收，终止
                break
        else:
            # 逃逸，终止
            break
    
    return nodes[:n_coll], weights[:n_coll], ...
```

---

## 4. 为什么之前需要"经验校准"

因为当前代码：
- 每次碰撞权重=1（忽略了源强度和衰减）
- 导致RDF只是一个"碰撞计数密度"
- 与物理上的G相差一个复杂的归一化因子

**经验校准的问题：**
- 没有物理基础
- 可能随网格、光子数变化
- 不可靠

---

## 5. RMC3D中的G定义

在原RMC3D代码中：
```matlab
RDF = RDFs/nray;  % 辐射密度函数（碰撞概率密度）
RI = sigma*RDF'*Temp.^4/pi;  % 辐射强度
```

**RMC3D的物理：**
- RDF是碰撞的**概率密度**
- 乘以σT⁴转换为辐射能量
- 这是**辐射传热**问题的G

**PINN的物理：**
- G = ∫ I dΩ（传输方程的解）
- 给定源项S(r)，无温度场
- 这是**RTE**问题的G

**两者G的物理含义不同！**

---

## 6. 解决方案

### 方案A：修正RMC代码（推荐）

实现带权重的碰撞估计器：
```python
# 在碰撞点记录权重 = S(r₀) × exp(-βs) / kappa
# 这样G = (1/V) × Σ weights
```

### 方案B：重新定义PINN的G

如果PINN的G实际上是"碰撞密度"而非"积分辐射强度"，需要调整PINN的后处理。

### 方案C：对比归一化的结果

不追求绝对值一致，而是对比：
- 形状（profile）
- 相对分布
- 中心/边缘比例

---

## 7. 建议

1. **先确认PINN中G的精确定义**
   - 查看`RadTrans3D_Complex.py`中的G计算
   - 确认单位（W/m² vs 无量纲）

2. **根据PINN的定义修正RMC**
   - 如果G = ∫ I dΩ，实现正确的碰撞估计器
   - 如果G是其他定义，相应调整

3. **或者使用相对误差对比**
   - 归一化到峰值
   - 对比形状而非绝对值

---

## 关键问题

**请检查PINN代码中G的计算方式：**

在`EquationModels/RadTrans3D_Complex.py`或后处理代码中，找到：
```python
# G是如何计算的？
G = ...  # ?
```

这决定了RMC应该如何计算G。
