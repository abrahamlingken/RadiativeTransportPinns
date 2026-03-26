# PINN 训练优化指南 - 3D RTE 高散射问题

## 问题诊断

当前配置 `kappa=5.0` 导致：
- 光学厚度 τ = 5.0（极强衰减）
- 源区半径 0.2 ≈ 平均自由程（梯度极陡峭）
- tanh 激活难以捕捉 r=0.2 处的锐利截止

## 方案一：网络架构优化（推荐优先尝试）

### 1.1 激活函数选择

```python
# 当前（平滑，难以捕捉锐利边界）
NETWORK_PROPERTIES = {
    "activation": "tanh",  # 平滑，饱和梯度
}

# 推荐选项 A：Swish（自门控，非单调，更适合陡峭梯度）
NETWORK_PROPERTIES = {
    "activation": "swish",
}

# 推荐选项 B：Adaptive Activation（可学习频率）
# 在 ModelClassTorch2.py 中添加：
class AdaptiveActivation(nn.Module):
    def __init__(self, activation=nn.Tanh()):
        super().__init__()
        self.activation = activation
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        return self.activation(self.scale * x)
```

### 1.2 网络深度与宽度

```python
# 当前配置
NETWORK_PROPERTIES = {
    "hidden_layers": 8,
    "neurons": 128,
}

# 推荐配置（更深更宽，增强表达能力）
NETWORK_PROPERTIES = {
    "hidden_layers": 10,      # +2层
    "neurons": 256,           # 2倍宽度
    "residual_parameter": 0.01,  # 降低残差权重初始值
}
```

### 1.3 傅里叶特征嵌入（Fourier Features）

```python
# 在输入层添加傅里叶特征，增强高频捕捉能力
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim, mapping_size=256, scale=10):
        super().__init__()
        self.B = torch.randn((input_dim, mapping_size)) * scale
    
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# 使用：将输入 [x,y,z,theta,phi] 映射到高维
```

## 方案二：优化器与学习率调度

### 2.1 两阶段训练策略

```python
def train_two_stage(case_key):
    """第一阶段：Adam快速收敛，第二阶段：LBFGS精细优化"""
    
    # ========== 第一阶段：Adam预热 ==========
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam, patience=50, factor=0.5, verbose=True
    )
    
    for epoch in range(2000):  # 2000轮Adam预热
        loss = compute_loss(...)
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()
        scheduler.step(loss)
        
        if epoch % 100 == 0:
            print(f"Adam Epoch {epoch}, Loss: {loss.item():.6e}")
    
    # ========== 第二阶段：LBFGS精细优化 ==========
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=0.1,                  # 降低学习率
        max_iter=100000,
        tolerance_grad=1e-12,    # 更严格
        tolerance_change=1e-14,  # 更严格
        history_size=200,        # 更大历史
    )
    
    # 继续LBFGS训练...
```

### 2.2 自适应损失权重

```python
# 当前：固定权重
lambda_residual = 0.1

# 推荐：自适应权重（边界条件更重要）
def compute_adaptive_loss(model, x_coll, x_b, y_b, physics):
    # 边界损失（通常更难满足）
    u_pred_b, u_train_b = physics.apply_bc(x_b, y_b, model)
    loss_bc = torch.mean((u_pred_b - u_train_b)**2)
    
    # 残差损失
    res = physics.compute_res(model, x_coll)
    loss_res = torch.mean(res**2)
    
    # 自适应权重：边界条件权重随训练增加
    epoch = get_current_epoch()
    lambda_bc = 10.0 + epoch / 100  # 从10开始逐渐增加
    lambda_res = 1.0
    
    total_loss = lambda_bc * loss_bc + lambda_res * loss_res
    return total_loss
```

## 方案三：课程学习（Curriculum Learning）

### 3.1 渐进式难度增加

```python
def curriculum_training(case_key):
    """从简单问题逐步过渡到复杂问题"""
    
    # 阶段 1：训练 kappa=1.0（简单）
    print("Stage 1: Training with kappa=1.0...")
    physics_easy = RadTrans3D_Physics(kappa_val=1.0, sigma_s_val=0.0, ...)
    train(model, physics_easy, epochs=1000)
    
    # 阶段 2：训练 kappa=2.0（中等）
    print("Stage 2: Training with kappa=2.0...")
    physics_medium = RadTrans3D_Physics(kappa_val=2.0, sigma_s_val=0.0, ...)
    train(model, physics_medium, epochs=1000)
    
    # 阶段 3：训练 kappa=5.0（目标）
    print("Stage 3: Training with kappa=5.0...")
    physics_hard = RadTrans3D_Physics(kappa_val=5.0, sigma_s_val=0.0, ...)
    train(model, physics_hard, epochs=2000)
```

### 3.2 渐进式源区收缩

```python
def progressive_source_sharpening(case_key):
    """逐步收缩源区，让网络学习锐利边界"""
    
    # 修改 I_b 函数接受收缩参数
    for sharpness in [2.0, 3.0, 4.0, 5.0]:
        print(f"Training with source sharpness = {sharpness}...")
        
        # 临时修改源项
        physics.sharpness = sharpness
        train_for_iterations(model, physics, iterations=500)
```

## 方案四：采样策略优化

### 4.1 自适应重采样

```python
def adaptive_resampling(model, physics, n_points=16384):
    """在残差大的区域增加采样点"""
    
    # 均匀采样初始点
    x_uniform = sample_uniform(n_points)
    
    # 计算残差
    with torch.no_grad():
        res = physics.compute_res(model, x_uniform)
        res_magnitude = torch.abs(res)
    
    # 根据残差大小重采样（残差大的区域采样更多）
    weights = res_magnitude / res_magnitude.sum()
    indices = torch.multinomial(weights, n_points, replacement=True)
    x_adaptive = x_uniform[indices]
    
    return x_adaptive
```

### 4.2 边界强化采样

```python
def generate_boundary_points_enhanced(n_points, n_boundary_per_face=256):
    """在边界处加密采样"""
    
    # 每个面采样点数
    points_per_face = []
    for face in ['x=0', 'x=1', 'y=0', 'y=1', 'z=0', 'z=1']:
        # 在边界附近增加采样密度
        dense_points = sample_near_boundary(face, n_boundary_per_face)
        points_per_face.append(dense_points)
    
    return torch.cat(points_per_face, dim=0)
```

## 方案五：物理约束增强

### 5.1 单调性约束

```python
def add_monotonicity_loss(model, physics):
    """添加单调递减约束（从中心向外）"""
    
    # 采样径向方向上的点
    r_samples = torch.linspace(0, 0.5, 20)
    center = torch.tensor([[0.5, 0.5, 0.5]])
    
    G_values = []
    for r in r_samples:
        x = center + r * torch.tensor([[1, 0, 0]])  # 沿x方向
        inputs = torch.cat([x, torch.tensor([[np.pi/2, 0]])], dim=-1)
        G = model(inputs)
        G_values.append(G)
    
    G_values = torch.stack(G_values)
    
    # 惩罚：如果 G 不是单调递减
    diff = G_values[1:] - G_values[:-1]
    monotonicity_loss = torch.sum(torch.relu(diff))  # 惩罚正值（递增）
    
    return monotonicity_loss
```

### 5.2 能量守恒约束

```python
def add_energy_conservation_loss(model, physics):
    """添加能量守恒约束"""
    
    # 计算总发射功率
    total_emission = compute_total_emission(physics)
    
    # 计算总吸收功率（沿所有边界积分）
    total_absorption = compute_boundary_absorption(model, physics)
    
    # 损失：发射与吸收的差
    conservation_loss = (total_emission - total_absorption)**2
    
    return conservation_loss
```

## 推荐实施顺序

| 优先级 | 方案 | 预期改进 | 实施难度 |
|--------|------|----------|----------|
| 1 | **激活函数改为 Swish** | +20-30% 精度 | ⭐ 简单 |
| 2 | **网络加宽至256** | +15-25% 精度 | ⭐ 简单 |
| 3 | **两阶段训练 (Adam+LBFGS)** | +30-40% 精度 | ⭐⭐ 中等 |
| 4 | **自适应边界权重** | +10-20% 精度 | ⭐⭐ 中等 |
| 5 | **课程学习 (渐进κ)** | +40-50% 精度 | ⭐⭐⭐ 复杂 |
| 6 | **傅里叶特征嵌入** | +20-30% 精度 | ⭐⭐⭐ 复杂 |

## 快速实施代码

```python
# 在 train_3d_multicase.py 中修改

NETWORK_PROPERTIES = {
    "hidden_layers": 10,
    "neurons": 256,
    "residual_parameter": 0.01,  # 降低残差权重
    "activation": "swish",  # 或 "tanh" with adaptive scaling
    # ... 其他参数
}

# 优化器配置
if case_key == '3D_A' and config['kappa'] >= 5.0:
    # 高kappa使用特殊配置
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 先Adam预热2000轮
    for epoch in range(2000):
        ...
    # 再LBFGS精细优化
    optimizer = optim.LBFGS(..., lr=0.1, tolerance_grad=1e-12)
```

## 预期结果

实施上述优化后，预期 Case A (κ=5.0)：
- 当前误差：~65%
- 优化后误差：<15%
- 训练时间：+50-100%（值得投入）
