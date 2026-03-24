"""
RadTrans3D_Complex.py - 3D稳态辐射传输方程物理引擎 (GPU极致优化版)

支持真实物理介质:
- 吸收系数 kappa(x, y, z)
- 散射系数 sigma_s(x, y, z)
- 黑体辐射源 I_b(x, y, z)
- 各向异性散射: 3D Henyey-Greenstein相函数

核心优化:
- 完全消灭for循环的散射积分
- 使用torch.einsum和广播机制
- 支持24GB显存的批量处理
"""

import torch
import numpy as np
from scipy.special import roots_legendre
import math

pi = math.pi

# ==============================================================================
# 设备配置
# ==============================================================================
if torch.cuda.is_available():
    dev = torch.device('cuda')
    print(f"[RadTrans3D_Complex] Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    dev = torch.device('cpu')
    print("[RadTrans3D_Complex] Using CPU device")

# ==============================================================================
# 全局配置参数
# ==============================================================================
space_dimensions = 3
time_dimensions = 0
output_dimension = 1
input_dimensions = 5  # x, y, z, theta, phi
type_of_points = "sobol"

# 3D立体角求积配置 (theta, phi)
N_THETA_QUAD = 8   # 极角方向求积点数
N_PHI_QUAD = 16    # 方位角方向求积点数
N_DIR_QUAD = N_THETA_QUAD * N_PHI_QUAD  # 总方向数: 128

# HG不对称因子 (全局可修改)
G_HG = 0.0  # 0=各向同性, >0=前向, <0=后向

# ==============================================================================
# 任务 1: 复杂物理参数函数
# ==============================================================================

def kappa(x, y, z):
    """
    吸收系数 kappa(x, y, z)
    
    Args:
        x, y, z: 空间坐标，shape: [N]
    
    Returns:
        吸收系数，shape: [N]
    
    示例实现: 高斯分布吸收体 (可自定义)
    """
    # shape: [N]
    kappa_val = torch.ones_like(x) * 0.5  # 基准值
    
    # 添加空间变化 (示例: 中心高斯增强)
    x0, y0, z0 = 0.5, 0.5, 0.5
    r_sq = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
    gaussian = torch.exp(-r_sq / 0.1)  # shape: [N]
    
    kappa_val = kappa_val + 0.3 * gaussian  # shape: [N]
    return kappa_val  # shape: [N]


def sigma_s(x, y, z):
    """
    散射系数 sigma_s(x, y, z)
    
    Args:
        x, y, z: 空间坐标，shape: [N]
    
    Returns:
        散射系数，shape: [N]
    """
    # shape: [N]
    sigma_val = torch.ones_like(x) * 0.5  # 基准值
    
    # 可添加空间变化
    return sigma_val  # shape: [N]


def I_b(x, y, z):
    """
    黑体辐射源项 I_b(x, y, z)
    
    Args:
        x, y, z: 空间坐标，shape: [N]
    
    Returns:
        黑体辐射强度，shape: [N]
    """
    # shape: [N]
    x0, y0, z0 = 0.5, 0.5, 0.5
    r = torch.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)  # shape: [N]
    
    # 球形热源: 中心高，向外递减
    source = torch.clamp(1.0 - 2.0 * r, min=0.0)  # shape: [N]
    return source  # shape: [N]


# ==============================================================================
# 任务 1: 3D Henyey-Greenstein 相函数
# ==============================================================================

def kernel_HG_3d(s, s_prime, g=G_HG):
    """
    3D Henyey-Greenstein相函数
    
    Phi_HG(s, s', g) = (1 - g^2) / (1 + g^2 - 2*g*cos(Theta))^{1.5}
    
    其中 cos(Theta) = s · s' (向量点乘)
    
    Args:
        s: 入射方向向量，shape: [N, 3] 或 [N, 1, 3]
        s_prime: 散射方向向量，shape: [N, N_q, 3] 或 [1, N_q, 3]
        g: HG不对称因子
    
    Returns:
        相函数值，shape: [N, N_q]
    """
    # 计算 cos(Theta) = s · s'
    # s: [N, 3] 或 [N, 1, 3], s_prime: [N, N_q, 3] 或 [1, N_q, 3]
    cos_theta = torch.sum(s * s_prime, dim=-1)  # shape: [N, N_q]
    
    # HG相函数
    g_sq = g ** 2
    numerator = 1.0 - g_sq  # scalar
    denominator = torch.pow(1.0 + g_sq - 2.0 * g * cos_theta, 1.5)  # shape: [N, N_q]
    
    Phi = numerator / denominator  # shape: [N, N_q]
    return Phi  # shape: [N, N_q]


# ==============================================================================
# 任务 2: 3D立体角散射积分 (完全消灭for循环)
# ==============================================================================

def generate_quadrature_3d(n_theta=N_THETA_QUAD, n_phi=N_PHI_QUAD):
    """
    生成3D立体角求积点和权重
    
    使用Gauss-Legendre求积(theta) + 均匀分布/梯形法则(phi)
    
    Returns:
        theta_q: 极角求积点，shape: [n_theta]
        phi_q: 方位角求积点，shape: [n_phi]
        w_theta: 极角权重，shape: [n_theta]
        w_phi: 方位角权重，shape: [n_phi]
        dir_vectors: 方向向量，shape: [n_theta*n_phi, 3]
        solid_angle_weights: 立体角权重 dOmega = sin(theta)*dtheta*dphi，shape: [n_theta*n_phi]
    """
    # Gauss-Legendre求积点 (theta ∈ [0, π])
    xi, w_xi = roots_legendre(n_theta)  # xi ∈ [-1, 1]
    theta_q = torch.tensor(np.arccos(-xi), dtype=torch.float32, device=dev)  # shape: [n_theta]
    # dtheta = d(arccos(-xi)) = dxi / sqrt(1-xi^2)，权重转换
    w_theta = torch.tensor(w_xi, dtype=torch.float32, device=dev)  # shape: [n_theta]
    
    # 均匀分布 (phi ∈ [0, 2π])
    phi_q = torch.linspace(0, 2*pi, n_phi+1, device=dev)[:-1]  # shape: [n_phi]
    phi_q = phi_q + pi / n_phi  # 中点偏移
    w_phi = torch.ones(n_phi, device=dev) * (2 * pi / n_phi)  # shape: [n_phi]
    
    # 创建2D网格
    theta_grid, phi_grid = torch.meshgrid(theta_q, phi_q, indexing='ij')  # shape: [n_theta, n_phi]
    
    # 转换为方向向量 s = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    sin_theta = torch.sin(theta_grid)  # shape: [n_theta, n_phi]
    cos_theta = torch.cos(theta_grid)  # shape: [n_theta, n_phi]
    cos_phi = torch.cos(phi_grid)  # shape: [n_theta, n_phi]
    sin_phi = torch.sin(phi_grid)  # shape: [n_theta, n_phi]
    
    # 方向向量，shape: [n_theta, n_phi, 3]
    dir_vectors = torch.stack([
        sin_theta * cos_phi,  # xi = sin(theta)cos(phi)
        sin_theta * sin_phi,  # eta = sin(theta)sin(phi)
        cos_theta             # mu = cos(theta)
    ], dim=-1)  # shape: [n_theta, n_phi, 3]
    
    # 展平为 [N_q, 3]
    dir_vectors = dir_vectors.reshape(-1, 3)  # shape: [n_theta*n_phi, 3]
    
    # 立体角权重: dOmega = sin(theta) * dtheta * dphi
    # w_theta是Gauss权重，w_phi是均匀权重
    theta_weights = w_theta.reshape(-1, 1)  # shape: [n_theta, 1]
    phi_weights = w_phi.reshape(1, -1)      # shape: [1, n_phi]
    
    # 2D权重网格，shape: [n_theta, n_phi]
    weight_grid = theta_weights * phi_weights * sin_theta
    
    # 展平，shape: [n_theta*n_phi]
    solid_angle_weights = weight_grid.reshape(-1)  # shape: [N_q]
    
    return theta_q, phi_q, w_theta, w_phi, dir_vectors, solid_angle_weights


# 预计算求积点 (全局缓存)
_THETA_Q, _PHI_Q, W_THETA, W_PHI, DIR_VECTORS, SOLID_ANGLE_WEIGHTS = generate_quadrature_3d()

# 创建完整的2D网格 (用于scatter计算)
N_THETA = N_THETA_QUAD
N_PHI = N_PHI_QUAD
_THETA_GRID, _PHI_GRID = torch.meshgrid(_THETA_Q, _PHI_Q, indexing='ij')  # shape: [n_theta, n_phi]
THETA_FLAT = _THETA_GRID.reshape(-1)  # shape: [N_q]
PHI_FLAT = _PHI_GRID.reshape(-1)      # shape: [N_q]


def compute_scattering_3d(x, y, z, theta, phi, model, n_theta=N_THETA_QUAD, n_phi=N_PHI_QUAD, g=G_HG):
    """
    计算3D散射积分: ∫∫ Phi(s, s') * u(x, y, z, theta', phi') dOmega'
    
    完全消灭for循环的GPU并行实现！
    
    Args:
        x, y, z: 空间坐标，shape: [N]
        theta: 当前极角，shape: [N]
        phi: 当前方位角，shape: [N]
        model: PINN网络
        n_theta, n_phi: 求积阶数
        g: HG不对称因子
    
    Returns:
        散射积分值，shape: [N]
    """
    N = x.shape[0]  # 配点数
    N_q = n_theta * n_phi  # 方向数
    
    # ============================================================
    # 步骤 1: 构造当前方向向量 s (用于相函数计算)
    # ============================================================
    sin_theta_curr = torch.sin(theta)  # shape: [N]
    cos_theta_curr = torch.cos(theta)  # shape: [N]
    cos_phi_curr = torch.cos(phi)      # shape: [N]
    sin_phi_curr = torch.sin(phi)      # shape: [N]
    
    # s: [N, 3] - 当前方向向量
    s_curr = torch.stack([
        sin_theta_curr * cos_phi_curr,  # xi
        sin_theta_curr * sin_phi_curr,  # eta
        cos_theta_curr                  # mu
    ], dim=-1)  # shape: [N, 3]
    
    # ============================================================
    # 步骤 2: 张量极速膨胀 - 构造 [N*N_q, 5] 输入张量
    # ============================================================
    
    # 空间坐标: [N, 3] -> 使用 repeat_interleave 拉长至 [N*N_q, 3]
    # 每个空间点重复 N_q 次，与所有方向组合
    xyz = torch.stack([x, y, z], dim=-1)  # shape: [N, 3]
    xyz_repeated = torch.repeat_interleave(xyz, N_q, dim=0)  # shape: [N*N_q, 3]
    
    # 方向坐标: [N_q] -> 使用 repeat 铺陈至 [N*N_q]
    # theta_q和phi_q是求积点，不是当前点的方向
    # 使用展平后的网格点
    theta_q_expanded = THETA_FLAT.repeat(N)  # shape: [N*N_q]
    phi_q_expanded = PHI_FLAT.repeat(N)      # shape: [N*N_q]
    
    # 或者直接使用预计算的DIR_VECTORS
    # DIR_VECTORS: [N_q, 3] -> 重复 N 次 -> [N*N_q, 3]
    dir_repeated = DIR_VECTORS.repeat(N, 1)  # shape: [N*N_q, 3]
    
    # 合并输入: [x, y, z, theta, phi]
    # 注意: 这里输入网络的是5维: (x, y, z, theta, phi)
    inputs = torch.cat([
        xyz_repeated,                    # shape: [N*N_q, 3]
        theta_q_expanded.unsqueeze(1),   # shape: [N*N_q, 1]
        phi_q_expanded.unsqueeze(1)      # shape: [N*N_q, 1]
    ], dim=-1)  # shape: [N*N_q, 5]
    
    # ============================================================
    # 步骤 3: 网络预测 u(x, s')
    # ============================================================
    # 根据显存批量处理
    batch_size = 50000  # 约 50000 * 5 * 4 bytes = 1MB，24GB显存可处理很大batch
    
    if inputs.shape[0] > batch_size:
        # 分批次预测
        u_pred_list = []
        for i in range(0, inputs.shape[0], batch_size):
            batch = inputs[i:i+batch_size]
            u_batch = model(batch).reshape(-1)  # shape: [batch_size]
            u_pred_list.append(u_batch)
        u_pred = torch.cat(u_pred_list, dim=0)  # shape: [N*N_q]
    else:
        u_pred = model(inputs).reshape(-1)  # shape: [N*N_q]
    
    # ============================================================
    # 步骤 4: Reshape 为 [N, N_q] 并计算相函数矩阵
    # ============================================================
    u_matrix = u_pred.reshape(N, N_q)  # shape: [N, N_q]
    
    # 计算相函数矩阵 Phi[s_curr, s_quadrature]
    # s_curr: [N, 3], DIR_VECTORS: [N_q, 3]
    # 需要广播: s_curr[:, None, :] * DIR_VECTORS[None, :, :]
    s_curr_expanded = s_curr.unsqueeze(1)           # shape: [N, 1, 3]
    dir_quad_expanded = DIR_VECTORS.unsqueeze(0)    # shape: [1, N_q, 3]
    
    Phi_matrix = kernel_HG_3d(s_curr_expanded, dir_quad_expanded, g=g)  # shape: [N, N_q]
    
    # ============================================================
    # 步骤 5: 矩阵化求和 (消灭最后的for循环！)
    # ============================================================
    # 散射积分: sum_j Phi_ij * u_j * w_j
    # SOLID_ANGLE_WEIGHTS: [N_q]
    weights_expanded = SOLID_ANGLE_WEIGHTS.unsqueeze(0)  # shape: [1, N_q]
    
    # 加权相函数: [N, N_q] * [N, N_q] * [1, N_q] = [N, N_q]
    integrand = Phi_matrix * u_matrix * weights_expanded  # shape: [N, N_q]
    
    # 沿方向维度求和: [N, N_q] -> [N]
    scatter_integral = torch.sum(integrand, dim=1)  # shape: [N]
    
    return scatter_integral  # shape: [N]


# ==============================================================================
# 任务 3: 组装3D完全体残差
# ==============================================================================

def compute_res(network, x_f_train, space_dimensions, solid_object, computing_error):
    """
    计算3D RTE残差 (完全体)
    
    PDE: s·∇u + (kappa + sigma_s)*u = (sigma_s / 4pi) * scatter + kappa*I_b
    
    残差 = s·∇u + (kappa + sigma_s)*u - (sigma_s / 4pi) * scatter - kappa*I_b
    
    Args:
        network: PINN网络
        x_f_train: 配点张量，shape: [N, 5] (x, y, z, theta, phi)
        space_dimensions: 空间维度 (3)
        solid_object: 固体对象 (兼容接口)
        computing_error: 是否计算误差 (兼容接口)
    
    Returns:
        残差，shape: [N]
    """
    # 启用梯度计算
    x_f_train = x_f_train.requires_grad_(True)
    
    # 解包输入
    x = x_f_train[:, 0]      # shape: [N]
    y = x_f_train[:, 1]      # shape: [N]
    z = x_f_train[:, 2]      # shape: [N]
    theta = x_f_train[:, 3]  # shape: [N]
    phi = x_f_train[:, 4]    # shape: [N]
    
    # ============================================================
    # 步骤 1: 计算方向余弦
    # ============================================================
    sin_theta = torch.sin(theta)  # shape: [N]
    cos_theta = torch.cos(theta)  # shape: [N]
    cos_phi = torch.cos(phi)      # shape: [N]
    sin_phi = torch.sin(phi)      # shape: [N]
    
    # 方向余弦: s = (xi, eta, mu)
    xi = sin_theta * cos_phi    # shape: [N]
    eta = sin_theta * sin_phi   # shape: [N]
    mu = cos_theta              # shape: [N]
    
    # ============================================================
    # 步骤 2: 网络预测 u(x, s)
    # ============================================================
    u = network(x_f_train).reshape(-1)  # shape: [N]
    
    # ============================================================
    # 步骤 3: 通过 autograd 获取空间梯度
    # ============================================================
    grad_outputs = torch.ones_like(u)
    
    grad_u = torch.autograd.grad(
        u, x_f_train,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]  # shape: [N, 5]
    
    grad_u_x = grad_u[:, 0]  # shape: [N]
    grad_u_y = grad_u[:, 1]  # shape: [N]
    grad_u_z = grad_u[:, 2]  # shape: [N]
    
    # ============================================================
    # 步骤 4: 计算方向导数 s·∇u
    # ============================================================
    s_dot_grad_u = xi * grad_u_x + eta * grad_u_y + mu * grad_u_z  # shape: [N]
    
    # ============================================================
    # 步骤 5: 计算物理系数
    # ============================================================
    kappa_val = kappa(x, y, z)      # shape: [N]
    sigma_val = sigma_s(x, y, z)    # shape: [N]
    I_b_val = I_b(x, y, z)          # shape: [N]
    
    # ============================================================
    # 步骤 6: 计算散射积分 (完全并行！)
    # ============================================================
    scatter_val = compute_scattering_3d(x, y, z, theta, phi, network, g=G_HG)  # shape: [N]
    
    # ============================================================
    # 步骤 7: 组装残差
    # ============================================================
    # RTE: s·∇u + (kappa + sigma_s)*u = (sigma_s / 4pi) * scatter + kappa*I_b
    # 注意: scatter_val 已经包含了积分，但还没有除以4pi (kernel_HG_3d返回的是Phi_HG，不是Phi_HG/4pi)
    
    lhs = s_dot_grad_u + (kappa_val + sigma_val) * u  # shape: [N]
    rhs = (sigma_val / (4.0 * pi)) * scatter_val + kappa_val * I_b_val  # shape: [N]
    
    residual = lhs - rhs  # shape: [N]
    
    return residual  # shape: [N]


# ==============================================================================
# 辅助函数: 入射辐射 G(x) 计算
# ==============================================================================

def compute_incident_radiation_3d(x, y, z, model, n_theta=N_THETA_QUAD, n_phi=N_PHI_QUAD):
    """
    计算入射辐射 G(x) = ∫∫ u(x, s) dOmega
    
    Args:
        x, y, z: 空间坐标，shape: [N]
        model: PINN网络
        n_theta, n_phi: 求积阶数
    
    Returns:
        G: 入射辐射，shape: [N]
    """
    N = x.shape[0]
    N_q = n_theta * n_phi
    
    # 构造输入
    xyz = torch.stack([x, y, z], dim=-1)  # shape: [N, 3]
    xyz_repeated = torch.repeat_interleave(xyz, N_q, dim=0)  # shape: [N*N_q, 3]
    
    theta_q_expanded = THETA_FLAT.repeat(N)  # shape: [N*N_q]
    phi_q_expanded = PHI_FLAT.repeat(N)      # shape: [N*N_q]
    
    inputs = torch.cat([
        xyz_repeated,
        theta_q_expanded.unsqueeze(1),
        phi_q_expanded.unsqueeze(1)
    ], dim=-1)  # shape: [N*N_q, 5]
    
    # 批量预测
    batch_size = 50000
    if inputs.shape[0] > batch_size:
        u_list = []
        for i in range(0, inputs.shape[0], batch_size):
            u_batch = model(inputs[i:i+batch_size]).reshape(-1)
            u_list.append(u_batch)
        u_pred = torch.cat(u_list, dim=0)
    else:
        u_pred = model(inputs).reshape(-1)  # shape: [N*N_q]
    
    u_matrix = u_pred.reshape(N, N_q)  # shape: [N, N_q]
    
    # 积分: G = sum(u * dOmega)
    weights_expanded = SOLID_ANGLE_WEIGHTS.unsqueeze(0)  # shape: [1, N_q]
    G = torch.sum(u_matrix * weights_expanded, dim=1)  # shape: [N]
    
    return G  # shape: [N]


# ==============================================================================
# 数据采样函数 (用于PINN训练)
# ==============================================================================

def get_points(samples, dim, type_point_param="sobol", random_seed=0):
    """生成采样点（返回GPU张量）"""
    from scipy.stats import qmc
    
    if type_point_param == "uniform":
        torch.manual_seed(random_seed)
        points = torch.rand([samples, dim], device=dev)
    elif type_point_param == "sobol":
        sampler = qmc.Sobol(d=dim, scramble=False)
        if random_seed > 0:
            sampler.fast_forward(random_seed)
        data = sampler.random(n=samples)
        points = torch.from_numpy(data).float().to(dev)
    return points


def add_collocations(n_collocation):
    """添加内部配点"""
    print(f"[RadTrans3D_Complex] Adding {n_collocation} collocation points")
    
    points = get_points(n_collocation, 5, "sobol", 0)
    
    # 空间坐标: [0, 1]^3
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # 角度坐标: theta ∈ [0, π], phi ∈ [0, 2π]
    theta = points[:, 3] * pi      # [0, 1] -> [0, π]
    phi = points[:, 4] * 2 * pi    # [0, 1] -> [0, 2π]
    
    # 内部点无监督标签
    u = torch.full((n_collocation, 1), float('nan'), device=dev)
    
    inputs = torch.stack([x, y, z, theta, phi], dim=-1)  # shape: [N, 5]
    return inputs, u


def add_boundary(n_boundary):
    """添加边界点"""
    print(f"[RadTrans3D_Complex] Adding {n_boundary} boundary points")
    
    points = get_points(n_boundary, 5, "sobol", 16)
    
    # 分布在6个面上
    n_per_face = n_boundary // 6
    
    x = points[:, 0].clone()
    y = points[:, 1].clone()
    z = points[:, 2].clone()
    theta = points[:, 3] * pi
    phi = points[:, 4] * 2 * pi
    
    # 设置边界位置
    for i in range(6):
        mask = torch.arange(n_boundary, device=dev) // n_per_face == i
        if i == 0:  # x = 0
            x[mask] = 0.0
        elif i == 1:  # x = 1
            x[mask] = 1.0
        elif i == 2:  # y = 0
            y[mask] = 0.0
        elif i == 3:  # y = 1
            y[mask] = 1.0
        elif i == 4:  # z = 0
            z[mask] = 0.0
        elif i == 5:  # z = 1
            z[mask] = 1.0
    
    inputs = torch.stack([x, y, z, theta, phi], dim=-1)
    
    # 零Dirichlet边界 (流入边界)
    u_boundary = torch.zeros(n_boundary, 1, device=dev)
    
    return inputs, u_boundary


def apply_BC(x_boundary, u_boundary, model):
    """应用边界条件"""
    x = x_boundary[:, 0]
    y = x_boundary[:, 1]
    z = x_boundary[:, 2]
    
    # 计算外法向
    n = torch.zeros_like(x_boundary[:, :3])
    n[:, 0] = torch.where(x < 0.01, -1.0, torch.where(x > 0.99, 1.0, 0.0))
    n[:, 1] = torch.where(y < 0.01, -1.0, torch.where(y > 0.99, 1.0, 0.0))
    n[:, 2] = torch.where(z < 0.01, -1.0, torch.where(z > 0.99, 1.0, 0.0))
    
    # 方向向量
    theta = x_boundary[:, 3]
    phi = x_boundary[:, 4]
    s = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], dim=-1)
    
    # 流入边界判断 (n·s < 0)
    n_dot_s = torch.sum(n * s, dim=1)
    inflow_mask = n_dot_s < 0
    
    if inflow_mask.sum() == 0:
        return torch.tensor([], device=dev), torch.tensor([], device=dev)
    
    x_inflow = x_boundary[inflow_mask]
    u_inflow = u_boundary[inflow_mask]
    
    u_pred = model(x_inflow).reshape(-1)
    
    return u_pred, u_inflow.reshape(-1)


# ==============================================================================
# 可视化函数
# ==============================================================================

def plotting(model, images_path, extrema=None, solid=None):
    """可视化辐射场 G(x)"""
    try:
        from pyevtk.hl import gridToVTK
    except ImportError:
        print("[Warning] pyevtk not installed, skipping VTK export")
        return
    
    print("[RadTrans3D_Complex] Generating visualization...")
    
    n = 25  # 网格分辨率
    x = torch.linspace(0, 1, n, device=dev)
    y = torch.linspace(0, 1, n, device=dev)
    z = torch.linspace(0, 1, n, device=dev)
    
    # 生成网格
    X_grid, Y_grid, Z_grid = torch.meshgrid(x, y, z, indexing='ij')
    
    # 展平
    x_flat = X_grid.reshape(-1)
    y_flat = Y_grid.reshape(-1)
    z_flat = Z_grid.reshape(-1)
    
    # 计算G
    G = compute_incident_radiation_3d(x_flat, y_flat, z_flat, model)
    
    # 转numpy
    X_np = X_grid.cpu().numpy()
    Y_np = Y_grid.cpu().numpy()
    Z_np = Z_grid.cpu().numpy()
    G_np = G.reshape(n, n, n).cpu().numpy()
    
    # 保存VTK
    import os
    os.makedirs(images_path, exist_ok=True)
    gridToVTK(os.path.join(images_path, "G_field"), X_np, Y_np, Z_np, 
              pointData={"G": G_np})
    
    print(f"[RadTrans3D_Complex] Saved to {images_path}/G_field.vtu")
    print(f"  G range: [{G_np.min():.6f}, {G_np.max():.6f}]")


def compute_generalization_error(model, extrema, images_path=None):
    """计算泛化误差 (预留接口)"""
    return 0.0, 0.0
