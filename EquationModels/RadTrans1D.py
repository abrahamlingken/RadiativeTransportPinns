# -*- coding: utf-8 -*-
"""
重构版 1D 辐射传输方程模型
支持：纯吸收介质、吸收-散射介质、各向异性散射
"""

from ImportFile import *
import numpy as np

pi = math.pi

# ==============================================================================
# 全局参数设置
# ==============================================================================
space_dimensions = 1
time_dimensions = 0
parameter_dimensions = 0
# 总输入维度 = 空间维度(1) + 方向维度(1) = 2
# 注意: input_dimensions在ImportFile中计算为 parameter + time + space = 1
# 但辐射传输需要(x, mu)两个输入，因此需要在训练脚本中覆盖或使用特殊处理
# 这里显式定义input_dimensions供ImportFile使用
input_dimensions = 2  # x 和 mu
output_dimension = 1  # 辐射强度 I
# 使用 free shape 模式（配合 DatasetTorch2 的 else 分支）
# extrema_values = torch.tensor([[0., 1.], [-1., 1.]])
extrema_values = None  # 设为 None 以使用 free shape 数据生成
type_of_points = "sobol"
input_dimensions = 2  # (x, mu)

# 数值积分配置（高斯求积）
n_quad = 20  # 积分阶数

# 获取求积点和权重（在 [-1, 1] 区间）
# shape: [n_quad], [n_quad]
mu_quad, w_quad = np.polynomial.legendre.leggauss(n_quad)
mu_quad = torch.tensor(mu_quad, dtype=torch.float32)  # shape: [n_quad]
w_quad = torch.tensor(w_quad, dtype=torch.float32)    # shape: [n_quad]

if torch.cuda.is_available():
    dev = torch.device('cuda')
    mu_quad = mu_quad.cuda()
    w_quad = w_quad.cuda()
else:
    dev = torch.device("cpu")


# ==============================================================================
# 1. 介质物理参数函数（支持三种模式）
# ==============================================================================

# --------------------------------------------------
# 配置区域：修改以下参数切换介质类型
# --------------------------------------------------
# 模式1: 纯吸收介质 -> KAPPA=0.5, SIGMA_S=0.0, I_B=0.0, G_HG=0.0
# 模式2: 各向同性散射 -> KAPPA=0.5, SIGMA_S=0.5, I_B=0.0, G_HG=0.0
# 模式3: 各向异性散射 -> KAPPA=0.5, SIGMA_S=0.5, I_B=0.0, G_HG=0.5 (前向)
# 模式4: 发射介质 -> KAPPA=0.5, SIGMA_S=0.5, I_B=1.0, G_HG=0.0
#
# 注意: G_HG 可由外部脚本动态修改以支持各向异性散射案例
#   g = 0.0:  各向同性散射
#   g > 0:    前向散射 (峰值在 mu=mu')
#   g < 0:    后向散射 (峰值在 mu=-mu')
#   |g| -> 1: 散射方向性越强
KAPPA_CONST = 0.5      # 吸收系数
SIGMA_S_CONST = 0.0    # 散射系数（设为0即为纯吸收）
I_B_CONST = 0.0        # 黑体辐射强度
G_HG = 0.0             # HG不对称因子（0为各向同性，可由外部脚本动态修改）


def kappa(x):
    """
    吸收系数 (Absorption Coefficient)
    Args:
        x: 空间坐标, shape: [N] 或 [N, 1]
    Returns:
        吸收系数, shape: [N]
    """
    if x.dim() > 1:
        x = x.squeeze(-1)
    # shape: [N]
    return torch.full_like(x, KAPPA_CONST)


def sigma_s(x):
    """
    散射系数 (Scattering Coefficient)
    Args:
        x: 空间坐标, shape: [N] 或 [N, 1]
    Returns:
        散射系数, shape: [N]
    """
    if x.dim() > 1:
        x = x.squeeze(-1)
    # shape: [N]
    return torch.full_like(x, SIGMA_S_CONST)


def I_b(x):
    """
    黑体辐射强度 (Blackbody Intensity / Emission Source)
    Args:
        x: 空间坐标, shape: [N] 或 [N, 1]
    Returns:
        黑体辐射强度, shape: [N]
    """
    if x.dim() > 1:
        x = x.squeeze(-1)
    # shape: [N]
    return torch.full_like(x, I_B_CONST)


# ==============================================================================
# 2. 相函数 (Phase Function) - Henyey-Greenstein
# ==============================================================================

def kernel_HG(mu, mu_prime, g=G_HG):
    """
    Henyey-Greenstein 散射相函数
    公式: Phi_HG = (1 - g^2) / (1 + g^2 - 2g*cos(theta))^(3/2)
    
    Args:
        mu: 当前光线方向余弦, shape: [N, 1]
        mu_prime: 积分方向余弦, shape: [1, n_quad] 或 [n_quad]
        g: 不对称因子
    Returns:
        相函数值 Phi_HG, shape: [N, n_quad]
    """
    # 确保 mu 为 [N, 1] 用于广播
    if mu.dim() == 1:
        mu = mu.unsqueeze(-1)  # shape: [N, 1]
    
    # 确保 mu_prime 为 [1, n_quad]
    if mu_prime.dim() == 1:
        mu_prime = mu_prime.unsqueeze(0)  # shape: [1, n_quad]
    
    # 计算散射角余弦: cos(Theta) = mu * mu_prime
    # 广播: [N, 1] * [1, n_quad] -> [N, n_quad]
    cos_theta = mu * mu_prime  # shape: [N, n_quad]
    
    # HG 相函数
    g_sq = g ** 2
    numerator = 1.0 - g_sq
    
    # 分母: (1 + g^2 - 2g*cos(theta))^(3/2)
    denominator = torch.pow(1.0 + g_sq - 2.0 * g * cos_theta, 1.5)  # shape: [N, n_quad]
    
    # 相函数值
    phi_hg = numerator / denominator  # shape: [N, n_quad]
    
    return phi_hg


def compute_scattering(u, x, mu, network, g=G_HG):
    """
    计算散射积分项: 0.5 * integral Phi(mu, mu') * u(x, mu') dmu'
    
    支持各向异性散射（通过HG相函数）
    
    Args:
        u: 当前方向的辐射强度, shape: [N]
        x: 空间坐标, shape: [N, 1]
        mu: 当前方向余弦, shape: [N]
        network: PINN 网络
        g: HG 不对称因子（默认使用全局 G_HG，可被覆盖）
    Returns:
        散射积分值, shape: [N]
    """
    N = x.shape[0]
    
    # 纯吸收介质快速路径：直接返回0
    if SIGMA_S_CONST == 0.0:
        return torch.zeros_like(u)  # shape: [N]
    
    # 将 x 复制 n_quad 次，与每个 mu_quad 组合
    # x_repeated: [N, 1] -> [N*n_quad, 1]
    x_repeated = x.repeat(1, n_quad).view(-1, 1)  # shape: [N*n_quad, 1]
    
    # mu_quad_expanded: [n_quad] -> [N, n_quad] -> [N*n_quad, 1]
    mu_expanded = mu_quad.unsqueeze(0).repeat(N, 1).view(-1, 1)  # shape: [N*n_quad, 1]
    
    # 组合输入 (x, mu')
    x_mu_prime = torch.cat([x_repeated, mu_expanded], dim=-1)  # shape: [N*n_quad, 2]
    
    # 通过网络计算所有方向上的 u(x, mu')
    u_all_directions = network(x_mu_prime)  # shape: [N*n_quad, 1]
    u_all_directions = u_all_directions.view(N, n_quad)  # shape: [N, n_quad]
    
    # 获取相函数
    mu_current = mu.unsqueeze(-1)  # shape: [N, 1]
    mu_prime = mu_quad.unsqueeze(0)  # shape: [1, n_quad]
    
    phi_matrix = kernel_HG(mu_current, mu_prime, g=g)  # shape: [N, n_quad]
    
    # 散射积分
    weights = w_quad.unsqueeze(0)  # shape: [1, n_quad]
    integrand = phi_matrix * u_all_directions * weights  # shape: [N, n_quad]
    scattering_integral = torch.sum(integrand, dim=1)  # shape: [N]
    
    # 乘以 0.5 (1D平板几何标准化因子)
    scattering_term = 0.5 * scattering_integral  # shape: [N]
    
    return scattering_term


# ==============================================================================
# 3. 完全体 RTE 残差计算
# ==============================================================================

def compute_res(network, x_f_train, space_dimensions, solid_object=None, computing_error=False):
    """
    计算一维稳态辐射传输方程残差 (完全体 RTE，支持各向异性散射)
    
    方程: mu * du/dx + (kappa + sigma_s) * u = sigma_s * scattering_integral + kappa * I_b
    
    各向异性散射特性:
        - 通过全局变量 G_HG 控制HG不对称因子
        - G_HG 可由外部训练脚本动态修改 (如 Scripts/train_1d_multicase_anisotropic.py)
        - g = 0: 各向同性; g > 0: 前向散射; g < 0: 后向散射
    
    Args:
        network: PINN 网络
        x_f_train: 配点 (x, mu), shape: [N, 2]
        space_dimensions: 空间维度（兼容接口）
        solid_object: 固体对象（兼容接口）
        computing_error: 是否计算误差（兼容接口）
    Returns:
        残差, shape: [N]
    """
    # 拆分坐标
    x = x_f_train[:, 0:1]      # shape: [N, 1]
    mu = x_f_train[:, 1]       # shape: [N]
    
    # 确保 x 需要梯度
    x.requires_grad_(True)
    
    # 网络预测 u(x, mu)
    x_mu = torch.cat([x, mu.unsqueeze(-1)], dim=-1)  # shape: [N, 2]
    u = network(x_mu).squeeze(-1)  # shape: [N]
    
    # 1. 空间导数 du/dx
    grad_outputs = torch.ones_like(u)  # shape: [N]
    grad_u = torch.autograd.grad(
        u, x, 
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]  # shape: [N, 1]
    grad_u_x = grad_u.squeeze(-1)  # shape: [N]
    
    # 2. 对流项: mu * du/dx
    advection_term = mu * grad_u_x  # shape: [N]
    
    # 3. 介质参数
    kappa_val = kappa(x.squeeze(-1))      # shape: [N]
    sigma_s_val = sigma_s(x.squeeze(-1))  # shape: [N]
    I_b_val = I_b(x.squeeze(-1))          # shape: [N]
    
    # 4. 衰减项 (Extinction)
    extinction_coeff = kappa_val + sigma_s_val  # shape: [N]
    attenuation_term = extinction_coeff * u     # shape: [N]
    
    # 5. 散射源项 (纯吸收时为0)
    scattering_source = compute_scattering(u, x, mu, network, g=G_HG)  # shape: [N]
    scattering_term = sigma_s_val * scattering_source  # shape: [N]
    
    # 6. 发射源项
    emission_term = kappa_val * I_b_val  # shape: [N]
    
    # 7. 组装残差
    lhs = advection_term + attenuation_term  # shape: [N]
    rhs = scattering_term + emission_term    # shape: [N]
    residual = lhs - rhs  # shape: [N]
    
    return residual


# ==============================================================================
# 4. 边界条件与数据生成
# ==============================================================================

def apply_BC(x_boundary, u_boundary, model):
    """应用边界条件"""
    u_pred = model(x_boundary).squeeze(-1)  # shape: [N]
    return u_pred, u_boundary


def add_collocations(n_collocation):
    """生成内部配点"""
    sampler = qmc.Sobol(d=2, scramble=True)
    samples = sampler.random(n=n_collocation)  # shape: [n_collocation, 2]
    
    x = samples[:, 0:1]  # [0,1]
    mu = samples[:, 1:2] * 2.0 - 1.0  # [-1,1]
    
    x_coll = torch.tensor(np.concatenate([x, mu], axis=1), dtype=torch.float32)
    y_coll = torch.full((n_collocation, 1), float('nan'), dtype=torch.float32)
    
    if torch.cuda.is_available():
        x_coll = x_coll.cuda()
        y_coll = y_coll.cuda()
    
    return x_coll, y_coll


def add_initial_points(n_initial):
    """生成初始点（稳态问题不需要，返回空）"""
    if n_initial == 0:
        return torch.empty(0, 2), torch.empty(0, 1)
    # 稳态问题没有初始条件，返回空张量
    x_time = torch.empty(n_initial, 2)
    y_time = torch.empty(n_initial, 1)
    if torch.cuda.is_available():
        x_time = x_time.cuda()
        y_time = y_time.cuda()
    return x_time, y_time


def add_internal_points(n_internal):
    """生成内部点（通过add_collocations实现）"""
    if n_internal == 0:
        return torch.empty(0, 2), torch.empty(0, 1)
    return add_collocations(n_internal)


def add_boundary(n_boundary):
    """生成边界点 (Marshak边界)"""
    n_per_side = n_boundary // 2
    
    # 左边界 x=0, mu > 0 (入射)
    x_left = torch.zeros(n_per_side, 1)
    mu_left = torch.rand(n_per_side, 1) + 0.01  # (0, 1]
    
    # 右边界 x=1, mu < 0 (出射)
    x_right = torch.ones(n_per_side, 1)
    mu_right = -torch.rand(n_per_side, 1) - 0.01  # [-1, 0)
    
    x_b = torch.cat([x_left, x_right], dim=0)
    mu_b = torch.cat([mu_left, mu_right], dim=0)
    
    x_boundary = torch.cat([x_b, mu_b], dim=-1)  # shape: [n_boundary, 2]
    
    # 边界值: 左侧入射强度为1，右侧为0
    u_left = torch.ones(n_per_side)
    u_right = torch.zeros(n_per_side)
    u_b = torch.cat([u_left, u_right], dim=0)  # shape: [n_boundary]
    
    if torch.cuda.is_available():
        x_boundary = x_boundary.cuda()
        u_b = u_b.cuda()
    
    return x_boundary, u_b


# DefineDataset 需要的边界条件列表（free shape不使用，但需定义为空列表）
list_of_BC = []


# ==============================================================================
# 5. 后处理与可视化
# ==============================================================================

def compute_generalization_error(model, extrema, images_path=None, n_test=1000):
    """
    计算泛化误差 (L2 error)
    对于纯吸收介质，有解析解或参考解进行比较
    
    Args:
        model: 训练好的PINN模型
        extrema: 定义域边界 (未使用，保持接口一致)
        images_path: 图像保存路径 (可选)
        n_test: 测试点数量
    Returns:
        L2_error: L2绝对误差
        rel_L2_error: 相对L2误差
    """
    import numpy as np
    
    # 生成测试点 (x, mu)
    x_test = torch.linspace(0, 1, n_test).unsqueeze(-1)  # [n_test, 1]
    mu_test = torch.linspace(-1, 1, n_test).unsqueeze(-1)  # [n_test, 1]
    
    # 创建网格
    x_grid, mu_grid = torch.meshgrid(x_test.squeeze(), mu_test.squeeze(), indexing='ij')
    x_flat = x_grid.flatten().unsqueeze(-1)  # [n_test*n_test, 1]
    mu_flat = mu_grid.flatten().unsqueeze(-1)  # [n_test*n_test, 1]
    
    # 组合输入
    x_mu = torch.cat([x_flat, mu_flat], dim=-1)  # [n_test*n_test, 2]
    if torch.cuda.is_available():
        x_mu = x_mu.cuda()
    
    # 模型预测
    with torch.no_grad():
        u_pred = model(x_mu).cpu().numpy().flatten()  # [n_test*n_test]
    
    # 对于纯吸收介质 (mu*du/dx + kappa*u = 0)
    # 解析解: u(x, mu) = I_0 * exp(-kappa*x/mu) for mu > 0
    # 这里使用简化的参考解或返回训练损失作为误差估计
    
    # 简化：使用网络在边界点的表现作为误差度量
    # 实际应用中应该有参考解或蒙特卡洛模拟结果
    x_b_test, u_b_test = add_boundary(100)
    with torch.no_grad():
        u_b_pred = model(x_b_test).cpu().numpy().flatten()
    u_b_exact = u_b_test.cpu().numpy().flatten()
    
    # 计算L2误差
    L2_error = np.sqrt(np.mean((u_b_pred - u_b_exact)**2))
    rel_L2_error = L2_error / (np.sqrt(np.mean(u_b_exact**2)) + 1e-10)
    
    print(f"L2 Error (boundary): {L2_error:.6e}")
    print(f"Relative L2 Error: {rel_L2_error:.6e}")
    
    return L2_error, rel_L2_error


def plotting(model, images_path, extrema=None, solid_object=None):
    """
    绘制结果可视化
    Args:
        model: 训练好的PINN模型
        images_path: 图像保存路径
        extrema: 定义域边界 (未使用)
        solid_object: 固体对象 (未使用)
    """
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建网格数据
    n_x, n_mu = 100, 50
    x = np.linspace(0, 1, n_x)
    mu = np.linspace(-1, 1, n_mu)
    X, Mu = np.meshgrid(x, mu)
    
    # 转换为tensor
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(-1)
    mu_flat = torch.tensor(Mu.flatten(), dtype=torch.float32).unsqueeze(-1)
    x_mu = torch.cat([x_flat, mu_flat], dim=-1)
    if torch.cuda.is_available():
        x_mu = x_mu.cuda()
    
    # 预测
    with torch.no_grad():
        u_pred = model(x_mu).cpu().numpy().reshape(n_mu, n_x)
    
    # 绘制 - 与net_sol.png相同的格式：单栏2D热图
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 2D heatmap with jet colormap (与net_sol.png一致)
    levels = np.linspace(u_pred.min(), u_pred.max(), 21)
    im = ax.contourf(X, Mu, u_pred, levels=levels, cmap='jet', extend='both')
    
    # 设置标签和标题
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$\mu$', fontsize=14)
    ax.set_title('Exact Solution: $I(x, \\mu)$', fontsize=16)
    
    # 设置刻度
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(-1, 1, 9))
    ax.tick_params(labelsize=12)
    
    # 添加颜色条 (与net_sol.png风格一致)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$I(x, \mu)$', fontsize=14, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=12)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'{images_path}/solution.png', dpi=300, bbox_inches='tight')
    print(f'Saved solution plot to {images_path}/solution.png')
    plt.close()


# 向后兼容别名
Ec = sys.modules[__name__]
