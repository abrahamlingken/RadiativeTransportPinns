"""
严格对照论文Section 3.3的3D稳态单色辐射传输模型

论文参数:
- 区域: D = [0,1]^3 (单位立方体)
- 中心点源: c = (0.5, 0.5, 0.5)
- 源项: f(x) = k(x) * I_b(x)
- I_b(x) = max(0, 0.5 - r), r = |x - c| (半径0.5的球内线性递减)
- 吸收系数: k(x) = I_b(x)
- 散射系数: sigma = 1 (各向同性)
- 边界条件: 零Dirichlet (无外部辐射进入)
- 求积阶数: N_S = 100 (论文值，需大显存)

网络参数 (Table 3):
- N_int = 16384
- N_sb = 12288
- K-1 = 8 (隐藏层数)
- d̃ = 24 (每层的神经元数)
- lambda = 0.1
"""

from ImportFile import *
from pyevtk.hl import gridToVTK

pi = math.pi

space_dimensions = 3
time_dimensions = 0
output_dimension = 1
extrema_values = None

# 方向参数: phi ∈ [0, 2π], theta ∈ [0, π]
parameters_values = torch.tensor([[0, 2 * pi],  # phi
                                  [0, pi],      # theta
                                  [0, 1]])      # 占位符，保持与之前代码兼容
type_of_points = "sobol"
input_dimensions = 3
kernel_type = "isotropic"

# 论文使用N_S = 100，但为了内存考虑可以先用50
n_quad = 20  # 求积阶数（20x20=400个方向点，平衡精度和内存）

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")


def tile(a, dim, n_tile):
    """辅助函数：在指定维度上重复张量"""
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index.to(dev))


def I0(x, y, z):
    """
    源项I_b(x) = max(0, 0.5 - r)
    其中r = |x - c|, c = (0.5, 0.5, 0.5)
    论文公式 (3.4)
    """
    x0, y0, z0 = 0.5, 0.5, 0.5
    r = torch.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    source = 0.5 - r
    # 使用torch.where确保在r > 0.5时为0
    return torch.where(r > 0.5, torch.zeros_like(source), source)


def K(x, y, z):
    """
    吸收系数 k(x) = I_b(x)
    论文设定与源项相同
    """
    return I0(x, y, z)


def S(x, y, z):
    """
    散射系数 sigma(x) = 1
    论文设定为常数1
    """
    return torch.ones_like(x)


def kernel(s, s_prime, kernel_type):
    """
    散射核函数
    各向同性: Phi = 1/(4*pi)
    """
    if kernel_type == "isotropic":
        return 1.0 / (4.0 * pi)
    elif kernel_type == "HG":
        # Henyey-Greenstein核（本算例不使用）
        gamma = 0.5
        cos_theta = torch.sum(s * s_prime, dim=1, keepdim=True)
        k = (1 - gamma**2) / (1 + gamma**2 - 2*gamma*cos_theta)**1.5
        return k / (4.0 * pi)


def get_s(params):
    """
    将球坐标(theta, phi)转换为方向向量s = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    """
    s = torch.zeros(params.shape[0], 3, device=dev)
    phi = params[:, 0]
    theta = params[:, 1]
    s[:, 0] = torch.sin(theta) * torch.cos(phi)
    s[:, 1] = torch.sin(theta) * torch.sin(phi)
    s[:, 2] = torch.cos(theta)
    return s


def compute_scatter(x_train, model):
    """
    计算散射积分: ∫∫ u(x, s') * Phi(s·s') ds'
    使用Gauss-Legendre求积公式，分批处理避免OOM
    
    论文使用N_S = 100个求积点，这里使用n_quad×n_quad网格
    """
    # Gauss-Legendre求积点和权重
    quad_points, w = np.polynomial.legendre.leggauss(n_quad)
    
    # 创建2D求积网格 (phi, theta)
    phi_quad = pi * (quad_points + 1)  # 映射到[0, 2π]
    theta_quad = pi/2 * (quad_points + 1)  # 映射到[0, π]
    
    # 创建所有方向组合
    phi_grid, theta_grid = np.meshgrid(phi_quad, theta_quad)
    phi_flat = phi_grid.flatten()
    theta_flat = theta_grid.flatten()
    
    # 转换为方向向量
    directions = np.zeros((len(phi_flat), 3))
    directions[:, 0] = np.sin(theta_flat) * np.cos(phi_flat)
    directions[:, 1] = np.sin(theta_flat) * np.sin(phi_flat)
    directions[:, 2] = np.cos(theta_flat)
    
    # 权重处理（包含sin(theta)雅可比行列式）
    w_phi = w * pi  # dphi = pi * dxi
    w_theta = w * pi/2  # dtheta = (pi/2) * deta
    W = np.outer(w_theta, w_phi).flatten()  # 2D权重
    sin_theta = np.sin(theta_flat)
    weights = W * sin_theta  # 包含sin(theta)
    
    # 转换为torch
    directions = torch.from_numpy(directions).float().to(dev)
    weights = torch.from_numpy(weights).float().to(dev)
    
    # 分批处理以避免OOM
    n_points = x_train.shape[0]
    n_dirs = directions.shape[0]
    phi_factor = 1.0 / (4.0 * pi)
    
    # 确定批次大小（根据显存调整）
    # 每个点需要n_dirs次评估，总内存 = batch_size * n_dirs * 6 (input_dim) * 4 bytes
    # 24GB显存，n_dirs=400，建议batch_size=256或512
    batch_size = 256
    
    scatter_values = torch.zeros(n_points, device=dev)
    phys_coord = x_train[:, :3]
    
    for i in range(0, n_points, batch_size):
        end_i = min(i + batch_size, n_points)
        phys_batch = phys_coord[i:end_i]
        
        # 扩展当前批次的物理坐标到所有方向
        phys_repeated = phys_batch.repeat_interleave(n_dirs, dim=0)
        dirs_repeated = directions.repeat(end_i - i, 1)
        
        # 合并输入
        inputs = torch.cat([phys_repeated, dirs_repeated], dim=1)
        
        # 网络预测（需要梯度用于训练）
        u_batch = model(inputs).reshape(-1)
        
        # reshape为 (batch_size, n_dirs)
        u_batch = u_batch.reshape(end_i - i, n_dirs)
        
        # 计算加权积分
        scatter_values[i:end_i] = phi_factor * torch.matmul(u_batch, weights)
        
        # 清理缓存
        del inputs, u_batch, phys_repeated, dirs_repeated
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    
    return scatter_values


def compute_res(network, x_f_train, space_dimensions, solid_object, computing_error):
    """
    计算PDE残差
    稳态辐射传输方程:
    s·∇u + (k + σ)u = σ∫Φ(s·s')u ds' + k*I_b
    
    残差 = s·∇u + (k + σ)u - σ*scatter - k*I_b
    """
    x_f_train.requires_grad = True
    
    x = x_f_train[:, 0]
    y = x_f_train[:, 1]
    z = x_f_train[:, 2]
    s = x_f_train[:, 3:6]  # 方向向量
    
    # 网络输出 u(x, s)
    u = network(x_f_train).reshape(-1)
    
    # 计算梯度 ∇u
    grad_u = torch.autograd.grad(
        u, x_f_train,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    grad_u_x = grad_u[:, 0]
    grad_u_y = grad_u[:, 1]
    grad_u_z = grad_u[:, 2]
    
    # 方向导数 s·∇u
    s_dot_grad_u = s[:, 0] * grad_u_x + s[:, 1] * grad_u_y + s[:, 2] * grad_u_z
    
    # 系数
    k = K(x, y, z)
    sigma = S(x, y, z)
    source = I0(x, y, z)
    
    # 散射积分
    scatter = compute_scatter(x_f_train, network)
    
    # PDE残差
    res = s_dot_grad_u + (k + sigma) * u - sigma * scatter - k * source
    
    return res


def generator_domain_samples(points, boundary=False):
    """
    生成域内或边界上的采样点
    对于边界点，设置在立方体的6个面上
    """
    if boundary:
        n_per_face = points.shape[0] // 6
        for i in range(3):  # x, y, z三个方向
            # 两个面：一个在某坐标=0，一个在=1
            mask_low = np.arange(points.shape[0]) // n_per_face == 2*i
            mask_high = np.arange(points.shape[0]) // n_per_face == 2*i + 1
            
            if 2*i < points.shape[0] // n_per_face:
                points[mask_low, i] = 0.0
            if 2*i + 1 < points.shape[0] // n_per_face:
                points[mask_high, i] = 1.0
    
    return points


def generator_param_samples(points):
    """
    生成方向采样点 (phi, theta)
    phi ∈ [0, 2π], theta ∈ [0, π]
    """
    phi = points[:, 0] * 2 * pi  # [0, 1] -> [0, 2π]
    theta = points[:, 1] * pi     # [0, 1] -> [0, π]
    
    s = torch.zeros(points.shape[0], 3, device=dev)
    s[:, 0] = torch.sin(theta) * torch.cos(phi)
    s[:, 1] = torch.sin(theta) * torch.sin(phi)
    s[:, 2] = torch.cos(theta)
    
    return s


def get_points(samples, dim, type_point_param, random_seed):
    """生成采样点（返回GPU张量）"""
    if type_point_param == "uniform":
        torch.random.manual_seed(random_seed)
        points = torch.rand([samples, dim], device=dev)
    elif type_point_param == "sobol":
        sampler = qmc.Sobol(d=dim, scramble=False)
        if random_seed > 0:
            sampler.fast_forward(random_seed)
        data = sampler.random(n=samples)
        points = torch.from_numpy(data).float().to(dev)
    return points


def add_boundary(n_boundary):
    """
    添加边界点
    论文使用零Dirichlet边界条件
    只考虑流入边界 (n·s < 0)
    """
    print(f"Adding Boundary: {n_boundary} points")
    
    # 生成边界上的物理点和方向
    points = get_points(n_boundary, 5, type_of_points, 16)
    dom = points[:, :3].clone()
    angles = points[:, 3:]
    
    # 确保点在边界上
    dom = generator_domain_samples(dom, boundary=True)
    s = generator_param_samples(angles)
    
    # 确保所有张量在同一设备上
    dom = dom.to(dev)
    ub = torch.zeros(n_boundary, 1, device=dev)
    
    return torch.cat([dom, s], dim=1), ub


def add_collocations(n_collocation):
    """
    添加内部配点
    """
    print(f"Adding Collocations: {n_collocation} points")
    
    points = get_points(n_collocation, 5, type_of_points, 16)
    dom_int = points[:, :3]
    angles_int = points[:, 3:]
    
    # 确保点在域内 (0,1)^3
    dom = generator_domain_samples(dom_int, boundary=False)
    s = generator_param_samples(angles_int)
    
    # 确保所有张量在同一设备上
    dom = dom.to(dev)
    # 内部点没有监督标签（NaN表示）
    u = torch.full((n_collocation, 1), float('nan'), device=dev)
    
    return torch.cat([dom, s], dim=1), u


def add_internal_points(n_internal):
    """
    添加内部点（与add_collocations功能相同，用于兼容性）
    """
    print(f"Adding Internal Points: {n_internal} points")
    return add_collocations(n_internal)


def apply_BC(x_boundary, u_boundary, model):
    """
    应用边界条件
    只选择流入边界上的点 (n·s < 0)
    """
    x = x_boundary[:, 0]
    y = x_boundary[:, 1]
    z = x_boundary[:, 2]
    s = x_boundary[:, 3:6]
    
    # 计算外法向
    # 对于立方体[0,1]^3，边界上的外法向就是坐标本身（0或1）映射到-1或1
    n = torch.zeros_like(x_boundary[:, :3])
    n[:, 0] = torch.where(x < 0.01, torch.tensor(-1.), torch.where(x > 0.99, torch.tensor(1.), torch.tensor(0.)))
    n[:, 1] = torch.where(y < 0.01, torch.tensor(-1.), torch.where(y > 0.99, torch.tensor(1.), torch.tensor(0.)))
    n[:, 2] = torch.where(z < 0.01, torch.tensor(-1.), torch.where(z > 0.99, torch.tensor(1.), torch.tensor(0.)))
    
    # 计算n·s
    n_dot_s = (n * s).sum(dim=1)
    
    # 只选择流入边界 (n·s < 0)
    inflow_mask = n_dot_s < 0
    
    x_boundary_inf = x_boundary[inflow_mask]
    u_boundary_inf = u_boundary[inflow_mask]
    
    if x_boundary_inf.shape[0] == 0:
        # 如果没有流入点，返回空
        return torch.tensor([], device=dev), torch.tensor([], device=dev)
    
    u_pred = model(x_boundary_inf)
    
    return u_pred.reshape(-1), u_boundary_inf.reshape(-1)


def exact(x, y, s1, s2):
    """精确解（如果有的话）"""
    return torch.zeros_like(x)


def convert(vector):
    """转换函数"""
    return torch.from_numpy(np.array(vector)).float()


def compute_generalization_error(model, extrema, images_path=None):
    """计算泛化误差（论文没有提供3D的解析解）"""
    return 0.0, 0.0


def plotting(model, images_path, extrema, solid):
    """
    可视化辐射场G
    G = ∫∫ u(x, s) ds (对角度积分)
    """
    from scipy import integrate
    
    print("Generating visualization...")
    
    # 创建网格（使用较高的分辨率以获得平滑的可视化）
    n = 30  # 可以根据内存调整
    x = torch.linspace(0, 1, n)
    y = torch.linspace(0, 1, n)
    z = torch.linspace(0, 1, n)
    
    # 生成所有网格点
    phys_coord = torch.zeros(n**3, 3)
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                phys_coord[idx, 0] = x[i]
                phys_coord[idx, 1] = y[j]
                phys_coord[idx, 2] = z[k]
                idx += 1
    
    phys_coord = phys_coord.to(dev)
    
    # 计算G（入射辐射）
    G = compute_incident_radiation(phys_coord, model)
    
    # 转换为3D数组
    X = np.zeros((n, n, n))
    Y = np.zeros((n, n, n))
    Z = np.zeros((n, n, n))
    G_ = np.zeros((n, n, n))
    
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                X[i, j, k] = x[i].item()
                Y[i, j, k] = y[j].item()
                Z[i, j, k] = z[k].item()
                G_[i, j, k] = G[idx].cpu().detach().numpy()
                idx += 1
    
    # 保存VTK文件
    gridToVTK(images_path + "/structured", X, Y, Z, pointData={"G": G_})
    
    print(f"Visualization saved to {images_path}/structured.vtu")
    print(f"G range: [{G_.min():.6f}, {G_.max():.6f}]")


def compute_incident_radiation(phys_coord, model):
    """
    计算入射辐射 G(x) = ∫∫ u(x, s) ds
    使用与compute_scatter相同的求积公式
    """
    quad_points, w = np.polynomial.legendre.leggauss(n_quad)
    
    phi_quad = pi * (quad_points + 1)
    theta_quad = pi/2 * (quad_points + 1)
    
    phi_grid, theta_grid = np.meshgrid(phi_quad, theta_quad)
    phi_flat = phi_grid.flatten()
    theta_flat = theta_grid.flatten()
    
    directions = np.zeros((len(phi_flat), 3))
    directions[:, 0] = np.sin(theta_flat) * np.cos(phi_flat)
    directions[:, 1] = np.sin(theta_flat) * np.sin(phi_flat)
    directions[:, 2] = np.cos(theta_flat)
    
    w_phi = w * pi
    w_theta = w * pi/2
    W = np.outer(w_theta, w_phi).flatten()
    sin_theta = np.sin(theta_flat)
    weights = W * sin_theta
    
    directions = torch.from_numpy(directions).float().to(dev)
    weights = torch.from_numpy(weights).float().to(dev)
    
    n_points = phys_coord.shape[0]
    n_dirs = directions.shape[0]
    
    # 分批处理以避免OOM
    batch_size = 1000
    G = torch.zeros(n_points, device=dev)
    
    for i in range(0, n_points, batch_size):
        end_i = min(i + batch_size, n_points)
        phys_batch = phys_coord[i:end_i]
        
        # 扩展
        phys_repeated = phys_batch.repeat_interleave(n_dirs, dim=0)
        dirs_repeated = directions.repeat(end_i - i, 1)
        
        inputs = torch.cat([phys_repeated, dirs_repeated], dim=1)
        u = model(inputs).reshape(-1)
        u = u.reshape(end_i - i, n_dirs)
        
        # 积分
        G[i:end_i] = torch.matmul(u, weights)
    
    return G
