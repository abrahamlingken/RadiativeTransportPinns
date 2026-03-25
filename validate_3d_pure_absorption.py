import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ==========================================
# 0. 关键修复：配置正确的模块搜索路径
# ==========================================
# 获取当前脚本所在的项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.join(PROJECT_ROOT, 'Core')

# 将根目录和 Core 目录加入系统路径
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CORE_PATH not in sys.path:
    sys.path.insert(0, CORE_PATH)

# 必须在这里导入模型类，否则 torch.load 无法反序列化 (Unpickle) 模型
import ModelClassTorch2
from ModelClassTorch2 import Pinns 

# 设置顶刊风格字体
try:
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 14
    rcParams['legend.fontsize'] = 11
except:
    pass 

# ==========================================
# 1. 物理参数与精确解计算器
# ==========================================
KAPPA = 1.0
CENTER = np.array([0.5, 0.5, 0.5])

def source_term(x, y, z):
    """匹配源项: max(0, 1.0 - 2.0*r)"""
    r = np.sqrt((x - CENTER[0])**2 + (y - CENTER[1])**2 + (z - CENTER[2])**2)
    return np.maximum(0.0, 1.0 - 2.0 * r)

def compute_exact_intensity(x, y, z, s_vec, num_points=100):
    """反向射线追踪 (Ray tracing backward) 计算解析解"""
    pos = np.array([x, y, z])
    s_vec = np.array(s_vec)
    s_vec = s_vec / np.linalg.norm(s_vec)
    
    t_bounds = []
    for i in range(3):
        if s_vec[i] > 1e-8:
            t_bounds.append(pos[i] / s_vec[i])     
        elif s_vec[i] < -1e-8:
            t_bounds.append((pos[i] - 1.0) / s_vec[i]) 
    
    L = min(t_bounds) if t_bounds else 0.0
    
    l_vals = np.linspace(0, L, num_points)
    integrand = np.zeros_like(l_vals)
    
    for i, l in enumerate(l_vals):
        curr_pos = pos - l * s_vec
        S_val = source_term(curr_pos[0], curr_pos[1], curr_pos[2])
        integrand[i] = KAPPA * S_val * np.exp(-KAPPA * l)
        
    try:
        intensity = np.trapezoid(integrand, l_vals)
    except AttributeError:
        intensity = np.trapz(integrand, l_vals) 
    return intensity

# ==========================================
# 2. 从 PINN 提取预测解
# ==========================================
def load_model_and_predict(model_path, x_coords, y_coords, z_coords, theta, phi):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 此时上下文中已经有了 ModelClassTorch2，可以安全执行 torch.load 了
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    
    N = x_coords.size
    inputs = np.zeros((N, 5), dtype=np.float32)
    inputs[:, 0] = x_coords.flatten()
    inputs[:, 1] = y_coords.flatten()
    inputs[:, 2] = z_coords.flatten()
    inputs[:, 3] = theta
    inputs[:, 4] = phi
    
    inputs_tensor = torch.tensor(inputs).to(device)
    
    with torch.no_grad():
        u_pred = model(inputs_tensor).cpu().numpy().reshape(x_coords.shape)
        
    return u_pred

# ==========================================
# 3. 顶会/顶刊绘图模块
# ==========================================
def plot_results(model_path="Results_3D_CaseA/model.pkl"):
    print("开始生成对比图像...")
    
    # 设定观察方向: 沿着 x 轴正向 (theta=pi/2, phi=0)
    theta_view = np.pi / 2
    phi_view = 0.0
    s_vec = [np.sin(theta_view)*np.cos(phi_view), 
             np.sin(theta_view)*np.sin(phi_view), 
             np.cos(theta_view)]
             
    # ---------------------------------------------------------
    # 图1：沿中心线的 1D 对比
    # ---------------------------------------------------------
    print("  正在生成图1 (1D 中心线)...")
    x_line = np.linspace(0, 1, 100)
    y_line = np.full_like(x_line, 0.5)
    z_line = np.full_like(x_line, 0.5)
    
    exact_line = [compute_exact_intensity(x, 0.5, 0.5, s_vec) for x in x_line]
    
    if os.path.exists(model_path):
        pred_line = load_model_and_predict(model_path, x_line, y_line, z_line, theta_view, phi_view)
    else:
        print(f"  [错误] 找不到模型文件 {model_path}，请检查路径。")
        return

    fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
    ax1.plot(x_line, exact_line, 'k-', linewidth=2.5, label='Exact Analytical')
    ax1.plot(x_line, pred_line, color='#D62728', linestyle='--', linewidth=2.0, 
            marker='o', markersize=6, markerfacecolor='none', markevery=5, 
            label='PINN Prediction', zorder=3)
    ax1.set_xlabel(r'Spatial position $x$ ($y=0.5, z=0.5$)')
    ax1.set_ylabel(r'Intensity $I(x, \mathbf{s})$')
    ax1.set_title(r'3D Pure Absorption: Centerline Comparison')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.savefig('3D_Pure_Absorption_1D_Line.pdf', bbox_inches='tight')
    plt.close(fig1)

    # ---------------------------------------------------------
    # 图2：中心截面 (z=0.5) 的 2D 等高线对比与误差图
    # ---------------------------------------------------------
    print("  正在生成图2 (2D z=0.5 截面)...这需要十几秒计算解析解，请稍候。")
    N_grid = 100
    x_grid = np.linspace(0, 1, N_grid)
    y_grid = np.linspace(0, 1, N_grid)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.full_like(X, 0.5)

    # 计算 2D 精确解
    Exact_2D = np.zeros_like(X)
    for i in range(N_grid):
        for j in range(N_grid):
            Exact_2D[i, j] = compute_exact_intensity(X[i, j], Y[i, j], 0.5, s_vec)

    # 获取 2D 预测解
    Pred_2D = load_model_and_predict(model_path, X, Y, Z, theta_view, phi_view)

    # 计算绝对误差
    Error_2D = np.abs(Exact_2D - Pred_2D)

    # 绘制 1x3 的对比图
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    vmin = min(Exact_2D.min(), Pred_2D.min())
    vmax = max(Exact_2D.max(), Pred_2D.max())
    levels = np.linspace(vmin, vmax, 50)
    
    cf1 = axes[0].contourf(X, Y, Exact_2D, levels=levels, cmap='jet')
    axes[0].set_title('(a) Exact Solution $I(x,y,z=0.5)$')
    axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')
    fig2.colorbar(cf1, ax=axes[0], fraction=0.046, pad=0.04)
    
    cf2 = axes[1].contourf(X, Y, Pred_2D, levels=levels, cmap='jet')
    axes[1].set_title('(b) PINN Prediction $I(x,y,z=0.5)$')
    axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$y$')
    fig2.colorbar(cf2, ax=axes[1], fraction=0.046, pad=0.04)
    
    error_levels = np.linspace(0, Error_2D.max(), 50)
    cf3 = axes[2].contourf(X, Y, Error_2D, levels=error_levels, cmap='magma')
    axes[2].set_title(f'(c) Absolute Error (Max: {Error_2D.max():.2e})')
    axes[2].set_xlabel('$x$'); axes[2].set_ylabel('$y$')
    fig2.colorbar(cf3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('3D_Pure_Absorption_2D_Contours.pdf', bbox_inches='tight')
    plt.close(fig2)
    
    print("全部图像生成完毕！")
    print(" -> 1D曲线图已保存为 3D_Pure_Absorption_1D_Line.pdf")
    print(" -> 2D截面图已保存为 3D_Pure_Absorption_2D_Contours.pdf")

if __name__ == "__main__":
    plot_results("Results_3D_CaseA/model.pkl")