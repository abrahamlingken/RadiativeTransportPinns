#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_3d_multicase.py - 3D RTE 多工况自动训练流水线

支持三个测试用例：
    - Case 3D_A (纯吸收): kappa=1.0, sigma_s=0.0, g=0.0
    - Case 3D_B (各向同性散射): kappa=0.5, sigma_s=0.5, g=0.0
    - Case 3D_C (前向散射): kappa=0.1, sigma_s=0.9, g=0.6

使用 Chunking 机制避免 OOM，支持 24GB 显存 GPU
"""

import sys
import os

# 添加项目根目录到路径（必须在其他导入之前）
# 获取项目根目录（脚本所在目录）
if '__file__' in dir():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
else:
    PROJECT_ROOT = os.getcwd()

# 确保项目根目录在路径中
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 添加Core目录到路径
CORE_PATH = os.path.join(PROJECT_ROOT, 'Core')
if CORE_PATH not in sys.path:
    sys.path.insert(0, CORE_PATH)

import json
import time
import math
import importlib

# ============================================
# 3D 案例配置
# ============================================

CASE_CONFIGS = {
    '3D_A': {
        'name': 'Case3D_A_PureAbsorption',
        'description': 'Pure absorption (kappa=1.0, sigma_s=0.0, g=0.0)',
        'kappa': 1.0,
        'sigma_s': 0.0,
        'g': 0.0,
        'folder': os.path.join(PROJECT_ROOT, 'Results_3D_CaseA')
    },
    '3D_B': {
        'name': 'Case3D_B_Isotropic',
        'description': 'Isotropic scattering (kappa=0.5, sigma_s=0.5, g=0.0)',
        'kappa': 0.5,
        'sigma_s': 0.5,
        'g': 0.0,
        'folder': os.path.join(PROJECT_ROOT, 'Results_3D_CaseB')
    },
    '3D_C': {
        'name': 'Case3D_C_ForwardScattering',
        'description': 'Forward scattering (kappa=0.1, sigma_s=0.9, g=0.6)',
        'kappa': 0.1,
        'sigma_s': 0.9,
        'g': 0.6,
        'folder': os.path.join(PROJECT_ROOT, 'Results_3D_CaseC')
    }
}


# ============================================
# 动态修改物理参数
# ============================================

def set_3d_physics_params(kappa_val, sigma_val, g_val):
    """
    动态修改 RadTrans3D_Complex 模块的物理参数
    
    在导入模块前调用无效，需在训练循环中重新导入或确保已导入后修改
    """
    # 导入模块
    import EquationModels.RadTrans3D_Complex as RadTrans
    
    # 修改全局参数
    RadTrans.G_HG = g_val
    
    # 重新定义物理函数以使用新参数
    def kappa_new(x, y, z):
        return torch.ones_like(x) * kappa_val
    
    def sigma_s_new(x, y, z):
        return torch.ones_like(x) * sigma_val
    
    # 替换模块中的函数
    RadTrans.kappa = kappa_new
    RadTrans.sigma_s = sigma_s_new
    
    # 强制重新加载确保参数生效
    importlib.reload(RadTrans)
    
    # 再次设置（因为reload会重置）
    RadTrans.G_HG = g_val
    RadTrans.kappa = kappa_new
    RadTrans.sigma_s = sigma_s_new
    
    print(f"  [Physics] kappa={kappa_val}, sigma_s={sigma_val}, g={g_val}")
    return RadTrans


# ============================================
# Chunking Loss 实现
# ============================================

def create_custom_loss(Ec, chunk_size=4096):
    """
    创建带Chunking机制的CustomLoss
    
    Args:
        Ec: 方程模块
        chunk_size: 分块大小，24GB显存建议4096
    """
    import torch
    import torch.nn as nn
    
    class CustomLoss(nn.Module):
        def __init__(self, chunk_size=4096):
            super(CustomLoss, self).__init__()
            self.chunk_size = chunk_size
        
        def forward(self, network, x_u_train, u_train, x_b_train, u_b_train,
                    x_f_train, x_obj, u_obj, dataclass, training_ic, computing_error=False):
            
            lambda_residual = network.lambda_residual
            space_dimensions = dataclass.space_dimensions
            
            # ========== 边界损失 ==========
            u_pred_var_list = []
            u_train_var_list = []
            
            for j in range(dataclass.output_dimension):
                u_pred_b, u_train_b = Ec.apply_BC(x_b_train, u_b_train, network)
                u_pred_var_list.append(u_pred_b)
                u_train_var_list.append(u_train_b)
                
                if Ec.time_dimensions != 0:
                    from EquationModels.RadTrans3D_Complex import apply_IC
                    u_pred_0, u_train_0 = apply_IC(x_u_train, u_train, network)
                    u_pred_var_list.append(u_pred_0)
                    u_train_var_list.append(u_train_0)
            
            loss_b = torch.mean(torch.stack([
                torch.mean((u_pred_var_list[i] - u_train_var_list[i])**2)
                for i in range(len(u_pred_var_list))
            ]))
            
            # ========== 残差损失 (带Chunking) ==========
            N_coll = x_f_train.shape[0]
            
            if N_coll <= self.chunk_size:
                # 小batch直接计算
                res = Ec.compute_res(network, x_f_train, space_dimensions, None, computing_error)
                loss_f = torch.mean(res**2)
            else:
                # 大batch分块计算
                n_chunks = (N_coll + self.chunk_size - 1) // self.chunk_size
                loss_f_sum = 0.0
                
                for i in range(n_chunks):
                    start_idx = i * self.chunk_size
                    end_idx = min((i + 1) * self.chunk_size, N_coll)
                    x_chunk = x_f_train[start_idx:end_idx]
                    
                    res_chunk = Ec.compute_res(network, x_chunk, space_dimensions, None, computing_error)
                    loss_f_sum = loss_f_sum + torch.mean(res_chunk**2)
                    
                    # 显式清理显存
                    del res_chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                loss_f = loss_f_sum / n_chunks
            
            loss = lambda_residual * loss_f + loss_b
            return loss
    
    return CustomLoss(chunk_size=chunk_size)


# ============================================
# 单个案例训练
# ============================================

def train_single_case(case_key, chunk_size=4096):
    """
    训练单个3D案例
    
    Args:
        case_key: '3D_A', '3D_B', 或 '3D_C'
        chunk_size: 残差计算分块大小
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    
    config = CASE_CONFIGS[case_key]
    folder_path = config['folder']
    
    print(f"\n{'='*70}")
    print(f"Training {case_key}: {config['name']}")
    print(f"{'='*70}")
    print(f"  Description: {config['description']}")
    print(f"  Output folder: {folder_path}")
    
    # 创建输出目录
    os.makedirs(folder_path, exist_ok=True)
    
    # ========== 步骤 1: 清除模块缓存 ==========
    for mod_name in list(sys.modules.keys()):
        if 'EquationModels' in mod_name or 'ModelClassTorch2' in mod_name:
            del sys.modules[mod_name]
    
    # ========== 步骤 2: 设置物理参数并导入模块 ==========
    print("\n[1] Setting up physics parameters...")
    
    # 先导入模块
    import EquationModels.RadTrans3D_Complex as Ec
    
    # 修改物理参数
    Ec.G_HG = config['g']
    
    # 重新定义物理函数
    kappa_val = config['kappa']
    sigma_val = config['sigma_s']
    
    def kappa_func(x, y, z):
        return torch.ones_like(x) * kappa_val
    
    def sigma_func(x, y, z):
        return torch.ones_like(x) * sigma_val
    
    Ec.kappa = kappa_func
    Ec.sigma_s = sigma_func
    
    print(f"  kappa = {kappa_val}")
    print(f"  sigma_s = {sigma_val}")
    print(f"  g (HG) = {config['g']}")
    
    # ========== 步骤 3: 创建 FakeImportFile ==========
    class FakeImportFile:
        pass
    
    fake_import = FakeImportFile()
    
    # 复制 Ec 的所有属性
    for attr in dir(Ec):
        if not attr.startswith('_'):
            setattr(fake_import, attr, getattr(Ec, attr))
    
    # 添加必要的模块（ModelClassTorch2 依赖这些）
    fake_import.torch = torch
    fake_import.nn = nn
    fake_import.optim = optim
    fake_import.np = np
    fake_import.json = json
    fake_import.time = time
    fake_import.math = math
    fake_import.pi = math.pi
    
    # 添加 qmc（用于 Sobol 采样）
    try:
        from scipy.stats import qmc
        fake_import.qmc = qmc
    except ImportError:
        pass
    
    sys.modules['ImportFile'] = fake_import
    
    # ========== 步骤 4: 导入 ModelClassTorch2 ==========
    # 确保路径正确（使用全局 PROJECT_ROOT）
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    # 从 Core 目录导入
    from Core.DatasetTorch2 import DefineDataset
    from Core import ModelClassTorch2 as _mc
    
    _mc.Ec = Ec
    Pinns = _mc.Pinns
    init_xavier = _mc.init_xavier
    
    # ========== 步骤 5: 设置 CustomLoss (带Chunking) ==========
    CustomLoss = create_custom_loss(Ec, chunk_size=chunk_size)
    _mc.CustomLoss = CustomLoss
    
    # ========== 步骤 6: 配置参数 ==========
    SAMPLING_SEED = 32
    N_COLL = 16384      # 配点数
    N_U = 12288         # 边界点数
    N_INT = 0
    RETRAIN = 32
    
    NETWORK_PROPERTIES = {
        "hidden_layers": 8,
        "neurons": 128,
        "residual_parameter": 0.1,
        "kernel_regularizer": 2,
        "regularization_parameter": 0,
        "batch_size": (N_COLL + N_U + N_INT),
        "epochs": 1,
        "activation": "tanh"
    }
    
    # 保存配置
    with open(os.path.join(folder_path, "case_config.json"), 'w') as f:
        json.dump({
            'case': case_key,
            'kappa': kappa_val,
            'sigma_s': sigma_val,
            'g': config['g'],
            'N_coll': N_COLL,
            'N_boundary': N_U,
            'chunk_size': chunk_size,
            'network': NETWORK_PROPERTIES
        }, f, indent=2)
    
    # ========== 步骤 7: 获取维度信息 ==========
    space_dimensions = Ec.space_dimensions
    time_dimension = Ec.time_dimensions
    
    if hasattr(Ec, 'parameters_values'):
        parameter_dimensions = Ec.parameters_values.shape[0]
    else:
        parameter_dimensions = 2  # theta, phi
    
    input_dimensions = parameter_dimensions + time_dimension + space_dimensions
    output_dimension = Ec.output_dimension
    
    print(f"\n[2] Network configuration:")
    print(f"  Input dimensions: {input_dimensions}")
    print(f"  Space: {space_dimensions}, Time: {time_dimension}, Param: {parameter_dimensions}")
    print(f"  Chunk size: {chunk_size}")
    
    # ========== 步骤 8: 生成训练数据 ==========
    print(f"\n[3] Generating training points...")
    torch.manual_seed(SAMPLING_SEED)
    
    x_coll, y_coll = Ec.add_collocations(N_COLL)
    x_b, y_b = Ec.add_boundary(N_U)
    
    print(f"  Collocation: {x_coll.shape[0]}")
    print(f"  Boundary: {x_b.shape[0]}")
    
    # ========== 步骤 9: 创建数据集 ==========
    class SimpleDataset:
        def __init__(self, x_coll, y_coll, x_b, y_b, batch_size,
                     space_dims=3, time_dims=0, param_dims=2, output_dim=1):
            self.x_coll = x_coll
            self.y_coll = y_coll
            self.x_b = x_b
            self.y_b = y_b
            self.batch_size = batch_size
            self.n_collocation = x_coll.shape[0]
            self.n_boundary = x_b.shape[0]
            self.space_dimensions = space_dims
            self.time_dimensions = time_dims
            self.parameter_dimensions = param_dims
            self.output_dimension = output_dim
            self.obj = None
            self.BC = None
            self.data_coll = [(x_coll, y_coll)]
            self.data_boundary = [(x_b, y_b)]
            self.data_initial_internal = []
    
    training_set_class = SimpleDataset(
        x_coll, y_coll, x_b, y_b, N_COLL + N_U,
        space_dims=space_dimensions,
        time_dims=time_dimension,
        param_dims=parameter_dimensions,
        output_dim=output_dimension
    )
    
    # ========== 步骤 10: 创建模型 ==========
    print(f"\n[4] Creating model...")
    model = Pinns(input_dimensions, output_dimension, NETWORK_PROPERTIES)
    
    torch.manual_seed(RETRAIN)
    init_xavier(model)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model = model.cuda()
        x_coll = x_coll.cuda()
        y_coll = y_coll.cuda()
        x_b = x_b.cuda()
        y_b = y_b.cuda()
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  Using CPU")
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # ========== 步骤 11: 训练 ==========
    print(f"\n[5] Training...")
    
    optimizer_LBFGS = optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=50000,
        max_eval=50000,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe"
    )
    
    training_history = {'epochs': [], 'total_loss': [], 'time': []}
    start_time = time.time()
    iteration = 0
    
    # 创建loss实例
    loss_fn = CustomLoss(chunk_size=chunk_size)
    
    def closure():
        nonlocal iteration
        
        optimizer_LBFGS.zero_grad()
        
        loss_val = loss_fn(
            model,
            torch.empty(0, input_dimensions, device=device),
            torch.empty(0, 1, device=device),
            x_b, y_b,
            x_coll,
            None, None,
            training_set_class, None
        )
        
        loss_val.backward()
        
        if iteration % 10 == 0:
            training_history['epochs'].append(iteration)
            training_history['total_loss'].append(float(loss_val.detach().cpu()))
            training_history['time'].append(time.time() - start_time)
            
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Iter {iteration:5d} | Loss: {loss_val.item():.6e} | Time: {elapsed:.1f}s")
                
                # 打印显存使用情况
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"          GPU Memory: {allocated:.2f} GB")
        
        iteration += 1
        return loss_val
    
    try:
        optimizer_LBFGS.step(closure)
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n  [ERROR] Out of memory! Try reducing chunk_size.")
            raise
        else:
            raise
    
    # ========== 步骤 12: 保存结果 ==========
    final_loss = training_history['total_loss'][-1] if training_history['total_loss'] else float('inf')
    training_time = time.time() - start_time
    
    print(f"\n[6] Saving results...")
    print(f"  Final loss: {final_loss:.6e}")
    print(f"  Iterations: {iteration}")
    print(f"  Time: {training_time:.1f}s")
    
    # 保存模型
    torch.save(model, os.path.join(folder_path, "model.pkl"))
    
    # 保存训练历史
    with open(os.path.join(folder_path, "training_history.json"), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 保存训练信息
    info = {
        'case': case_key,
        'N_coll': N_COLL,
        'N_sb': N_U,
        'network': NETWORK_PROPERTIES,
        'final_loss': final_loss,
        'iterations': iteration,
        'training_time': training_time,
        'chunk_size': chunk_size,
        'device': device
    }
    with open(os.path.join(folder_path, "training_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    # ========== 步骤 13: 生成训练曲线 ==========
    print("[7] Generating training curves...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        epochs = np.array(training_history['epochs'])
        loss = np.array(training_history['total_loss'])
        times = np.array(training_history['time'])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        
        axes[0, 0].plot(epochs, loss, 'b-', linewidth=1.5)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('(a) Linear Scale')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].semilogy(epochs, loss, 'b-', linewidth=1.5)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss (log)')
        axes[0, 1].set_title('(b) Log Scale')
        axes[0, 1].grid(True, alpha=0.3, which='both')
        
        axes[1, 0].semilogy(times, loss, 'g-', linewidth=1.5)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('(c) vs Time')
        axes[1, 0].grid(True, alpha=0.3, which='both')
        
        if len(loss) > 1:
            dlog = np.diff(np.log10(loss + 1e-10))
            axes[1, 1].plot(epochs[1:], dlog, 'r-', linewidth=1)
            axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('d(log L)/diter')
            axes[1, 1].set_title('(d) Convergence Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, 'training_curves.png'), dpi=300)
        print(f"  Saved: {folder_path}/training_curves.png")
        
    except Exception as e:
        print(f"  Error generating figures: {e}")
    
    print(f"\n[8] Case {case_key} completed!")
    print(f"  Results saved to: {folder_path}/")
    
    return {
        'case': case_key,
        'final_loss': final_loss,
        'iterations': iteration,
        'time': training_time
    }


# ============================================
# 主程序
# ============================================

def main():
    """主程序：运行所有3D案例"""
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description='3D RTE Multi-Case Training')
    parser.add_argument('--case', type=str, default='all',
                       choices=['all', '3D_A', '3D_B', '3D_C'],
                       help='Which case to run (default: all)')
    parser.add_argument('--chunk-size', type=int, default=4096,
                       help='Chunk size for residual computation (default: 4096)')
    parser.add_argument('--estimate-memory', action='store_true',
                       help='Estimate optimal chunk size based on GPU memory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("3D RTE Multi-Case Training Pipeline")
    print("="*70)
    
    # 估算显存
    if args.estimate_memory or torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        print(f"\nGPU Memory: {gpu_mem:.1f} GB")
        
        # 根据显存调整chunk_size
        if gpu_mem >= 24:
            recommended_chunk = 8192
        elif gpu_mem >= 16:
            recommended_chunk = 4096
        elif gpu_mem >= 8:
            recommended_chunk = 2048
        else:
            recommended_chunk = 1024
        
        print(f"Recommended chunk_size: {recommended_chunk}")
        
        if args.chunk_size == 4096:  # 用户未指定，使用推荐值
            args.chunk_size = recommended_chunk
            print(f"Using recommended chunk_size: {args.chunk_size}")
    
    print(f"\nChunk size: {args.chunk_size}")
    print(f"Cases to run: {args.case}")
    
    # 确定运行的案例
    if args.case == 'all':
        cases_to_run = ['3D_A', '3D_B', '3D_C']
    else:
        cases_to_run = [args.case]
    
    # 运行所有案例
    results = []
    for case_key in cases_to_run:
        try:
            result = train_single_case(case_key, chunk_size=args.chunk_size)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Case {case_key} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'case': case_key,
                'final_loss': None,
                'iterations': 0,
                'time': 0,
                'error': str(e)
            })
    
    # 汇总结果
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    print(f"{'Case':<10} {'Final Loss':<15} {'Iterations':<12} {'Time (s)':<10}")
    print("-"*70)
    
    for r in results:
        loss_str = f"{r['final_loss']:.6e}" if r['final_loss'] is not None else "FAILED"
        iter_str = str(r['iterations'])
        time_str = f"{r['time']:.1f}"
        print(f"{r['case']:<10} {loss_str:<15} {iter_str:<12} {time_str:<10}")
    
    # 保存汇总
    with open('training_summary_3d.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummary saved to: training_summary_3d.json")
    print("="*70)


if __name__ == "__main__":
    main()
