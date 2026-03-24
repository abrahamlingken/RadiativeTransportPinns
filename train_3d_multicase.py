#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_3d_multicase.py - 3D RTE 多工况自动训练流水线 (OOP优雅版)

特性：
1. 纯面向对象设计，无猴子补丁
2. 真正的梯度累加Chunking，不OOM
3. 无torch.cuda.empty_cache()拖慢
4. 支持多工况自动切换

案例：
    - Case 3D_A (纯吸收): kappa=1.0, sigma_s=0.0, g=0.0
    - Case 3D_B (各向同性散射): kappa=0.5, sigma_s=0.5, g=0.0
    - Case 3D_C (前向散射): kappa=0.1, sigma_s=0.9, g=0.6
"""

import sys
import os
import json
import time
import math
import argparse

# ========================================================================
# 路径设置（必须在其他导入之前）
# ========================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CORE_PATH = os.path.join(PROJECT_ROOT, 'Core')
if CORE_PATH not in sys.path:
    sys.path.insert(0, CORE_PATH)

# 现在可以安全导入项目模块
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import qmc

# 从Core导入
from ModelClassTorch2 import Pinns, init_xavier

# ========================================================================
# 案例配置
# ========================================================================

CASE_CONFIGS = {
    '3D_A': {
        'name': 'Case3D_A_PureAbsorption',
        'description': 'Pure absorption (kappa=1.0, sigma_s=0.0, g=0.0)',
        'kappa': 1.0,
        'sigma_s': 0.0,
        'g': 0.0,
        'folder': 'Results_3D_CaseA'
    },
    '3D_B': {
        'name': 'Case3D_B_Isotropic',
        'description': 'Isotropic scattering (kappa=0.5, sigma_s=0.5, g=0.0)',
        'kappa': 0.5,
        'sigma_s': 0.5,
        'g': 0.0,
        'folder': 'Results_3D_CaseB'
    },
    '3D_C': {
        'name': 'Case3D_C_ForwardScattering',
        'description': 'Forward scattering (kappa=0.1, sigma_s=0.9, g=0.6)',
        'kappa': 0.1,
        'sigma_s': 0.9,
        'g': 0.6,
        'folder': 'Results_3D_CaseC'
    }
}


# ========================================================================
# 训练函数
# ========================================================================

def train_single_case(case_key, chunk_size=4096):
    """
    训练单个3D案例
    
    内存管理策略：
    - 使用物理引擎实例化，无全局状态
    - 真正的梯度累加：每个chunk立即backward，不保留计算图
    - 无empty_cache()拖慢
    """
    from EquationModels.RadTrans3D_Complex import RadTrans3D_Physics
    
    config = CASE_CONFIGS[case_key]
    folder_path = os.path.join(PROJECT_ROOT, config['folder'])
    
    print(f"\n{'='*70}")
    print(f"Training {case_key}: {config['name']}")
    print(f"{'='*70}")
    print(f"  Description: {config['description']}")
    print(f"  Output folder: {folder_path}")
    
    os.makedirs(folder_path, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(folder_path, "case_config.json"), 'w') as f:
        json.dump({
            'case': case_key,
            'kappa': config['kappa'],
            'sigma_s': config['sigma_s'],
            'g': config['g'],
            'chunk_size': chunk_size
        }, f, indent=2)
    
    # ========================================================================
    # 步骤1：创建物理引擎（OOP核心！）
    # ========================================================================
    print("\n[1] Creating physics engine...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 实例化物理引擎，所有参数封装在此
    physics = RadTrans3D_Physics(
        kappa_val=config['kappa'],
        sigma_s_val=config['sigma_s'],
        g_val=config['g'],
        n_theta=8,
        n_phi=16,
        dev=device
    )
    
    # ========================================================================
    # 步骤2：生成训练数据
    # ========================================================================
    print("\n[2] Generating training points...")
    
    N_COLL = 16384
    N_U = 12288
    SAMPLING_SEED = 32
    
    torch.manual_seed(SAMPLING_SEED)
    
    # 使用物理引擎的方法生成点
    x_coll, y_coll = physics.generate_collocation_points(N_COLL, SAMPLING_SEED)
    x_b, y_b = physics.generate_boundary_points(N_U, 16)
    
    print(f"  Collocation: {x_coll.shape[0]}")
    print(f"  Boundary: {x_b.shape[0]}")
    
    # ========================================================================
    # 步骤3：创建网络模型
    # ========================================================================
    print("\n[3] Creating neural network...")
    
    NETWORK_PROPERTIES = {
        "hidden_layers": 8,
        "neurons": 128,
        "residual_parameter": 0.1,
        "kernel_regularizer": 2,
        "regularization_parameter": 0,
        "batch_size": N_COLL + N_U,
        "epochs": 1,
        "activation": "tanh"
    }
    
    input_dims = 5  # x, y, z, theta, phi
    output_dims = 1
    
    model = Pinns(input_dims, output_dims, NETWORK_PROPERTIES)
    
    torch.manual_seed(42)
    init_xavier(model)
    
    if torch.cuda.is_available():
        model = model.cuda()
        x_coll = x_coll.cuda()
        x_b = x_b.cuda()
        y_b = y_b.cuda()
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  Using CPU")
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # ========================================================================
    # 步骤4：配置优化器
    # ========================================================================
    print("\n[4] Configuring optimizer...")
    
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=50000,
        max_eval=50000,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe"
    )
    
    # ========================================================================
    # 步骤5：训练循环（真正的梯度累加！）
    # ========================================================================
    print(f"\n[5] Training with chunk_size={chunk_size}...")
    print("  Note: Each chunk computes loss, backward(), then releases graph immediately")
    
    training_history = {'epochs': [], 'total_loss': [], 'time': []}
    start_time = time.time()
    iteration = 0
    
    def closure():
        """
        L-BFGS closure函数
        
        真正的内存安全策略：
        1. zero_grad() 清除旧梯度
        2. 遍历每个chunk：
           - 计算该chunk的loss
           - 立即backward()（梯度累加到param.grad）
           - 计算图立即释放（不保留）
        3. 返回总loss（标量，无计算图）
        """
        nonlocal iteration
        
        # 1. 清除旧梯度
        optimizer.zero_grad()
        
        total_loss = 0.0
        n_chunks = 0
        
        # ========================================================================
        # 边界损失（通常数据量小，不需要chunking）
        # ========================================================================
        u_pred_b, u_train_b = physics.apply_bc(x_b, y_b, model)
        
        if u_pred_b.shape[0] > 0:
            loss_b = torch.mean((u_pred_b - u_train_b) ** 2)
            # 边界损失直接backward（通常不大）
            loss_b.backward()
            total_loss += loss_b.item()
        
        # ========================================================================
        # 残差损失（大batch，使用真正的梯度累加）
        # ========================================================================
        N_coll = x_coll.shape[0]
        lambda_res = model.lambda_residual
        
        # 分块处理，每个chunk立即backward
        for i in range(0, N_coll, chunk_size):
            end_i = min(i + chunk_size, N_coll)
            x_chunk = x_coll[i:end_i]
            
            # 计算该chunk的残差
            res_chunk = physics.compute_res(model, x_chunk)
            
            # 该chunk的loss
            loss_f_chunk = torch.mean(res_chunk ** 2)
            
            # 关键：立即backward，梯度累加到param.grad
            # 这是真正的梯度累加！backward后不保留计算图
            loss_f_chunk.backward()
            
            # 累加标量loss（detach，无计算图）
            total_loss += lambda_res * loss_f_chunk.item()
            n_chunks += 1
            
            # 不调用torch.cuda.empty_cache()！
            # PyTorch会自动管理显存，频繁调用反而拖慢
        
        # 返回总loss（标量，无计算图，仅用于L-BFGS监控）
        iteration += 1
        
        # 记录历史
        if iteration % 10 == 0:
            training_history['epochs'].append(iteration)
            training_history['total_loss'].append(total_loss)
            training_history['time'].append(time.time() - start_time)
            
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Iter {iteration:5d} | Loss: {total_loss:.6e} | Time: {elapsed:.1f}s | Chunks: {n_chunks}")
        
        # 返回标量（L-BFGS需要）
        return torch.tensor(total_loss, device=device, requires_grad=False)
    
    # 执行训练
    try:
        optimizer.step(closure)
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
    
    # ========================================================================
    # 步骤6：保存结果
    # ========================================================================
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
        'final_loss': final_loss,
        'iterations': iteration,
        'training_time': training_time,
        'chunk_size': chunk_size,
        'device': str(device),
        'n_collocation': N_COLL,
        'n_boundary': N_U
    }
    with open(os.path.join(folder_path, "training_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    # 保存训练曲线
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
    
    return {
        'case': case_key,
        'final_loss': final_loss,
        'iterations': iteration,
        'time': training_time
    }


# ========================================================================
# 主程序
# ========================================================================

def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='3D RTE Multi-Case Training (OOP Version)')
    parser.add_argument('--case', type=str, default='all',
                       choices=['all', '3D_A', '3D_B', '3D_C'],
                       help='Which case to run (default: all)')
    parser.add_argument('--chunk-size', type=int, default=4096,
                       help='Chunk size for gradient accumulation (default: 4096)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("3D RTE Multi-Case Training Pipeline (OOP Version)")
    print("="*70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Cases: {args.case}")
    
    # 确定运行的案例
    if args.case == 'all':
        cases = ['3D_A', '3D_B', '3D_C']
    else:
        cases = [args.case]
    
    # 运行所有案例
    results = []
    for case_key in cases:
        try:
            result = train_single_case(case_key, chunk_size=args.chunk_size)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Case {case_key} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'case': case_key,
                'error': str(e)
            })
    
    # 汇总
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    print(f"{'Case':<10} {'Final Loss':<15} {'Iterations':<12} {'Time (s)':<10}")
    print("-"*70)
    
    for r in results:
        loss_str = f"{r['final_loss']:.6e}" if 'final_loss' in r else "FAILED"
        iter_str = str(r.get('iterations', 'N/A'))
        time_str = f"{r.get('time', 0):.1f}"
        print(f"{r['case']:<10} {loss_str:<15} {iter_str:<12} {time_str:<10}")
    
    # 保存汇总
    with open('training_summary_3d.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummary saved to: training_summary_3d.json")
    print("="*70)


if __name__ == "__main__":
    main()
