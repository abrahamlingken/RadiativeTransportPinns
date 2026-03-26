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
        'description': 'Pure absorption (kappa=5.0, sigma_s=0.0, g=0.0), beta=5.0',
        'kappa': 5.0,
        'sigma_s': 0.0,
        'g': 0.0,
        'folder': 'Results_3D_CaseA'
    },
    '3D_B': {
        'name': 'Case3D_B_Isotropic',
        'description': 'Isotropic scattering (kappa=0.5, sigma_s=4.5, g=0.0), beta=5.0',
        'kappa': 0.5,
        'sigma_s': 4.5,
        'g': 0.0,
        'folder': 'Results_3D_CaseB'
    },
    '3D_C': {
        'name': 'Case3D_C_ForwardScattering',
        'description': 'Forward scattering (kappa=0.5, sigma_s=4.5, g=0.8), beta=5.0',
        'kappa': 0.5,
        'sigma_s': 4.5,
        'g': 0.8,
        'folder': 'Results_3D_CaseC'
    }
}


# ========================================================================
# 训练函数
# ========================================================================

def curriculum_training_stage(model, physics_base, kappa_values, iterations_per_stage, 
                               x_coll, x_b, y_b, chunk_size, device):
    """
    课程学习：逐步增加kappa难度
    """
    from EquationModels.RadTrans3D_Complex import RadTrans3D_Physics
    
    print("\n  [Curriculum Training] Starting progressive kappa training...")
    
    for stage, kappa in enumerate(kappa_values):
        print(f"\n  Stage {stage+1}/{len(kappa_values)}: Training with kappa={kappa}")
        
        # 创建当前阶段的物理引擎
        physics_stage = RadTrans3D_Physics(
            kappa_val=kappa,
            sigma_s_val=physics_base.sigma_s_val,
            g_val=physics_base.g_val,
            n_theta=physics_base.n_theta,
            n_phi=physics_base.n_phi,
            dev=device
        )
        
        # 简单的Adam优化阶段
        optimizer = optim.Adam(model.parameters(), lr=1e-3 * (0.8 ** stage))
        
        for it in range(iterations_per_stage):
            optimizer.zero_grad()
            
            # 边界损失
            u_pred_b, u_train_b = physics_stage.apply_bc(x_b, y_b, model)
            if u_pred_b.numel() > 0:
                loss_b = torch.mean((u_pred_b - u_train_b)**2)
                loss_b.backward()
            
            # 残差损失（分块）
            N_coll = x_coll.shape[0]
            n_chunks = (N_coll + chunk_size - 1) // chunk_size
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, N_coll)
                x_chunk = x_coll[start_idx:end_idx]
                res_chunk = physics_stage.compute_res(model, x_chunk)
                weight = (end_idx - start_idx) / N_coll
                loss_chunk = 0.01 * torch.mean(res_chunk**2) * weight
                loss_chunk.backward()
                del res_chunk, loss_chunk, x_chunk
            
            optimizer.step()
            
            if it % 200 == 0:
                print(f"    Iter {it}/{iterations_per_stage} completed")
    
    print("  Curriculum training completed!")


def train_single_case(case_key, chunk_size=4096, use_curriculum=False):
    """
    训练单个3D案例
    
    内存管理策略：
    - 使用物理引擎实例化，无全局状态
    - 真正的梯度累加：每个chunk立即backward，不保留计算图
    - 无empty_cache()拖慢
    
    Args:
        use_curriculum: 是否使用课程学习（渐进增加kappa）
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
    
    # 优化网络配置：针对高kappa问题的锐利梯度
    if config['kappa'] >= 5.0:
        # 高衰减问题需要更深更宽的网络
        print("  [High kappa detected] Using enhanced network architecture")
        NETWORK_PROPERTIES = {
            "hidden_layers": 10,        # 更深的网络
            "neurons": 256,             # 更宽的网络
            "residual_parameter": 0.01,  # 降低残差权重，增加边界权重
            "kernel_regularizer": 2,
            "regularization_parameter": 0,
            "batch_size": N_COLL + N_U,
            "epochs": 1,
            "activation": "swish"       # Swish更适合陡峭梯度
        }
    else:
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
    
    # =======================================================================
    # 步骤3.5：课程学习预热（高kappa问题）
    # =======================================================================
    if use_curriculum and config['kappa'] >= 5.0:
        print("\n[3.5] Curriculum learning pre-training...")
        # 渐进增加kappa: 1.0 -> 2.0 -> 3.0 -> 5.0
        kappa_stages = [1.0, 2.0, 3.0, config['kappa']]
        curriculum_training_stage(
            model, physics, kappa_stages, 
            iterations_per_stage=500,  # 每个阶段500轮
            x_coll=x_coll, x_b=x_b, y_b=y_b,
            chunk_size=chunk_size, device=device
        )
        print("  Curriculum pre-training completed!")
    
    # ========================================================================
    # 步骤4：配置优化器
    # ========================================================================
    print("\n[4] Configuring optimizer...")
    
    # Case C使用更宽松的收敛条件（高散射问题更难收敛）
    if case_key == '3D_C':
        lr = 0.25  # 降低学习率
        tol_grad = 1e-8  # 更宽松的梯度容忍度
        tol_change = 1e-10
        print("  [Case C] Using adjusted LBFGS parameters for high-scattering regime")
    else:
        lr = 0.5
        tol_grad = 1e-7
        tol_change = 1e-9
    
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=100000,  # 增加最大迭代
        max_eval=100000,
        tolerance_grad=tol_grad,
        tolerance_change=tol_change,
        history_size=150,
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
        nonlocal iteration
        
        optimizer.zero_grad()
        total_loss = 0.0
        
        # ========== 1. 计算边界损失 (高kappa问题需要更强的边界约束) ==========
        u_pred_b, u_train_b = physics.apply_bc(x_b, y_b, model)
        if u_pred_b.numel() > 0:
            loss_b = torch.mean((u_pred_b - u_train_b)**2)
            
            # 自适应边界权重：高kappa问题边界更重要
            if config['kappa'] >= 5.0:
                lambda_bc = 10.0 + min(iteration / 500, 10.0)  # 10-20动态增加
            else:
                lambda_bc = 1.0
            
            loss_b_weighted = lambda_bc * loss_b
            loss_b_weighted.backward()  # 立即反向传播
            total_loss += loss_b.item()  # 记录原始损失
            del u_pred_b, u_train_b, loss_b, loss_b_weighted
            
        # ========== 2. 计算残差损失 (真正的 Chunking 梯度累加) ==========
        N_coll = x_coll.shape[0]
        n_chunks = (N_coll + chunk_size - 1) // chunk_size
        lambda_residual = NETWORK_PROPERTIES["residual_parameter"]
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, N_coll)
            x_chunk = x_coll[start_idx:end_idx]
            
            # 计算当前块的 PDE 残差
            res_chunk = physics.compute_res(model, x_chunk)
            
            # 乘以权重，保证与整体 mean() 算出来的梯度缩放比例一致
            weight = (end_idx - start_idx) / N_coll
            loss_chunk = lambda_residual * torch.mean(res_chunk**2) * weight
            
            # 极其关键：立即反向传播并销毁图，保护显存！
            loss_chunk.backward()
            total_loss += loss_chunk.item()
            
            # 手动销毁大张量引用
            del res_chunk, loss_chunk, x_chunk
            
        # ========== 3. 日志记录与进度输出 ==========
        if iteration % 10 == 0:
            training_history['epochs'].append(iteration)
            training_history['total_loss'].append(total_loss)
            training_history['time'].append(time.time() - start_time)
            
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Iter {iteration:5d} | Loss: {total_loss:.6e} | Time: {elapsed:.1f}s")
                if torch.cuda.is_available():
                    print(f"          GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    
        iteration += 1
        
        # L-BFGS 要求闭包必须返回包含 total loss 的 Tensor
        return torch.tensor(total_loss, device=device, requires_grad=True)
    
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
    parser.add_argument('--curriculum', action='store_true',
                       help='Use curriculum learning for high-kappa cases')
    
    args = parser.parse_args()
    
    print("="*70)
    print("3D RTE Multi-Case Training Pipeline (OOP Version)")
    print("="*70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Cases: {args.case}")
    print(f"Curriculum Learning: {'Enabled' if args.curriculum else 'Disabled'}")
    
    # 确定运行的案例
    if args.case == 'all':
        cases = ['3D_A', '3D_B', '3D_C']
    else:
        cases = [args.case]
    
    # 运行所有案例
    results = []
    for case_key in cases:
        try:
            result = train_single_case(case_key, chunk_size=args.chunk_size, 
                                       use_curriculum=args.curriculum)
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
