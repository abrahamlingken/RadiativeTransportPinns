#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_3d_multicase.py - 3D RTE Multi-case Training Pipeline

Features:
1. Pure OOP design
2. True gradient accumulation (Chunking)
3. No torch.cuda.empty_cache() slowdown
4. Multi-case automatic switching

Cases:
    - Case 3D_A (Pure Absorption): kappa=5.0, sigma_s=0.0, g=0.0
    - Case 3D_B (Isotropic Scattering): kappa=0.5, sigma_s=4.5, g=0.0
    - Case 3D_C (Forward Scattering): kappa=0.5, sigma_s=4.5, g=0.8
"""

import sys
import os
import json
import time
import math
import argparse

# ========================================================================
# Path setup (must be before other imports)
# ========================================================================

# Calculate PROJECT_ROOT (we are now in Training/ subdirectory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

def train_single_case(case_key, chunk_size=4096):
    """
    训练单个3D案例
    
    Memory management strategy:
    - Use physics engine instance, no global state
    - True gradient accumulation: backward immediately per chunk
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
    
    # Instantiate physics engine
    physics = RadTrans3D_Physics(
        kappa_val=config['kappa'],
        sigma_s_val=config['sigma_s'],
        g_val=config['g'],
        n_theta=8,
        n_phi=16,
        dev=device
    )
    
    # ========================================================================
    # Step 2: Generate training data
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
    # Step 3: Create network model
    # ========================================================================
    print("\n[3] Creating neural network...")
    
    # Enhanced network config: deeper, wider, Swish activation
    print("  Using enhanced network: 10 layers x 256 neurons, Swish activation")
    NETWORK_PROPERTIES = {
        "hidden_layers": 10,        # Deeper network (was 8)
        "neurons": 256,             # Wider layer (was 128)
        "residual_parameter": 0.1,
        "kernel_regularizer": 2,
        "regularization_parameter": 0,
        "batch_size": N_COLL + N_U,
        "epochs": 1,
        "activation": "swish"       # Swish activation (was tanh)
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
    
    # Case C uses relaxed convergence (high scattering)
    if case_key == '3D_C':
        lr = 0.25  # Lower learning rate
        tol_grad = 1e-8  # Relaxed gradient tolerance
        tol_change = 1e-10
        print("  [Case C] Using adjusted LBFGS parameters for high-scattering regime")
    else:
        lr = 0.5
        tol_grad = 1e-7
        tol_change = 1e-9
    
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=100000,  # Increase max iterations
        max_eval=100000,
        tolerance_grad=tol_grad,
        tolerance_change=tol_change,
        history_size=150,
        line_search_fn="strong_wolfe"
    )
    
    # ========================================================================
    # 步骤5：训练循环（Adam预热 + L-BFGS精调，含权重退火）
    # ========================================================================
    print(f"\n[5] Training with chunk_size={chunk_size}...")
    print("  Strategy: Adam warm-up (2000 iters, lambda=0.1) -> L-BFGS fine-tune (lambda=1.0)")
    
    training_history = {'epochs': [], 'total_loss': [], 'time': []}
    start_time = time.time()
    iteration = 0
    annealing_printed = False  # Flag for weight switch notification
    
    def compute_losses_and_backward(optimizer, lambda_residual):
        """
        统一的损失计算与反向传播函数
        Support both Adam and L-BFGS optimizers
        """
        nonlocal iteration, annealing_printed
        
        optimizer.zero_grad()
        total_loss = 0.0
        
        # ========== 1. 计算边界损失 ==========
        u_pred_b, u_train_b = physics.apply_bc(x_b, y_b, model)
        if u_pred_b.numel() > 0:
            loss_b = torch.mean((u_pred_b - u_train_b)**2)
            loss_b.backward()
            total_loss += loss_b.item()
            del u_pred_b, u_train_b, loss_b
        
        # ========== 2. 计算残差损失 (Chunking梯度累加) ==========
        N_coll = x_coll.shape[0]
        n_chunks = (N_coll + chunk_size - 1) // chunk_size
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, N_coll)
            x_chunk = x_coll[start_idx:end_idx]
            
            # 计算当前块的 PDE 残差
            res_chunk = physics.compute_res(model, x_chunk)
            
            # Weight to match gradient scaling
            weight = (end_idx - start_idx) / N_coll
            loss_chunk = lambda_residual * torch.mean(res_chunk**2) * weight
            
            # Backward immediately and free graph
            loss_chunk.backward()
            total_loss += loss_chunk.item()
            
            # 手动销毁大张量引用
            del res_chunk, loss_chunk, x_chunk
        
        # ========== 3. 日志记录与权重退火通知 ==========
        if iteration % 10 == 0:
            training_history['epochs'].append(iteration)
            training_history['total_loss'].append(total_loss)
            training_history['time'].append(time.time() - start_time)
        
        if iteration % 100 == 0 or (iteration == 2000 and not annealing_printed):
            elapsed = time.time() - start_time
            print(f"  Iter {iteration:5d} | Loss: {total_loss:.6e} | lambda: {lambda_residual:.1f} | Time: {elapsed:.1f}s")
            if torch.cuda.is_available():
                print(f"          GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 权重退火切换通知（仅打印一次）
        if iteration == 2000 and not annealing_printed:
            print("\n" + "="*70)
            print("[Weight Annealing] lambda_residual increased from 0.1 to 1.0!")
            print("                  Switching from Adam to L-BFGS optimizer...")
            print("="*70 + "\n")
            annealing_printed = True
        
        iteration += 1
        
        # 返回总损失（L-BFGS需要Tensor，Adam需要标量）
        if isinstance(optimizer, optim.LBFGS):
            return torch.tensor(total_loss, device=device, requires_grad=True)
        else:
            return total_loss
    
    # =========================================================================
    # 阶段1: Adam预热 (迭代 0 ~ 1999)
    # 目标：优先学习边界条件，权重lambda_residual=0.1
    # =========================================================================
    print("\n[Phase 1] Adam warm-up: boundary focus (lambda_residual=0.1)")
    
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    lambda_phase1 = 0.1
    
    for iter_adam in range(2000):
        loss_val = compute_losses_and_backward(optimizer_adam, lambda_phase1)
        optimizer_adam.step()
        
        # Optional: early stop check
        if iter_adam > 500 and len(training_history['total_loss']) > 10:
            recent_losses = training_history['total_loss'][-10:]
            if max(recent_losses) - min(recent_losses) < 1e-6:
                print(f"  [Early stop] Loss stagnated at iter {iter_adam}, switching to L-BFGS early")
                break
    
    # =========================================================================
    # 阶段2: L-BFGS精调 (迭代 >= 2000)
    # Goal: strengthen PDE constraint
    # =========================================================================
    print("\n[Phase 2] L-BFGS fine-tuning: PDE focus (lambda_residual=1.0)")
    
    lambda_phase2 = 1.0  # 进入L-BFGS后权重必须绝对恒定！
    
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=100000,
        max_eval=100000,
        tolerance_grad=tol_grad,
        tolerance_change=tol_change,
        history_size=150,
        line_search_fn="strong_wolfe"
    )
    
    # L-BFGS使用closure机制
    def lbfgs_closure():
        return compute_losses_and_backward(optimizer_lbfgs, lambda_phase2)
    
    # 执行L-BFGS训练
    try:
        optimizer_lbfgs.step(lbfgs_closure)
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
    
    # ========================================================================
    # Step 6: Save results
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
# Main function
# ========================================================================

def main():
    """Main function"""
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
    
    # Determine cases to run
    if args.case == 'all':
        cases = ['3D_A', '3D_B', '3D_C']
    else:
        cases = [args.case]
    
    # Run all cases
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
    
    # Summary
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
    
    # 保存Summary
    with open('training_summary_3d.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummary saved to: training_summary_3d.json")
    print("="*70)


if __name__ == "__main__":
    main()
