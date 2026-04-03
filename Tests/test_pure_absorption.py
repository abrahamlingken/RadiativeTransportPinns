#!/usr/bin/env python
"""
纯吸收介质快速测试
验证重构后的 RadTrans1D 是否正确工作
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Core'))

import torch
import numpy as np

# 导入物理模型
from EquationModels.RadTrans1D import (
    kappa, sigma_s, I_b, kernel_HG, compute_scattering, compute_res,
    add_collocation, add_boundary, KAPPA_CONST, SIGMA_S_CONST, I_B_CONST
)

print("="*60)
print("纯吸收介质物理引擎测试")
print("="*60)

# 1. 检查介质参数
print("\n[1] 介质参数配置:")
print(f"  Kappa (吸收系数) = {KAPPA_CONST}")
print(f"  Sigma_s (散射系数) = {SIGMA_S_CONST}")
print(f"  I_b (黑体辐射) = {I_B_CONST}")
if SIGMA_S_CONST == 0.0 and I_B_CONST == 0.0:
    print("  模式确认: 纯吸收介质 (无散射、无发射)")

# 2. 测试介质函数
print("\n[2] 测试介质函数:")
x_test = torch.tensor([0.0, 0.5, 1.0])
k = kappa(x_test)
s = sigma_s(x_test)
ib = I_b(x_test)
print(f"  kappa(x) = {k.numpy()}")
print(f"  sigma_s(x) = {s.numpy()}")
print(f"  I_b(x) = {ib.numpy()}")
assert torch.allclose(s, torch.zeros_like(s)), "纯吸收介质散射系数应为0"
print("  [OK] 散射系数验证通过 (全为0)")

# 3. 测试HG相函数
print("\n[3] 测试HG相函数 (各向同性):")
mu = torch.tensor([[0.5], [0.0], [-0.5]])  # [3, 1]
mu_prime = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])  # [1, 5]
phi = kernel_HG(mu, mu_prime, g=0.0)
print(f"  输入 mu shape: {mu.shape}")
print(f"  输入 mu_prime shape: {mu_prime.shape}")
print(f"  输出 phi shape: {phi.shape}")  # 应为 [3, 5]
print(f"  相函数值 (应为常数1.0):\n{phi}")
assert phi.shape == (3, 5), f"Shape错误: {phi.shape}"
assert torch.allclose(phi, torch.ones_like(phi)), "各向同性相函数应为1"
print("  [OK] HG相函数验证通过")

# 4. 测试数据生成
print("\n[4] 测试数据生成:")
x_coll, y_coll = add_collocation(100)
x_b, u_b = add_boundary(40)
print(f"  配点数: {x_coll.shape[0]} (期望100)")
print(f"  边界点数: {x_b.shape[0]} (期望40)")
print(f"  配点范围 x: [{x_coll[:,0].min():.2f}, {x_coll[:,0].max():.2f}]")
print(f"  配点范围 mu: [{x_coll[:,1].min():.2f}, {x_coll[:,1].max():.2f}]")
print(f"  左边界入射强度: {u_b[:20].mean():.2f} (期望1.0)")
print(f"  右边界入射强度: {u_b[20:].mean():.2f} (期望0.0)")

# 5. 测试散射积分 (纯吸收应为0)
print("\n[5] 测试散射积分 (纯吸收模式):")
class DummyNetwork(torch.nn.Module):
    def forward(self, x):
        return torch.ones(x.shape[0], 1)

net = DummyNetwork()
x_test = torch.tensor([[0.5, 0.5], [0.3, -0.2]])  # [2, 2]
u_test = torch.ones(2)
mu_test = x_test[:, 1]
x_spatial = x_test[:, 0:1]

scattering = compute_scattering(u_test, x_spatial, mu_test, net, g=0.0)
print(f"  散射积分结果: {scattering.numpy()}")
assert torch.allclose(scattering, torch.zeros_like(scattering)), "纯吸收散射积分应为0"
print("  [OK] 散射积分快速路径验证通过 (全为0)")

print("\n" + "="*60)
print("所有测试通过！物理引擎工作正常。")
print("="*60)
print("\n现在可以运行训练:")
print("  python Scripts/train_1d.py")
