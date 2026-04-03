#!/usr/bin/env python
"""快速测试RMC模块导入和基本功能"""

import numpy as np
import sys

# 测试基本导入
print("Testing imports...")
try:
    from numba import jit
    print("  Numba: OK")
except ImportError as e:
    print(f"  Numba: FAILED - {e}")
    sys.exit(1)

# 测试Numba JIT
@jit(nopython=True)
def test_func(x):
    return x * 2

result = test_func(np.array([1.0, 2.0, 3.0]))
print(f"  JIT test: {result}")

# 测试核心函数定义
print("\nTesting core function definitions...")
try:
    # 导入主模块中的函数
    from rmc3d_case_abc import source_term, intersect_boundary_numba, scatter_hg
    
    # 测试source_term
    s = source_term(0.5, 0.5, 0.5)
    print(f"  source_term(center): {s:.4f}")
    
    s2 = source_term(0.9, 0.9, 0.9)  # r > 0.2
    print(f"  source_term(boundary): {s2:.4f}")
    
    # 测试scatter_hg
    ux, uy, uz = scatter_hg(1.0, 0.0, 0.0, 0.0)
    print(f"  scatter_hg (isotropic): ({ux:.4f}, {uy:.4f}, {uz:.4f})")
    
    print("\nAll tests passed!")
    
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
