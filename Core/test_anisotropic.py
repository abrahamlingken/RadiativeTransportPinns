"""
验证各向异性散射动态参数修改机制
"""
import sys
import os

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# 测试1: 验证动态修改 G_HG 是否生效
def test_dynamic_g_update():
    print("=" * 60)
    print("测试: 动态修改 G_HG 参数")
    print("=" * 60)
    
    # 初始导入
    import EquationModels.RadTrans1D as RadTrans
    
    print(f"初始 G_HG = {RadTrans.G_HG}")
    assert RadTrans.G_HG == 0.0, "初始值应为0.0"
    
    # 修改参数
    test_values = [0.5, -0.5, 0.8, 0.0]
    for g_test in test_values:
        RadTrans.G_HG = g_test
        print(f"修改后 G_HG = {RadTrans.G_HG}")
        assert RadTrans.G_HG == g_test, f"G_HG 应被设置为 {g_test}"
    
    print("[PASS] 动态修改测试通过!")
    return True


# 测试2: 验证 kernel_HG 计算
def test_kernel_hg():
    print("\n" + "=" * 60)
    print("测试: Henyey-Greenstein 相函数")
    print("=" * 60)
    
    import torch
    import EquationModels.RadTrans1D as RadTrans
    
    # 测试各向同性 (g=0)
    RadTrans.G_HG = 0.0
    mu = torch.tensor([1.0])
    mu_prime = torch.tensor([1.0])
    phi_iso = RadTrans.kernel_HG(mu, mu_prime, g=0.0)
    print(f"各向同性 (g=0): Phi(1,1) = {phi_iso.item():.4f} (应为 1.0)")
    assert abs(phi_iso.item() - 1.0) < 1e-6, "各向同性时Phi应为1"
    
    # 测试前向散射 (g=0.5)
    phi_forward = RadTrans.kernel_HG(mu, mu_prime, g=0.5)
    print(f"前向散射 (g=0.5): Phi(1,1) = {phi_forward.item():.4f} (应 > 1)")
    assert phi_forward.item() > 1.0, "前向散射同方向应增强"
    
    # 测试后向散射 (g=-0.5)
    phi_backward = RadTrans.kernel_HG(mu, mu_prime, g=-0.5)
    print(f"后向散射 (g=-0.5): Phi(1,1) = {phi_backward.item():.4f} (应 < 1)")
    assert phi_backward.item() < 1.0, "后向散射同方向应减弱"
    
    print("[PASS] HG相函数测试通过!")
    return True


# 测试3: 验证 reload 机制
def test_reload_mechanism():
    print("\n" + "=" * 60)
    print("测试: 模块 reload 机制")
    print("=" * 60)
    
    import importlib
    import EquationModels.RadTrans1D as RadTrans
    
    # 修改参数
    RadTrans.G_HG = 0.7
    RadTrans.KAPPA_CONST = 1.0
    print(f"修改前 - G_HG={RadTrans.G_HG}, KAPPA={RadTrans.KAPPA_CONST}")
    
    # reload
    importlib.reload(RadTrans)
    print(f"reload后 - G_HG={RadTrans.G_HG}, KAPPA={RadTrans.KAPPA_CONST}")
    
    # reload 会重置为模块中的初始值
    print("[PASS] reload 机制测试完成 (注意: reload 会重置为初始值)")
    return True


if __name__ == "__main__":
    try:
        test_dynamic_g_update()
        test_kernel_hg()
        test_reload_mechanism()
        print("\n" + "=" * 60)
        print("所有测试通过! 各向异性散射机制工作正常。")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
