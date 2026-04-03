#!/usr/bin/env python
"""
路径检查工具 - 验证项目文件路径配置是否正确
"""

import os
import sys

def check_paths():
    """检查项目关键路径"""
    print("="*70)
    print("项目路径检查工具")
    print("="*70)
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"\n[1] 项目根目录: {project_root}")
    
    # 检查关键目录
    key_dirs = [
        ('Core', 'Core'),
        ('EquationModels', 'EquationModels'),
        ('Scripts', 'Scripts'),
    ]
    
    print("\n[2] 关键目录检查:")
    for dir_name, dir_path in key_dirs:
        full_path = os.path.join(project_root, dir_path)
        exists = os.path.isdir(full_path)
        status = "[OK] Exists" if exists else "[MISSING] Missing"
        print(f"  {dir_name}: {status}")
        if exists:
            # 列出目录内容
            try:
                files = os.listdir(full_path)[:5]  # 只显示前5个
                print(f"    └─ 内容: {', '.join(files)}...")
            except Exception as e:
                print(f"    └─ 错误: {e}")
    
    # 检查关键文件
    key_files = [
        ('DatasetTorch2.py', os.path.join('Core', 'DatasetTorch2.py')),
        ('ModelClassTorch2.py', os.path.join('Core', 'ModelClassTorch2.py')),
        ('RadTrans3D_Complex.py', os.path.join('EquationModels', 'RadTrans3D_Complex.py')),
        ('train_3d_multicase.py', 'train_3d_multicase.py'),
        ('evaluate_anisotropic.py', 'evaluate_anisotropic.py'),
    ]
    
    print("\n[3] 关键文件检查:")
    for file_name, file_path in key_files:
        full_path = os.path.join(project_root, file_path)
        exists = os.path.isfile(full_path)
        status = "[OK] Exists" if exists else "[MISSING] Missing"
        print(f"  {file_name}: {status}")
    
    # 检查 DOM 结果文件
    dom_files = [
        'dom_solution_g0.5.npy',
        'dom_solution_g-0.5.npy',
        'dom_solution_g0.8.npy',
    ]
    
    print("\n[4] DOM 结果文件检查:")
    for dom_file in dom_files:
        full_path = os.path.join(project_root, dom_file)
        exists = os.path.isfile(full_path)
        status = "[OK] Exists" if exists else "[MISSING] Missing"
        print(f"  {dom_file}: {status}")
    
    # 检查训练结果目录
    result_dirs = [
        'Results_1D_CaseA',
        'Results_1D_CaseB',
        'Results_1D_CaseC',
        'Results_1D_CaseD',
        'Results_1D_CaseE',
        'Results_1D_CaseF',
        'Results_3D_CaseA',
        'Results_3D_CaseB',
        'Results_3D_CaseC',
    ]
    
    print("\n[5] 训练结果目录检查:")
    for result_dir in result_dirs:
        full_path = os.path.join(project_root, result_dir)
        exists = os.path.isdir(full_path)
        model_path = os.path.join(full_path, 'TrainedModel', 'model.pkl')
        model_exists = os.path.isfile(model_path)
        
        if exists and model_exists:
            status = "[OK] Exists (with model)"
        elif exists:
            status = "[OK] Exists (no model)"
        else:
            status = "[MISSING] Missing"
        print(f"  {result_dir}: {status}")
    
    # 检查 Python 路径
    print("\n[6] Python 路径检查:")
    print(f"  当前工作目录: {os.getcwd()}")
    print(f"  sys.path 前3项:")
    for i, p in enumerate(sys.path[:3]):
        print(f"    [{i}] {p}")
    
    # 测试导入
    print("\n[7] 导入测试:")
    
    # 添加项目根目录到路径
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    test_imports = [
        ('Core.DatasetTorch2', 'Core.DatasetTorch2'),
        ('Core.ModelClassTorch2', 'Core.ModelClassTorch2'),
        ('EquationModels.RadTrans3D_Complex', 'EquationModels.RadTrans3D_Complex'),
    ]
    
    for module_name, module_path in test_imports:
        try:
            __import__(module_name)
            print(f"  {module_name}: [OK] Import success")
        except ImportError as e:
            print(f"  {module_name}: [FAIL] Import failed - {e}")
    
    print("\n" + "="*70)
    print("路径检查完成")
    print("="*70)

if __name__ == "__main__":
    check_paths()
