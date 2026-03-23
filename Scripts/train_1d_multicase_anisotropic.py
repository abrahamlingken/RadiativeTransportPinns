#!/usr/bin/env python
"""
train_1d_multicase_anisotropic.py

各向异性散射多案例自动化训练脚本
支持 Henyey-Greenstein 相函数，通过不对称因子 g 控制散射方向

物理参数固定: kappa = 0.5, sigma_s = 0.5

案例定义:
- Case D (前向散射):    g = 0.5
- Case E (后向散射):    g = -0.5  
- Case F (强前向散射):  g = 0.8
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Core'))
from ImportFile import *

pi = math.pi
torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ==============================================================================
# 各向异性散射案例配置 (HG相函数)
# ==============================================================================
ANISOTROPIC_CASE_CONFIGS = {
    'D': {
        'name': 'CaseD_ForwardScattering',
        'description': 'Forward scattering (g=0.5)',
        'kappa': 0.5,
        'sigma_s': 0.5,
        'g': 0.5,           # HG不对称因子: 前向散射
        'folder': 'Results_1D_CaseD'
    },
    'E': {
        'name': 'CaseE_BackwardScattering',
        'description': 'Backward scattering (g=-0.5)',
        'kappa': 0.5,
        'sigma_s': 0.5,
        'g': -0.5,          # HG不对称因子: 后向散射
        'folder': 'Results_1D_CaseE'
    },
    'F': {
        'name': 'CaseF_StrongForward',
        'description': 'Strong forward scattering (g=0.8)',
        'kappa': 0.5,
        'sigma_s': 0.5,
        'g': 0.8,           # HG不对称因子: 强前向散射
        'folder': 'Results_1D_CaseF'
    }
}


def set_anisotropic_params(kappa_val, sigma_s_val, g_val):
    """
    动态设置 RadTrans1D 中的各向异性散射参数
    
    通过修改模块级别的全局变量，确保 compute_res 使用正确的 g 值
    """
    import EquationModels.RadTrans1D as RadTrans
    
    # 保存原始值
    original_values = {
        'KAPPA_CONST': RadTrans.KAPPA_CONST,
        'SIGMA_S_CONST': RadTrans.SIGMA_S_CONST,
        'G_HG': RadTrans.G_HG
    }
    
    # 设置新的物理参数
    RadTrans.KAPPA_CONST = kappa_val
    RadTrans.SIGMA_S_CONST = sigma_s_val
    RadTrans.G_HG = g_val
    
    print(f"\n[各向异性散射参数设置]")
    print(f"  KAPPA_CONST:  {original_values['KAPPA_CONST']:.2f} -> {kappa_val:.2f}")
    print(f"  SIGMA_S_CONST: {original_values['SIGMA_S_CONST']:.2f} -> {sigma_s_val:.2f}")
    print(f"  G_HG:          {original_values['G_HG']:.2f} -> {g_val:.2f}")
    
    # 物理意义解释
    if g_val > 0:
        print(f"  散射类型: 前向散射 (g={g_val} > 0)")
    elif g_val < 0:
        print(f"  散射类型: 后向散射 (g={g_val} < 0)")
    else:
        print(f"  散射类型: 各向同性 (g=0)")
    
    return original_values


def initialize_inputs_for_case(case_key, len_sys_argv=1):
    """
    为指定案例初始化训练参数
    """
    case_config = ANISOTROPIC_CASE_CONFIGS[case_key]
    
    if len_sys_argv == 1:
        # Random Seed
        sampling_seed_ = 32

        # 训练点配置 (各向异性散射需要更多配点捕捉角度依赖性)
        n_coll_ = 16384   # 内部配点
        n_u_ = 2048       # 边界点
        n_int_ = 0

        # 其他配置
        n_object = 0
        ob = None
        folder_path_ = case_config['folder']
        point_ = "sobol"
        validation_size_ = 0.0
        
        # 网络配置
        network_properties_ = {
            "hidden_layers": 8,
            "neurons": 32,
            "residual_parameter": 0.1,
            "kernel_regularizer": 2,
            "regularization_parameter": 0,
            "batch_size": (n_coll_ + n_u_ + n_int_),
            "epochs": 1,
            "activation": "tanh"
        }
        
        retrain_ = 32
        shuffle_ = False

    else:
        raise ValueError("Multi-case mode does not support command line args yet")

    return (sampling_seed_, n_coll_, n_u_, n_int_, n_object, ob, 
            folder_path_, point_, validation_size_, network_properties_, 
            retrain_, shuffle_)


def train_single_case(case_key, Ec):
    """
    训练单个各向异性散射案例
    
    Args:
        case_key: 'D', 'E', 或 'F'
        Ec: Equation module (RadTrans1D)
    """
    case_config = ANISOTROPIC_CASE_CONFIGS[case_key]
    
    print("\n" + "="*70)
    print(f"开始训练: {case_config['name']}")
    print(f"描述: {case_config['description']}")
    print("="*70)
    
    # Step 1: 设置各向异性散射参数 (关键！)
    set_anisotropic_params(
        case_config['kappa'],
        case_config['sigma_s'],
        case_config['g']
    )
    
    # Step 2: 获取训练配置
    (sampling_seed, N_coll, N_u, N_int, N_object, Ob, 
     folder_path, point, validation_size, network_properties, 
     retrain, shuffle) = initialize_inputs_for_case(case_key)
    
    print(f"\n[训练配置]")
    print(f"  结果保存路径: {folder_path}/")
    print(f"  配点数量: {N_coll}")
    print(f"  边界点数量: {N_u}")
    
    # Step 3: 创建结果目录
    images_path = folder_path + "/Images"
    model_path = folder_path + "/TrainedModel"
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    # Step 4: 准备训练数据
    if Ec.extrema_values is not None:
        extrema = Ec.extrema_values
        space_dimensions = Ec.space_dimensions
        time_dimension = Ec.time_dimensions
        parameter_dimensions = Ec.parameter_dimensions
    else:
        extrema = None
        space_dimensions = Ec.space_dimensions
        time_dimension = Ec.time_dimensions
        parameter_dimensions = 0
    
    try:
        parameters_values = Ec.parameters_values
        parameter_dimensions = parameters_values.shape[0]
        type_point_param = Ec.type_of_points
    except AttributeError:
        parameters_values = None
        parameter_dimensions = 0
        type_point_param = None
    
    # 输入输出维度
    if hasattr(Ec, 'input_dimensions') and Ec.input_dimensions is not None:
        input_dimensions = Ec.input_dimensions
    else:
        input_dimensions = parameter_dimensions + time_dimension + space_dimensions
    output_dimension = Ec.output_dimension
    
    # 训练/验证集划分
    N_u_train = int(N_u * (1 - validation_size))
    N_coll_train = int(N_coll * (1 - validation_size))
    N_int_train = int(N_int * (1 - validation_size))
    
    N_b_train = int(N_u_train / (2 * space_dimensions)) if space_dimensions > 0 else 0
    N_i_train = 0
    
    print(f"\n[数据集信息]")
    print(f"  训练配点数: {N_coll_train}")
    print(f"  训练边界点数: {N_u_train}")
    
    # Step 5: 创建数据集
    batch_dim = network_properties["batch_size"]
    if batch_dim == "full":
        batch_dim = N_u_train + N_coll_train + N_int_train
    
    training_set_class = DefineDataset(
        extrema, parameters_values, point,
        N_coll_train, N_b_train, N_i_train, N_int_train,
        batches=batch_dim, output_dimension=output_dimension,
        space_dimensions=space_dimensions, time_dimensions=time_dimension,
        parameter_dimensions=parameter_dimensions,
        random_seed=sampling_seed, shuffle=shuffle,
        type_point_param=type_point_param
    )
    training_set_class.assemble_dataset()
    
    # Step 6: 创建模型
    max_iter = 50000
    if network_properties["epochs"] != 1:
        max_iter = 1
    
    model = Pinns(
        input_dimension=input_dimensions,
        output_dimension=output_dimension,
        network_properties=network_properties
    )
    
    torch.manual_seed(retrain)
    init_xavier(model)
    
    if torch.cuda.is_available():
        print("  使用 GPU 训练")
        model.cuda()
    else:
        print("  使用 CPU 训练")
    
    # Step 7: 训练模型
    print("\n[开始训练]")
    start = time.time()
    model.train()
    epoch_ADAM = model.num_epochs - 1
    
    optimizer_LBFGS = optim.LBFGS(
        model.parameters(), lr=0.8, max_iter=max_iter, max_eval=50000,
        history_size=100, line_search_fn="strong_wolfe",
        tolerance_change=1.0 * np.finfo(float).eps
    )
    optimizer_ADAM = optim.Adam(model.parameters(), lr=0.00005)
    
    if N_coll_train != 0:
        final_error_train = fit(
            model, optimizer_ADAM, optimizer_LBFGS, epoch_ADAM,
            training_set_class, validation_set_clsss=None,
            verbose=True, training_ic=False
        )
    else:
        final_error_train = StandardFit(
            model, optimizer_ADAM, optimizer_LBFGS,
            training_set_class, validation_set_clsss=None, verbose=True
        )
    
    train_time = time.time() - start
    final_error_train = float(((10 ** final_error_train) ** 0.5).detach().cpu().numpy())
    
    print(f"\n[训练完成]")
    print(f"  训练时间: {train_time:.2f} 秒")
    print(f"  最终训练损失: {final_error_train:.6e}")
    
    # Step 8: 保存结果
    model = model.eval()
    
    # 计算泛化误差和绘图
    L2_test, rel_L2_test = Ec.compute_generalization_error(model, extrema, images_path)
    Ec.plotting(model, images_path, extrema, None)
    
    # 保存模型
    torch.save(model, model_path + "/model.pkl")
    
    # 保存训练信息
    with open(model_path + os.sep + "Information.csv", "w") as w:
        keys = list(network_properties.keys())
        vals = list(network_properties.values())
        w.write(keys[0])
        for i in range(1, len(keys)):
            w.write("," + keys[i])
        w.write("\n")
        w.write(str(vals[0]))
        for i in range(1, len(vals)):
            w.write("," + str(vals[i]))
    
    # 保存案例配置信息
    with open(folder_path + '/InfoModel.txt', 'w') as file:
        file.write("Case," + case_config['name'] + "\n")
        file.write("Description," + case_config['description'] + "\n")
        file.write("Kappa," + str(case_config['kappa']) + "\n")
        file.write("Sigma_s," + str(case_config['sigma_s']) + "\n")
        file.write("G_HG," + str(case_config['g']) + "\n")
        file.write("ScatteringType," + ("Forward" if case_config['g'] > 0 else "Backward" if case_config['g'] < 0 else "Isotropic") + "\n")
        file.write("Nu_train," + str(N_u_train) + "\n")
        file.write("Nf_train," + str(N_coll_train) + "\n")
        file.write("train_time," + str(train_time) + "\n")
        file.write("L2_norm_test," + str(L2_test) + "\n")
        file.write("rel_L2_norm," + str(rel_L2_test) + "\n")
        file.write("error_train," + str(final_error_train) + "\n")
    
    print(f"\n[结果保存]")
    print(f"  模型: {model_path}/model.pkl")
    print(f"  图像: {images_path}/")
    print(f"  日志: {folder_path}/InfoModel.txt")
    
    return {
        'case': case_key,
        'g': case_config['g'],
        'train_time': train_time,
        'final_error': final_error_train,
        'L2_error': L2_test,
        'rel_L2_error': rel_L2_test
    }


def main():
    """主函数：执行所有各向异性散射案例训练"""
    parser = argparse.ArgumentParser(
        description='各向异性散射多案例 PINN 训练脚本 (HG相函数)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python train_1d_multicase_anisotropic.py          # 运行所有案例 (D, E, F)
  python train_1d_multicase_anisotropic.py --case D # 仅运行 Case D (前向散射)
  python train_1d_multicase_anisotropic.py --case E # 仅运行 Case E (后向散射)
  python train_1d_multicase_anisotropic.py --case F # 仅运行 Case F (强前向散射)
        '''
    )
    
    parser.add_argument(
        '--case',
        type=str,
        choices=['D', 'E', 'F', 'all'],
        default='all',
        help='选择要运行的案例 (D:前向, E:后向, F:强前向) 或运行所有案例 (all)'
    )
    
    args = parser.parse_args()
    
    # 导入 Equation 模块
    import EquationModels.RadTrans1D as RadTrans
    
    # 确定要运行的案例列表
    if args.case == 'all':
        cases_to_run = ['D', 'E', 'F']
    else:
        cases_to_run = [args.case]
    
    print("="*70)
    print("各向异性散射 PINN 训练系统 (Henyey-Greenstein 相函数)")
    print("="*70)
    print(f"\n将运行案例:")
    for c in cases_to_run:
        config = ANISOTROPIC_CASE_CONFIGS[c]
        print(f"  Case {c}: {config['description']}")
    
    # 运行所有指定案例
    results = []
    for case_key in cases_to_run:
        try:
            result = train_single_case(case_key, RadTrans)
            results.append(result)
        except Exception as e:
            print(f"\n[错误] Case {case_key} 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 汇总结果
    print("\n" + "="*70)
    print("训练结果汇总")
    print("="*70)
    print(f"{'Case':<10} {'g':<10} {'训练时间(s)':<15} {'最终损失':<15} {'L2误差':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['case']:<10} {r['g']:>+8.2f}   {r['train_time']:<15.2f} "
              f"{r['final_error']:<15.6e} {r['L2_error']:<15.6e}")
    
    print("\n" + "="*70)
    print("所有案例训练完成!")
    print("="*70)


if __name__ == "__main__":
    main()
