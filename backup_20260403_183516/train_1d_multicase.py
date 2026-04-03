#!/usr/bin/env python
"""
train_1d_multicase.py

多案例自动化训练脚本 - 支持 Case A/B/C 三种介质配置

案例定义 (HG不对称因子 g=0):
- Case A (基准): τ=1.0, ω=0.5 → kappa=0.5, sigma_s=0.5
- Case B (光学厚): τ=4.0, ω=0.5 → kappa=2.0, sigma_s=2.0
- Case C (强散射): τ=1.0, ω=0.9 → kappa=0.1, sigma_s=0.9

使用方法:
    python train_1d_multicase.py          # 运行所有案例 (A, B, C)
    python train_1d_multicase.py --case A # 仅运行 Case A
    python train_1d_multicase.py --case B # 仅运行 Case B
    python train_1d_multicase.py --case C # 仅运行 Case C
"""

import sys
import os
import argparse
import json

# 添加 Core 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Core'))
from ImportFile import *

pi = math.pi
torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ==============================================================================
# 案例配置定义
# ==============================================================================
CASE_CONFIGS = {
    'A': {
        'name': 'CaseA_Baseline',
        'description': '基准案例: τ=1.0, ω=0.5',
        'kappa': 0.5,
        'sigma_s': 0.5,
        'g': 0.0,
        'folder': 'Results_1D_CaseA'
    },
    'B': {
        'name': 'CaseB_OpticallyThick',
        'description': '光学厚介质: τ=4.0, ω=0.5',
        'kappa': 2.0,
        'sigma_s': 2.0,
        'g': 0.0,
        'folder': 'Results_1D_CaseB'
    },
    'C': {
        'name': 'CaseC_HighScattering',
        'description': '强散射介质: τ=1.0, ω=0.9',
        'kappa': 0.1,
        'sigma_s': 0.9,
        'g': 0.0,
        'folder': 'Results_1D_CaseC'
    }
}


def set_equation_params(kappa_val, sigma_s_val, g_val=0.0):
    """
    动态修改 RadTrans1D 中的介质参数
    
    原理: 直接修改模块级别的全局变量
    """
    import EquationModels.RadTrans1D as RadTrans
    
    # 保存原始值（用于恢复）
    original_values = {
        'KAPPA_CONST': RadTrans.KAPPA_CONST,
        'SIGMA_S_CONST': RadTrans.SIGMA_S_CONST,
        'G_HG': RadTrans.G_HG
    }
    
    # 设置新值
    RadTrans.KAPPA_CONST = kappa_val
    RadTrans.SIGMA_S_CONST = sigma_s_val
    RadTrans.G_HG = g_val
    
    print(f"\n[介质参数设置]")
    print(f"  KAPPA_CONST: {original_values['KAPPA_CONST']:.2f} → {kappa_val:.2f}")
    print(f"  SIGMA_S_CONST: {original_values['SIGMA_S_CONST']:.2f} → {sigma_s_val:.2f}")
    print(f"  G_HG: {original_values['G_HG']:.2f} → {g_val:.2f}")
    print(f"  光学厚度 τ = {kappa_val + sigma_s_val:.2f}")
    print(f"  反照率 ω = {sigma_s_val / (kappa_val + sigma_s_val) if (kappa_val + sigma_s_val) > 0 else 0:.2f}")
    
    return original_values


def initialize_inputs_for_case(case_key, len_sys_argv=1):
    """
    为指定案例初始化训练参数
    
    Args:
        case_key: 'A', 'B', 或 'C'
        len_sys_argv: 命令行参数长度（用于兼容原始逻辑）
    """
    case_config = CASE_CONFIGS[case_key]
    
    if len_sys_argv == 1:
        # Random Seed for sampling the dataset
        sampling_seed_ = 32

        # Number of training+validation points
        # 可针对不同案例调整采样密度
        if case_key == 'B':
            # Case B: 光学厚介质，增加配点密度以捕捉快速衰减
            n_coll_ = 20480   # 增加 25%
            n_u_ = 2560       # 增加 25%
        elif case_key == 'C':
            # Case C: 强散射，增加边界点以更好处理复杂边界
            n_coll_ = 16384
            n_u_ = 3072       # 增加 50%
        else:
            # Case A: 默认配置
            n_coll_ = 16384
            n_u_ = 2048
        
        n_int_ = 0

        # Only for Navier Stokes
        n_object = 0
        ob = None

        # Additional Info - 使用案例特定的文件夹路径
        folder_path_ = case_config['folder']
        point_ = "sobol"
        validation_size_ = 0.0
        
        # Network properties - 可针对不同案例微调
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

    return sampling_seed_, n_coll_, n_u_, n_int_, n_object, ob, folder_path_, point_, validation_size_, network_properties_, retrain_, shuffle_


def train_single_case(case_key, Ec):
    """
    训练单个案例
    
    Args:
        case_key: 'A', 'B', 或 'C'
        Ec: Equation module (RadTrans1D)
    """
    case_config = CASE_CONFIGS[case_key]
    
    print("\n" + "="*70)
    print(f"开始训练: {case_config['name']}")
    print(f"描述: {case_config['description']}")
    print("="*70)
    
    # Step 1: 设置介质参数
    set_equation_params(
        case_config['kappa'],
        case_config['sigma_s'],
        case_config['g']
    )
    
    # Step 2: 获取训练配置
    sampling_seed, N_coll, N_u, N_int, N_object, Ob, folder_path, point, validation_size, network_properties, retrain, shuffle = \
        initialize_inputs_for_case(case_key)
    
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
    
    training_set_class = DefineDataset(extrema,
                                       parameters_values,
                                       point,
                                       N_coll_train,
                                       N_b_train,
                                       N_i_train,
                                       N_int_train,
                                       batches=batch_dim,
                                       output_dimension=output_dimension,
                                       space_dimensions=space_dimensions,
                                       time_dimensions=time_dimension,
                                       parameter_dimensions=parameter_dimensions,
                                       random_seed=sampling_seed,
                                       shuffle=shuffle,
                                       type_point_param=type_point_param)
    training_set_class.assemble_dataset()
    
    # Step 6: 创建模型
    max_iter = 50000
    if network_properties["epochs"] != 1:
        max_iter = 1
    
    model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension,
                  network_properties=network_properties)
    
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
    
    optimizer_LBFGS = optim.LBFGS(model.parameters(), lr=0.8, max_iter=max_iter, max_eval=50000, 
                                  history_size=100, line_search_fn="strong_wolfe",
                                  tolerance_change=1.0 * np.finfo(float).eps)
    optimizer_ADAM = optim.Adam(model.parameters(), lr=0.00005)
    
    if N_coll_train != 0:
        final_error_train = fit(model, optimizer_ADAM, optimizer_LBFGS, epoch_ADAM, 
                                training_set_class, validation_set_clsss=None, 
                                verbose=True, training_ic=False)
    else:
        final_error_train = StandardFit(model, optimizer_ADAM, optimizer_LBFGS, 
                                        training_set_class, validation_set_clsss=None, verbose=True)
    
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
        'train_time': train_time,
        'final_error': final_error_train,
        'L2_error': L2_test,
        'rel_L2_error': rel_L2_test
    }


def main():
    """主函数：解析命令行参数并执行训练"""
    parser = argparse.ArgumentParser(
        description='多案例 1D RTE PINN 训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python train_1d_multicase.py          # 运行所有案例 (A, B, C)
  python train_1d_multicase.py --case A # 仅运行 Case A
  python train_1d_multicase.py --case B # 仅运行 Case B
  python train_1d_multicase.py --case C # 仅运行 Case C
        '''
    )
    
    parser.add_argument(
        '--case',
        type=str,
        choices=['A', 'B', 'C', 'all'],
        default='all',
        help='选择要运行的案例 (A, B, C) 或运行所有案例 (all)'
    )
    
    args = parser.parse_args()
    
    # 导入 Equation 模块（在设置参数前导入）
    import EquationModels.RadTrans1D as RadTrans
    
    # 确定要运行的案例列表
    if args.case == 'all':
        cases_to_run = ['A', 'B', 'C']
    else:
        cases_to_run = [args.case]
    
    print("="*70)
    print("多案例 1D RTE PINN 自动化训练系统")
    print("="*70)
    print(f"将运行案例: {', '.join([CASE_CONFIGS[c]['name'] for c in cases_to_run])}")
    
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
    print(f"{'Case':<10} {'训练时间(s)':<15} {'最终损失':<15} {'L2误差':<15} {'相对L2':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['case']:<10} {r['train_time']:<15.2f} {r['final_error']:<15.6e} "
              f"{r['L2_error']:<15.6e} {r['rel_L2_error']:<15.6e}")
    
    print("\n" + "="*70)
    print("所有案例训练完成!")
    print("="*70)


if __name__ == "__main__":
    main()
