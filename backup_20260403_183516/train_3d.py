#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D训练脚本 - 论文Section 3.3配置

运行: python PINNS2_3D.py
"""

import sys
import os

# ============ 关键：必须最先执行模块替换 ============
# 1. 清除任何已缓存的Equations模块
for mod_name in list(sys.modules.keys()):
    if 'EquationModels' in mod_name or 'ModelClassTorch2' in mod_name or 'ImportFile' in mod_name:
        del sys.modules[mod_name]

# 2. 导入3D模块
import EquationModels.RadTrans3D_paper as Ec

# 3. 强制替换sys.modules
sys.modules['EquationModels'] = type(sys)('EquationModels')
sys.modules['EquationModels.RadTrans1D'] = Ec
sys.modules['EquationModels.RadTrans3D_paper'] = Ec

# 4. 创建临时的ImportFile替代
class FakeImportFile:
    pass

fake_import = FakeImportFile()

# 复制Ec的所有属性到fake_import
for attr in dir(Ec):
    if not attr.startswith('_'):
        setattr(fake_import, attr, getattr(Ec, attr))

sys.modules['ImportFile'] = fake_import

# 5. 添加Core目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Core'))

# 6. 现在导入PyTorch和其他库
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import math

# 6. 手动定义需要的函数（避免从ImportFile导入）
pi = math.pi

# 从DatasetTorch2导入（它不依赖Ec）
from DatasetTorch2 import DefineDataset

# 7. 现在安全地导入ModelClassTorch2
# 但我们需要修补其中的Ec引用
import ModelClassTorch2 as _mc

# 替换ModelClassTorch2中的Ec引用
_mc.Ec = Ec

# 现在获取我们需要的类
Pinns = _mc.Pinns
init_xavier = _mc.init_xavier

# ============ 自定义CustomLoss，使用正确的Ec ============
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, network, x_u_train, u_train, x_b_train, u_b_train, x_f_train, x_obj, u_obj, dataclass,
                training_ic, computing_error=False):
        
        lambda_residual = network.lambda_residual
        space_dimensions = dataclass.space_dimensions
        
        u_pred_var_list = list()
        u_train_var_list = list()
        
        for j in range(dataclass.output_dimension):
            # 使用Ec.apply_BC（现在指向3D版本）
            u_pred_b, u_train_b = Ec.apply_BC(x_b_train, u_b_train, network)
            u_pred_var_list.append(u_pred_b)
            u_train_var_list.append(u_train_b)
            
            if Ec.time_dimensions != 0:
                u_pred_0, u_train_0 = Ec.apply_IC(x_u_train, u_train, network)
                u_pred_var_list.append(u_pred_0)
                u_train_var_list.append(u_train_0)
        
        # 计算残差
        res = Ec.compute_res(network, x_f_train, space_dimensions, None, computing_error)
        
        # 边界损失
        loss_b = torch.mean(torch.stack([(torch.mean((u_pred_var_list[i] - u_train_var_list[i])**2)) 
                                         for i in range(len(u_pred_var_list))]))
        
        # 残差损失
        loss_f = torch.mean(res**2)
        
        # 总损失
        loss = lambda_residual * loss_f + loss_b
        
        return loss

# 替换ModelClassTorch2中的CustomLoss
_mc.CustomLoss = CustomLoss

# ============ 全局设置 ============
torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ============ 配置 ============
SAMPLING_SEED = 32
N_COLL = 16384
N_U = 12288
N_INT = 0
FOLDER_PATH = "Results_3D_Section33"
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

os.makedirs(FOLDER_PATH, exist_ok=True)

print("="*70)
print("3D Steady-State Monochromatic Radiative Transfer (Section 3.3)")
print("="*70)

# ============ 获取维度信息 ============
space_dimensions = Ec.space_dimensions
time_dimension = Ec.time_dimensions

if hasattr(Ec, 'parameters_values'):
    parameter_dimensions = Ec.parameters_values.shape[0]
else:
    parameter_dimensions = 0

input_dimensions = parameter_dimensions + time_dimension + space_dimensions
output_dimension = Ec.output_dimension

print(f"Input dimensions: {input_dimensions}")
print(f"Space: {space_dimensions}, Time: {time_dimension}, Param: {parameter_dimensions}")
print(f"Equation module: {Ec.__file__}")

# ============ 生成训练数据 ============
print("\nGenerating training points...")
x_coll, y_coll = Ec.add_collocations(N_COLL)
x_b, y_b = Ec.add_boundary(N_U)

print(f"  Collocation: {x_coll.shape[0]}")
print(f"  Boundary: {x_b.shape[0]}")

# ============ 创建SimpleDataset ============
class SimpleDataset:
    def __init__(self, x_coll, y_coll, x_b, y_b, batch_size,
                 space_dims=3, time_dims=0, param_dims=0,
                 output_dim=1, obj=None):
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
        self.obj = obj
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

# ============ 创建模型 ============
print("\nCreating model...")
model = Pinns(
    input_dimensions,
    output_dimension,
    NETWORK_PROPERTIES,
    additional_models=None
)

torch.manual_seed(RETRAIN)
init_xavier(model)

if torch.cuda.is_available():
    model = model.cuda()
    x_coll = x_coll.cuda()
    y_coll = y_coll.cuda()
    x_b = x_b.cuda()
    y_b = y_b.cuda()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# ============ 训练 ============
print("\n" + "="*70)
print("Training")
print("="*70)

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

def closure():
    global iteration
    
    optimizer_LBFGS.zero_grad()
    
    loss_val = CustomLoss()(
        model,
        torch.empty(0, input_dimensions).cuda() if torch.cuda.is_available() else torch.empty(0, input_dimensions),
        torch.empty(0, 1).cuda() if torch.cuda.is_available() else torch.empty(0, 1),
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
            print(f"Iter {iteration:5d} | Loss: {loss_val.item():.6e} | Time: {elapsed:.1f}s")
    
    iteration += 1
    return loss_val

try:
    optimizer_LBFGS.step(closure)
except KeyboardInterrupt:
    print("\nTraining interrupted")

# ============ 保存结果 ============
final_loss = training_history['total_loss'][-1]

print("\n" + "="*70)
print("Training completed")
print("="*70)
print(f"Final loss: {final_loss:.6e}")
print(f"Iterations: {iteration}")
print(f"Time: {time.time() - start_time:.1f}s")

torch.save(model, os.path.join(FOLDER_PATH, "model.pkl"))

with open(os.path.join(FOLDER_PATH, "training_history.json"), 'w') as f:
    json.dump(training_history, f, indent=2)

info = {
    'N_coll': N_COLL, 'N_sb': N_U,
    'network': NETWORK_PROPERTIES,
    'final_loss': final_loss,
    'iterations': iteration,
    'training_time': time.time() - start_time
}
with open(os.path.join(FOLDER_PATH, "training_info.json"), 'w') as f:
    json.dump(info, f, indent=2)

print(f"\nResults saved to: {FOLDER_PATH}/")

# 生成图表
print("\nGenerating figures...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    epochs = np.array(training_history['epochs'])
    loss = np.array(training_history['total_loss'])
    times = np.array(training_history['time'])
    
    if np.mean(loss) < 0:
        loss = 10 ** loss
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    axes[0, 0].plot(epochs, loss, 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Iteration'), axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('(a) Linear'), axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].semilogy(epochs, loss, 'b-', linewidth=1.5)
    axes[0, 1].set_xlabel('Iteration'), axes[0, 1].set_ylabel('Loss (log)')
    axes[0, 1].set_title('(b) Log Scale'), axes[0, 1].grid(True, alpha=0.3, which='both')
    
    axes[1, 0].semilogy(times, loss, 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)'), axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('(c) Time'), axes[1, 0].grid(True, alpha=0.3, which='both')
    
    if len(loss) > 1:
        dlog = np.diff(np.log10(loss))
        axes[1, 1].plot(epochs[1:], dlog, 'r-', linewidth=1)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Iteration'), axes[1, 1].set_ylabel('Rate')
        axes[1, 1].set_title('(d) Rate'), axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER_PATH, 'training_curves.png'), dpi=300)
    print("Figures saved")
    
except Exception as e:
    print(f"Error generating figures: {e}")

print("\n" + "="*70)
print("Done!")
print("="*70)
