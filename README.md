# **Physic Informed Neural Network for Simulating Radiative Transfer**

Repository to reproduce the experiments in the paper: https://arxiv.org/abs/2009.13291

## 项目概述

本项目实现了基于物理信息神经网络（Physics-Informed Neural Networks, PINNs）的辐射传输方程求解方法。项目使用PyTorch框架构建神经网络，通过将物理方程作为损失函数的一部分，实现对辐射传输过程的数值模拟。

## 环境配置

### 依赖项
- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Pandas
- pyDOE

### 安装步骤

1. 创建Conda环境：
```bash
conda env create -f environment.yml
```

2. 激活环境：
```bash
conda activate radiative-transport-pinns
```

3. 验证安装：
```bash
python -c "import torch; print(torch.__version__)"
```

## 项目文件说明

### 核心训练脚本

#### PINNS2.py
**用途**：项目的主要训练脚本，负责初始化参数、创建数据集、训练神经网络。

**主要功能**：
- 初始化训练参数（采样种子、训练点数量、网络结构等）
- 创建训练数据集（配置点、边界点、内部点）
- 构建神经网络模型
- 执行训练循环并保存结果

**运行方式**：
```bash
# 使用默认参数运行
python PINNS2.py

# 使用自定义参数运行（17个参数）
python PINNS2.py <sampling_seed> <n_coll> <n_u> <n_int> <n_object> <ob> <folder_path> <point> <validation_size> <network_properties> <retrain> <shuffle>
```

**参数说明**：
- `sampling_seed`: 数据集采样的随机种子
- `n_coll`: 配置点数量
- `n_u`: 边界点数量
- `n_int`: 内部点数量
- `n_object`: 对象数量（用于Navier Stokes）
- `ob`: 对象信息
- `folder_path`: 结果保存路径
- `point`: 采样点类型（"sobol"或"uniform"）
- `validation_size`: 验证集比例
- `network_properties`: 网络结构参数（JSON格式）
- `retrain`: 重训练次数
- `shuffle`: 是否打乱数据

### 数据处理脚本

#### DatasetTorch2.py
**用途**：数据集生成和采样工具类，负责创建训练所需的各种采样点。

**主要功能**：
- 实现多种采样方法（均匀采样、Sobol序列采样、拉丁超立方采样）
- 生成配置点（Collocation Points）
- 生成边界点（Boundary Points）
- 生成内部点（Interior Points）
- 支持参数空间的采样

**关键方法**：
- `add_boundary(n_samples)`: 添加边界点
- `add_collocation(n_samples)`: 添加配置点
- `add_interior(n_samples)`: 添加内部点
- `add_parameter(n_samples)`: 添加参数采样点

### 模型定义脚本

#### ModelClassTorch2.py
**用途**：神经网络模型定义和训练函数。

**主要功能**：
- 定义全连接神经网络结构
- 实现物理信息损失函数
- 提供训练和验证方法
- 支持多种激活函数

**网络结构参数**：
- `hidden_layers`: 隐藏层数量
- `neurons`: 每层神经元数量
- `activation`: 激活函数类型（"tanh"、"sigmoid"、"relu"等）
- `residual_parameter`: 残差连接参数
- `kernel_regularizer`: 核正则化参数
- `regularization_parameter`: 正则化系数

#### ImportFile.py
**用途**：核心导入文件，统一管理所有必要的库和模块。

**主要功能**：
- 导入PyTorch、NumPy、SciPy等基础库
- 导入项目自定义模块
- 设置全局配置参数

### 方程模型脚本（EquationModels文件夹）

#### RadiativeInverseBEST.py
**用途**：辐射传输逆问题求解模型，项目默认使用的模型文件。

**主要功能**：
- 定义辐射传输方程的物理约束
- 实现边界条件和初始条件
- 计算物理方程残差
- 支持参数反演

**适用场景**：辐射传输逆问题求解、参数识别

#### RadTrans1D.py
**用途**：一维辐射传输方程模型。

**主要功能**：
- 定义一维空间中的辐射传输方程
- 实现一维边界条件
- 计算一维物理残差

**适用场景**：一维辐射传输问题

#### RadTrans3D.py
**用途**：三维辐射传输方程模型。

**主要功能**：
- 定义三维空间中的辐射传输方程
- 实现三维边界条件
- 计算三维物理残差

**适用场景**：三维辐射传输问题

#### RadTrans3D_t.py
**用途**：含时间项的三维辐射传输方程模型。

**主要功能**：
- 定义时空三维辐射传输方程
- 实现初始条件和边界条件
- 计算时空物理残差

**适用场景**：时间相关的三维辐射传输问题

#### RadiativeFreqRan2.py
**用途**：频率相关的辐射传输方程模型。

**主要功能**：
- 定义频率相关的辐射传输方程
- 处理多频率辐射问题
- 计算频率相关物理残差

**适用场景**：多频率辐射传输问题

### 辅助工具脚本

#### single_retraining.py
**用途**：单次重训练脚本，用于在已有模型基础上进行微调。

**主要功能**：
- 加载预训练模型
- 执行额外的训练迭代
- 保存更新后的模型

#### ObjectClass.py
**用途**：对象类定义，用于处理复杂几何形状和边界条件。

**主要功能**：
- 定义几何对象
- 处理边界条件
- 计算对象相关的物理量

#### EnsambleTraining.py
**用途**：集成训练脚本，使用多个模型进行训练以提高稳定性。

**主要功能**：
- 训练多个神经网络模型
- 集成多个模型的预测结果
- 提高预测的鲁棒性

#### CollectUtils.py
**用途**：数据收集工具类，用于收集和整理训练结果。

**主要功能**：
- 收集训练过程中的损失值
- 整理预测结果
- 生成可视化数据

#### CollectEnsembleData.py
**用途**：集成数据收集脚本，用于收集集成训练的结果。

**主要功能**：
- 收集多个模型的预测结果
- 计算集成预测的统计量
- 生成集成结果报告

## 训练方法

### 1. 基本训练流程

#### 步骤1：选择模型
在`ImportFile.py`中修改导入的方程模型：
```python
# 默认使用RadiativeInverseBEST
from EquationModels.RadiativeInverseBEST import EquationClass as Ec
```

可用的模型：
- `RadiativeInverseBEST` - 辐射传输逆问题
- `RadTrans1D` - 一维辐射传输
- `RadTrans3D` - 三维辐射传输
- `RadTrans3D_t` - 含时间的三维辐射传输
- `RadiativeFreqRan2` - 频率相关辐射传输

#### 步骤2：配置训练参数
在`PINNS2.py`中修改`initialize_inputs`函数中的参数：

```python
# 采样种子
sampling_seed_ = 32

# 训练点数量
n_coll_ = 8192    # 配置点数量
n_u_ = 120        # 边界点数量
n_int_ = 4096     # 内部点数量

# 采样方法
point_ = "sobol"  # 可选: "sobol" 或 "uniform"

# 网络结构
network_properties_ = {
    "hidden_layers": 4,      # 隐藏层数量
    "neurons": 20,          # 每层神经元数量
    "residual_parameter": 1,
    "kernel_regularizer": 2,
    "regularization_parameter": 0,
    "batch_size": (n_coll_ + n_u_ + n_int_),
    "epochs": 1,            # 训练轮数
    "activation": "tanh"     # 激活函数
}

# 其他参数
folder_path_ = "Inverse"   # 结果保存路径
validation_size_ = 0.0      # 验证集比例
retrain_ = 32               # 重训练次数
shuffle_ = False            # 是否打乱数据
```

#### 步骤3：运行训练
```bash
# 使用默认参数训练
python PINNS2.py
```

### 2. 高级训练方法

#### 使用命令行参数训练
```bash
python PINNS2.py \
  32 \                    # sampling_seed
  8192 \                  # n_coll
  120 \                   # n_u
  4096 \                  # n_int
  0 \                     # n_object
  None \                  # ob
  Inverse \               # folder_path
  sobol \                 # point
  0.0 \                   # validation_size
  '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12408, "epochs": 1, "activation": "tanh"}' \  # network_properties
  32 \                    # retrain
  false                   # shuffle
```

#### 集成训练
使用多个模型进行训练以提高稳定性：

```bash
# 运行集成训练脚本
python EnsambleTraining.py
```

#### 单次重训练
在已有模型基础上进行微调：

```bash
# 运行重训练脚本
python single_retraining.py
```

### 3. 采样方法选择

项目支持三种采样方法：

1. **均匀采样（Uniform Sampling）**
   - 在定义域内均匀随机采样
   - 适用于简单问题
   - 使用方法：设置 `point_ = "uniform"`

2. **Sobol序列采样（Sobol Sequence Sampling）**
   - 使用低差异序列进行采样
   - 采样点分布更均匀，收敛更快
   - 适用于复杂问题
   - 使用方法：设置 `point_ = "sobol"`（推荐）

3. **拉丁超立方采样（Latin Hypercube Sampling）**
   - 在每个维度上均匀分布采样点
   - 适用于高维问题
   - 使用方法：在`DatasetTorch2.py`中修改采样类型

### 4. 网络结构调优

#### 调整网络深度
```python
network_properties_ = {
    "hidden_layers": 6,  # 增加隐藏层数量
    "neurons": 20,
    ...
}
```

#### 调整网络宽度
```python
network_properties_ = {
    "hidden_layers": 4,
    "neurons": 50,  # 增加每层神经元数量
    ...
}
```

#### 调整激活函数
```python
network_properties_ = {
    ...
    "activation": "tanh"   # 可选: "tanh", "sigmoid", "relu", "leaky_relu"
}
```

### 5. 训练监控

训练过程中，程序会输出以下信息：
- 当前训练轮数
- 总损失值
- 各项损失分量（物理损失、边界损失等）
- 预测误差

### 6. 结果评估

训练完成后，结果保存在指定的`folder_path`中：
- 模型参数文件
- 训练历史数据
- 预测结果可视化

使用`CollectUtils.py`和`CollectEnsembleData.py`收集和分析训练结果。

## 注意事项

1. **采样点数量**：增加采样点数量可以提高精度，但会增加计算成本
2. **网络结构**：网络过深可能导致训练困难，网络过浅可能导致欠拟合
3. **激活函数**：tanh函数通常在PINNs中表现较好
4. **采样方法**：Sobol序列采样通常比均匀采样收敛更快
5. **训练轮数**：根据问题复杂度调整，通常需要数百到数千轮

## 引用

如果您使用了本项目的代码，请引用以下论文：

```
@article{radiative2020,
  title={Physics-Informed Neural Networks for Radiative Transfer},
  journal={arXiv preprint arXiv:2009.13291},
  year={2020}
}
```


