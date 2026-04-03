"""
任务 1：Chunking 机制代码片段
展示如何在 PINNS2_3D.py 的 CustomLoss 中引入显存安全的分块计算

核心思想：
- 当 x_f_train 很大时 (如 20000 点)，不一次性送入 compute_res
- 切分成 chunks (如 4096)，逐个计算残差平方
- 最后求均值得到 loss_res
"""

import torch
import torch.nn as nn

# ============================================
# 方案：包装 compute_res 支持 Chunking
# ============================================

def compute_res_chunked(network, x_f_train, space_dimensions, solid_object, 
                        computing_error, chunk_size=4096):
    """
    分块计算残差，避免OOM
    
    Args:
        network: PINN网络
        x_f_train: 配点张量，shape: [N, 5]
        chunk_size: 每个chunk的大小
        
    Returns:
        res_squared_mean: 残差平方的均值 (标量)
    """
    N = x_f_train.shape[0]
    
    # 如果数据量小，直接计算
    if N <= chunk_size:
        res = Ec.compute_res(network, x_f_train, space_dimensions, 
                            solid_object, computing_error)
        return torch.mean(res ** 2)
    
    # 大batch分块处理
    res_squared_list = []
    
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        x_chunk = x_f_train[i:end_i]
        
        # 计算当前chunk的残差
        res_chunk = Ec.compute_res(network, x_chunk, space_dimensions, 
                                   solid_object, computing_error)
        
        # 保存残差平方的均值
        res_squared_list.append(torch.mean(res_chunk ** 2))
        
        # 清理显存
        del res_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 所有chunk残差平方的均值
    loss_res = torch.mean(torch.stack(res_squared_list))
    
    return loss_res


# ============================================
# 应用到 CustomLoss 中
# ============================================

class CustomLossWithChunking(nn.Module):
    """
    带Chunking机制的CustomLoss
    可配置 chunk_size 适应不同显存大小
    """
    def __init__(self, chunk_size=4096):
        super(CustomLossWithChunking, self).__init__()
        self.chunk_size = chunk_size  # 根据显存调整：24GB可设4096或8192
    
    def forward(self, network, x_u_train, u_train, x_b_train, u_b_train, 
                x_f_train, x_obj, u_obj, dataclass, training_ic, computing_error=False):
        
        lambda_residual = network.lambda_residual
        space_dimensions = dataclass.space_dimensions
        
        # ========== 边界损失 (通常数据量小，不需要chunking) ==========
        u_pred_var_list = []
        u_train_var_list = []
        
        for j in range(dataclass.output_dimension):
            u_pred_b, u_train_b = Ec.apply_BC(x_b_train, u_b_train, network)
            u_pred_var_list.append(u_pred_b)
            u_train_var_list.append(u_train_b)
            
            if Ec.time_dimensions != 0:
                u_pred_0, u_train_0 = Ec.apply_IC(x_u_train, u_train, network)
                u_pred_var_list.append(u_pred_0)
                u_train_var_list.append(u_train_0)
        
        # 边界损失
        loss_b = torch.mean(torch.stack([
            torch.mean((u_pred_var_list[i] - u_train_var_list[i])**2)
            for i in range(len(u_pred_var_list))
        ]))
        
        # ========== 残差损失 (使用Chunking！) ==========
        # 原代码：直接计算，容易OOM
        # res = Ec.compute_res(network, x_f_train, space_dimensions, None, computing_error)
        # loss_f = torch.mean(res**2)
        
        # 新代码：分块计算，显存安全
        loss_f = compute_res_chunked(
            network, x_f_train, space_dimensions, None, computing_error,
            chunk_size=self.chunk_size
        )
        
        # 总损失
        loss = lambda_residual * loss_f + loss_b
        
        return loss


# ============================================
# 使用示例 (添加到 PINNS2_3D.py 的修改)
# ============================================

"""
在 PINNS2_3D.py 中，将原来的 CustomLoss 类替换为：

# ============ 配置Chunking参数 ============
CHUNK_SIZE = 4096  # 根据显存调整：
                   # - 24GB RTX 4090: 4096 或 8192
                   # - 12GB GPU: 2048
                   # - 更小显存: 1024 或 512

# ============ 自定义CustomLoss ============
class CustomLoss(nn.Module):
    def __init__(self, chunk_size=4096):
        super(CustomLoss, self).__init__()
        self.chunk_size = chunk_size

    def forward(self, network, x_u_train, u_train, x_b_train, u_b_train, 
                x_f_train, x_obj, u_obj, dataclass, training_ic, computing_error=False):
        
        lambda_residual = network.lambda_residual
        space_dimensions = dataclass.space_dimensions
        
        # 边界损失
        u_pred_var_list = list()
        u_train_var_list = list()
        
        for j in range(dataclass.output_dimension):
            u_pred_b, u_train_b = Ec.apply_BC(x_b_train, u_b_train, network)
            u_pred_var_list.append(u_pred_b)
            u_train_var_list.append(u_train_b)
            
            if Ec.time_dimensions != 0:
                u_pred_0, u_train_0 = Ec.apply_IC(x_u_train, u_train, network)
                u_pred_var_list.append(u_pred_0)
                u_train_var_list.append(u_train_0)
        
        loss_b = torch.mean(torch.stack([(torch.mean((u_pred_var_list[i] - u_train_var_list[i])**2)) 
                                         for i in range(len(u_pred_var_list))]))
        
        # ========== 关键修改：使用Chunking计算残差损失 ==========
        N_coll = x_f_train.shape[0]
        
        if N_coll <= self.chunk_size:
            # 小batch直接计算
            res = Ec.compute_res(network, x_f_train, space_dimensions, None, computing_error)
            loss_f = torch.mean(res**2)
        else:
            # 大batch分块计算
            n_chunks = (N_coll + self.chunk_size - 1) // self.chunk_size
            loss_f_accum = 0.0
            
            for i in range(n_chunks):
                start_idx = i * self.chunk_size
                end_idx = min((i + 1) * self.chunk_size, N_coll)
                x_chunk = x_f_train[start_idx:end_idx]
                
                res_chunk = Ec.compute_res(network, x_chunk, space_dimensions, None, computing_error)
                loss_f_accum = loss_f_accum + torch.mean(res_chunk**2)
                
                # 显式清理
                del res_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            loss_f = loss_f_accum / n_chunks
        
        loss = lambda_residual * loss_f + loss_b
        return loss

# 在创建optimizer前设置chunk_size
CHUNK_SIZE = 4096  # 根据你的显存调整

# 替换ModelClassTorch2中的CustomLoss
_mc.CustomLoss = CustomLoss  # 使用默认chunk_size
# 或
_mc.CustomLoss = lambda: CustomLoss(chunk_size=CHUNK_SIZE)
"""


# ============================================
# 显存监控辅助函数 (可选)
# ============================================

def get_gpu_memory():
    """获取GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3     # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': total - reserved
        }
    return None


def estimate_chunk_size(gpu_memory_gb=None, safety_factor=0.7):
    """
    根据显存大小估算合适的chunk_size
    
    Args:
        gpu_memory_gb: GPU显存大小(GB)，None则自动检测
        safety_factor: 安全系数
    
    Returns:
        建议的chunk_size
    """
    if gpu_memory_gb is None and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # 经验公式：每个点约需要 5 * N_q * 4 bytes (N_q ≈ 128)
    # 额外开销系数 2.0
    bytes_per_point = 5 * 128 * 4 * 2.0  # ≈ 5KB per point
    
    safe_memory = gpu_memory_gb * safety_factor * 1024**3  # bytes
    chunk_size = int(safe_memory / bytes_per_point)
    
    # 对齐到1024的倍数
    chunk_size = (chunk_size // 1024) * 1024
    
    # 限制范围
    chunk_size = max(512, min(chunk_size, 16384))
    
    print(f"[Memory] GPU: {gpu_memory_gb:.1f}GB | "
          f"Estimated chunk_size: {chunk_size} (safety_factor={safety_factor})")
    
    return chunk_size


# 使用示例
if __name__ == "__main__":
    # 自动估算chunk_size
    chunk_size = estimate_chunk_size()
    print(f"Recommended chunk_size: {chunk_size}")
