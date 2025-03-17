import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import  BartConfig

class RelativePositionEncoding(nn.Module):
    """Transformer-XL风格的相对位置编码"""
    def __init__(self, d_model: int, max_rel_pos: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_rel_pos = max_rel_pos
        # 相对位置编码矩阵 [2*max_rel_pos+1, d_model]
        self.emb = nn.Embedding(2 * max_rel_pos + 1, d_model)
        
    def forward(self, q_len: int, k_len: int, device: torch.device):
        """生成相对位置索引矩阵
        Args:
            q_len: 查询序列长度
            k_len: 键序列长度
        Returns:
            rel_pos: [q_len, k_len, d_model]
        """
        range_q = torch.arange(q_len, device=device)[:, None]  # [q_len, 1]
        range_k = torch.arange(k_len, device=device)[None, :]  # [1, k_len]
        distance = range_q - range_k  # [q_len, k_len]
        
        # 限制在[-max_rel_pos, max_rel_pos]范围内
        distance = torch.clamp(distance, -self.max_rel_pos, self.max_rel_pos)
        distance += self.max_rel_pos  # 映射到[0, 2*max_rel_pos]
        
        return self.emb(distance)  # [q_len, k_len, d_model]

class InfiniAttention(nn.Module):
    def __init__(self, config: BartConfig, layer_idx: int):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.encoder_attention_heads
        self.head_dim = self.d_model // self.n_heads
        self.layer_idx = layer_idx
        
        # 初始化查询、键、值投影
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # 相对位置编码组件
        self.rel_pos_enc = RelativePositionEncoding(self.head_dim, max_rel_pos=1024)
        
        # 内存参数（每层独立）， 两个不可训练的内存块，用于存储信息 
        # self.memory_matrix = nn.Parameter(torch.zeros(1, self.n_heads, self.head_dim, self.head_dim))
        # self.memory_norm = nn.Parameter(torch.zeros(1, self.n_heads, self.head_dim))
        self.register_buffer("z", torch.zeros(1, self.n_heads, self.head_dim))
        self.register_buffer("M", torch.zeros(1, self.n_heads, self.head_dim, self.head_dim))
        
        # 可训练门控参数
        self.gate_alpha = nn.Parameter(torch.tensor(0.5))  # 初始融合权重
        
    def _update_memory(self, K: torch.Tensor, V: torch.Tensor):
        """更新全局内存矩阵"""
        # K: [batch, n_heads, seq_len, head_dim]
        # V: [batch, n_heads, seq_len, head_dim]
        sigma_K = torch.nn.functional.elu(K) + 1
        self.M += torch.einsum('bhd,bhv->dhv', sigma_K, V)
        self.z += sigma_K.sum(dim=(0,1), keepdim=True)
        
    def forward(self, hidden_states: torch.Tensor, output_attentions, attention_mask: torch.Tensor = None, layer_head_mask = None, ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 投影到查询、键、值空间  K, V 为从原始文本中获取到的信息,k,v是从扰动文本中获取的信息
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # ========== 相对位置编码 ==========
        rel_pos = self.rel_pos_enc(seq_len, seq_len, hidden_states.device)  # [L, L, D]
        rel_pos = rel_pos.unsqueeze(0).unsqueeze(1)  # [1, 1, L, L, D]
        
        # 将位置信息融入注意力计算
        q_with_pos = q.unsqueeze(3) + rel_pos  # [B, H, L, L, D]
        attn_scores = torch.einsum("bhqld,bhkld->bhqlk", q_with_pos, k.unsqueeze(2)) / (self.head_dim ** 0.5)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_scores += attention_mask.view(batch_size, 1, 1, seq_len) * -1e9
            
        # 局部注意力权重
        local_attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, L, L]
        local_context = torch.einsum("bhqlk,bhkvd->bhqvd", local_attn_weights, v)  # [B, H, L, D]
        
        # ========== 全局内存检索 ==========
        sigma_Q = torch.nn.functional.elu(q) + 1
        global_attn = torch.einsum("bhld,bhdk->bhlk", sigma_Q, self.M) / (torch.einsum("bhld,bhd->bhl", sigma_Q, self.z) + 1e-6)
        global_attn = global_attn.transpose(1, 2)  # [B, L, H, D]
        
        # ========== 门控融合 ==========
        gate = torch.sigmoid(self.gate_alpha)
        combined = gate * local_context + (1 - gate) * global_attn
        
        # 合并多头输出
        output = combined.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
    
            
        return output
