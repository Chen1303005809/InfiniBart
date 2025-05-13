import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import  BartConfig

import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for rotary positional encoding"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 计算频率倒数 (公式: 10000^(-2j/d) for j in [0, d//2 - 1])
        theta = torch.arange(0, self.dim, 2).float() / self.dim
        theta = 10000.0 ** (-theta)
        
        # 生成位置编码
        pos = torch.arange(max_seq_len).float()
        
        # 创建角度矩阵 (外积: pos * theta)
        angle = torch.einsum('n,d->nd', pos, theta)
        
        # 缓存cos和sin值
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
    
    def forward(self, q, k):
        """
        Args:
            q: query tensor (batch_size, n_heads, seq_len, head_dim)
            k: key tensor   (batch_size, n_heads, seq_len, head_dim)
        
        Returns:
            旋转后的query和key（保持原始维度顺序）
        """
        batch_size, n_heads, seq_len, head_dim = q.shape
        assert head_dim == self.dim, f"Head dimension must be {self.dim}"
        
        # 获取对应序列位置的编码
        cos = self.cos[:seq_len]  # (seq_len, dim//2)
        sin = self.sin[:seq_len]  # (seq_len, dim//2)
        
        # 调整维度用于广播 (batch_size=1, n_heads=1, seq_len, dim//2)
        cos = cos.view(1, 1, seq_len, -1)  # 新维度顺序 [1, 1, L, D/2]
        sin = sin.view(1, 1, seq_len, -1)
        
        # 分割最后维度为复数对
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        
        # 更高效的实现方式（避免显式reshape）
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        
        return q_rot, k_rot

class InfiniAttention(nn.Module):
    def __init__(self, config: BartConfig, layer_idx: int):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.encoder_attention_heads
        self.head_dim = self.d_model // self.n_heads
        self.layer_idx = layer_idx
        self.d_key = config.d_model // config.encoder_attention_heads
        self.d_value = config.d_model // config.encoder_attention_heads
        
        # 初始化查询、键、值投影
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # 相对位置编码组件
        self.rel_pos_enc = RotaryPositionEmbedding(self.head_dim, max_seq_len=4096)
        
        # 内存参数（每层独立）， 两个不可训练的内存块，用于存储信息 
        # self.memory_matrix = nn.Parameter(torch.zeros(1, self.n_heads, self.head_dim, self.head_dim))
        # self.memory_norm = nn.Parameter(torch.zeros(1, self.n_heads, self.head_dim))

        self.register_buffer("M", torch.zeros(self.n_heads, self.d_key, self.d_value))
        self.register_buffer("z", torch.zeros(self.n_heads, self.d_key))
        
        # 可训练门控参数
        self.gate_alpha = nn.Parameter(torch.tensor(0.5))  # 初始融合权重
        
    def _update_memory(self, K: torch.Tensor, V: torch.Tensor):
        """更新全局内存矩阵"""
        # K: [batch, n_heads, seq_len, head_dim]
        # V: [batch, n_heads, seq_len, head_dim]
        # 已在BART类中修改，故K与V均为4维， 
        sigma_K = torch.nn.functional.elu(K) + 1
        self.M += torch.einsum('bhnk,bhnv->bhkv', sigma_K, V).sum(dim=0)  # kd * （vd）^T = kv
        self.z += sigma_K.sum(dim=(0, 2))
        
    def forward(self, hidden_states: torch.Tensor, output_attentions = None, attention_mask: torch.Tensor = None, layer_head_mask = None, ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 投影到查询、键、值空间
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [Batch, Head, QLen, Dim]
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [Batch, Head, KLen, Dim]
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [Batch, Head, VLen, Dim]
        
        # 注入RoPE相对位置编码
        q, k = self.rel_pos_enc(q, k) 
        
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", q, k) 
        attn_scores = (attn_scores) / (self.head_dim ** 0.5)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_scores += attention_mask.view(batch_size, 1, 1, seq_len) * -1e9
            
        # 局部注意力权重
        local_attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, L, L]
        local_context = torch.einsum("bhqk,bhkd->bhqd", local_attn_weights, v)  # [B, H, L, D]
        
        # ========== 全局内存检索 ==========
        sigma_Q = torch.nn.functional.elu(q) + 1
        global_attn = torch.einsum("bhqk,hkv->bhqv", sigma_Q, self.M) / (torch.einsum("bhqk,hk->bhqk", sigma_Q, self.z) + 1e-6)
        
        # ========== 门控融合 ==========
        gate = torch.sigmoid(self.gate_alpha)
        combined = gate * local_context + (1 - gate) * global_attn
        
        # 合并多头输出
        output = combined.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        # if output_attentions:
        #     return output, local_attn_weights, _
        return output, local_attn_weights, _
