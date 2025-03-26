from transformers import BartForConditionalGeneration,  BartConfig
from CustomerAttention import InfiniAttention
import torch.nn as nn
import torch

# 修改整个BART模型
class InfiniBART(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # 替换所有编码层注意力
        for layer_idx, layer in enumerate(self.model.encoder.layers):
            layer.self_attn = InfiniAttention(config, layer_idx)
        self.current_original_id = None

    def forward(self, input_ids,  attention_mask, labels, shard_info, original_input_ids = None):
        # 当运行方式不是训练时，意味着没有被扰动的文字，所以可以直接
        if original_input_ids == None:
            original_input_ids = input_ids.clone()
        # 内存管理
        if self.training and shard_info != self.current_original_id:
            self._reset_memory()
            self.current_original_id = shard_info

        with torch.no_grad():
            original_outputs = super().forward(
                input_ids=original_input_ids,
                output_hidden_states=True,
            )
            # 提取各层原始分片的K/V
            for layer_idx, layer in enumerate(self.model.encoder.layers):
                K = layer.self_attn.k_proj(original_outputs.encoder_hidden_states[layer_idx])
                V = layer.self_attn.v_proj(original_outputs.encoder_hidden_states[layer_idx])
                batch_size = K.size(0)
                K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                layer.self_attn._update_memory(K, V)
        
        # 步骤2：用扰动分片正常前向
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def _reset_memory(self):
        """重置所有层的全局内存"""
        for layer in self.model.encoder.layers:
            layer.self_attn.M.zero_()
            layer.self_attn.z.zero_()