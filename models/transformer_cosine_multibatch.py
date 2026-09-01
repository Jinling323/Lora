import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.module_consistency_multibatch import (
    LearnableGlobalLocalMultiheadAttention,
    LoRAMoELinear,  
) # 新增LoRAMoELinear

# 整個 Encoder 的外殼，負責把同一種 Encoder Layer 疊很多層
class TransformerEncoder(nn.Module):

    # encoder_layer：要複製的單層 TransformerEncoderLayer / num_layers：總共堆疊幾層
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, shape,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        features = []  # 儲存每一層產生的 consistent_feature

        for layer in self.layers:
            output, consistent_feature = layer(output, shape, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            features.append(consistent_feature)

        if self.norm is not None:
            output = self.norm(output)

        return output, features  # output：最後一層 Encoder 的輸出，預期為 [L, B, D]

# 實際執行 Transformer 運算的單層結構
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_experts=4,
                 top_k=2, lora_rank=4, lora_alpha=4.0): #新增num_experts, top_k, lora_rank, lora_alpha參數
        super().__init__()
        self.self_attn = LearnableGlobalLocalMultiheadAttention(d_model, nhead, dropout=dropout)
        # Haze-Adaptive LoRA-MoE FFN.  Each Linear has its own router and four
        # LoRA experts; the router selects and mixes two experts per token.
        self.linear1 = LoRAMoELinear(
            d_model, dim_feedforward,
            num_experts=num_experts, top_k=top_k,
            lora_rank=lora_rank, lora_alpha=lora_alpha,
        )   #把原本的nn.Linear換成LoRAMoELinear，這個linear會有自己的router和四個LoRA專家，router會選擇並混合每個token的兩個專家
        self.dropout = nn.Dropout(dropout)
        self.linear2 = LoRAMoELinear(
            dim_feedforward, d_model,
            num_experts=num_experts, top_k=top_k,
            lora_rank=lora_rank, lora_alpha=lora_alpha,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    # 用來檢查是否有傳入 positional embedding；如果有，就將它加入 token feature。
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # Post-Norm Transformer，先做子模組，再做 LayerNorm
    def forward_post(self,
                     src, shape,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        src2, mask = self.self_attn(q, k, shape, src) # src2:attention 更新出的 token feature
        # mask tgt_len, tgt_len, bsz, 1
        # src tgt_len, bsz, embed_dim
        # 利用 consistency mask 對原始 features 加權聚合，也就是對每個位置找出與它局部一致的位置，再整合那些位置的特徵
        # mask:[L, L, B, 1], src:[L, B, D]
        # [L, L, B, D] => [目標 token, 來源 token, 圖片, feature]
        consistent_feature = torch.sum(mask*src, dim=-2).permute(1,0,2) # tgt_len, bsz, embed_dim

        # 第一個 Residual + Norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # LoRA-MoE FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 第二個 Residual + Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, consistent_feature

    # Pre-Norm 版本，即 LayerNorm 放在 attention／FFN 前面
    def forward_pre(self, src, shape,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # Attention 前先 Normalize
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, mask = self.self_attn(q, k, shape, src)
        feature = torch.squeeze(src, dim=1)
        consistent_feature = torch.matmul(mask, feature)
        #consistent_feature = torch.sum(mask * src, dim=1)
        #最後輸出維度維持為 [token數, batch size, feature維度]

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, consistent_feature   # src：目前送進 Transformer 的「輸入特徵」

    def forward(self, src, shape,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # 根據 normalize_before 選擇執行哪種架構
        if self.normalize_before:
            return self.forward_pre(src, shape, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, shape, src_mask, src_key_padding_mask, pos)

#建立 N 個結構相同、參數彼此獨立的 layer
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# 把 activation 字串轉成真正的 PyTorch function
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
