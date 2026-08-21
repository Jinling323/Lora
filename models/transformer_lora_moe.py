import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.module_consistency import LearnableGlobalLocalMultiheadAttention

class LoRAExpert(nn.Module):
    """
    One LoRA expert for a Linear layer.

    Standard LoRA:
        ΔW = B @ A
        A: [rank, in_features]
        B: [out_features, rank]

    The base Linear weight is kept in the original Linear layer.
    Only A/B are the LoRA trainable parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 4.0):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # LoRA initialization
        # A: Gaussian
        # B: Zero
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., in_features]
        low_rank = F.linear(x, self.lora_A)       # [..., rank]
        return F.linear(low_rank, self.lora_B) * self.scaling


class TopKRouter(nn.Module):
    """
    Router used by each LoRA-MoE Linear.

    4 experts by default, Top-k=2.
    The router first calculates an affinity score for every expert,
    keeps only the top-k scores, then applies softmax over those
    selected experts so their weights sum to 1.
    """

    def __init__(self, input_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()

        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x: Tensor):
        # x: [..., input_dim]
        logits = self.router(x)

        top_values, top_indices = torch.topk(
            logits, k=self.top_k, dim=-1
        )

        # Only selected experts participate in the weighted fusion.
        top_weights = F.softmax(top_values, dim=-1)

        return top_indices, top_weights


class LoRAMoELinear(nn.Module):
    """
    Original Linear + Top-k LoRA-MoE residual branch.

    Architecture:
        x
         ├──────────────> W0 x ───────────────┐
         │                                     +
         └─> Router ─> Top-k=2 ─> LoRA experts ┘
                                             │
                                             y

    There are 4 LoRA experts and only 2 are selected for each token.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        freeze_base: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k

        # This is the original Transformer Linear.
        self.base = nn.Linear(in_features, out_features, bias=bias)

        # Router is separate for Linear 1 and Linear 2 because their
        # input dimensions are different.
        self.router = TopKRouter(
            input_dim=in_features,
            num_experts=num_experts,
            top_k=top_k,
        )

        # Four independent LoRA experts.
        self.experts = nn.ModuleList([
            LoRAExpert(
                in_features=in_features,
                out_features=out_features,
                rank=lora_rank,
                alpha=lora_alpha,
            )
            for _ in range(num_experts)
        ])

        # For a pretrained Transformer, set freeze_base=True to follow
        # the original LoRA idea of freezing W0.
        if freeze_base:
            self.base.weight.requires_grad_(False)
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        # Original Linear output.
        base_output = self.base(x)

        # Router selects Top-k experts for every token.
        top_indices, top_weights = self.router(x)

        # Flatten all leading dimensions so this works for both
        # [sequence, batch, dim] and [batch, sequence, dim].
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        idx_flat = top_indices.reshape(-1, self.top_k)
        weight_flat = top_weights.reshape(-1, self.top_k)

        # Only compute LoRA experts for tokens assigned to them.
        # This preserves the sparse Top-k behavior conceptually:
        # unselected experts do not contribute.
        lora_output_flat = x_flat.new_zeros(
            x_flat.size(0), self.out_features
        )

        for expert_id, expert in enumerate(self.experts):
            # [num_selected_tokens, top_k]
            selected = (idx_flat == expert_id)

            if not selected.any():
                continue

            token_ids, slot_ids = torch.where(selected)

            expert_x = x_flat[token_ids]
            expert_y = expert(expert_x)

            # The selected expert output is multiplied by its
            # normalized Top-k gating weight.
            expert_y = expert_y * weight_flat[token_ids, slot_ids].unsqueeze(-1)

            lora_output_flat.index_add_(0, token_ids, expert_y)

        lora_output = lora_output_flat.reshape(
            *original_shape[:-1], self.out_features
        )

        # IMPORTANT:
        # LoRA-MoE output is added to the original Linear output
        # BEFORE ReLU for Linear 1, exactly matching the slide.
        output = base_output + lora_output
        return output


class TransformerEncoder(nn.Module):

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
        features = []

        for layer in self.layers:
            output, consistent_feature = layer(
                output,
                shape,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
            features.append(consistent_feature)

        if self.norm is not None:
            output = self.norm(output)

        return output, features


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        num_experts=4,
        top_k=2,
        lora_rank=4,
        lora_alpha=4.0,
        freeze_base_linear=False
    ):
        super().__init__()

        self.self_attn = LearnableGlobalLocalMultiheadAttention(d_model,nhead,dropout=dropout)

        # ============================================================
        # Haze-Adaptive LoRA-MoE FFN
        #
        #                    ┌─ LoRA Expert 1 ─┐
        #                    ├─ LoRA Expert 2 ─┤
        # src -> Linear 1 ---┼─ LoRA Expert 3 ─┼-> Top-k Router
        #                    └─ LoRA Expert 4 ─┘
        #                         |
        #             original Linear1 + LoRA-MoE
        #                         |
        #                        ReLU
        #                         |
        #                    Linear 2 + LoRA-MoE
        #                         |
        #                       FFN out
        #
        # Linear 1 and Linear 2 each have their OWN 4-expert router.
        # ============================================================

        self.linear1 = LoRAMoELinear(
            in_features=d_model,
            out_features=dim_feedforward,
            num_experts=num_experts,
            top_k=top_k,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base_linear)

        self.dropout = nn.Dropout(dropout)

        self.linear2 = LoRAMoELinear(
            in_features=dim_feedforward,
            out_features=d_model,
            num_experts=num_experts,
            top_k=top_k,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base_linear)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, shape,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None):
        
        q = k = self.with_pos_embed(src, pos)

        src2, mask = self.self_attn(q, k, shape, src)
        feature = torch.squeeze(src, dim=1)
        consistent_feature = torch.matmul(mask, feature)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ============================================================
        # Haze-Adaptive LoRA-MoE:
        #
        # 1. Original Linear 1
        # 2. Top-k=2 selects 2 of 4 LoRA experts
        # 3. Weighted LoRA output + original Linear 1 output
        # 4. ReLU
        # 5. Original Linear 2
        # 6. Top-k=2 selects 2 of 4 LoRA experts
        # 7. Weighted LoRA output + original Linear 2 output
        # 8. Return to the normal Transformer residual path
        # ============================================================
        src2 = self.linear1(src)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)

        src2 = self.linear2(src2)

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, consistent_feature

    def forward_pre(self, src, shape,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        src2, mask = self.self_attn(q, k, shape, src)
        feature = torch.squeeze(src, dim=1)
        consistent_feature = torch.matmul(mask, feature)

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)

        # Same Haze-Adaptive LoRA-MoE FFN in pre-norm mode.
        src2 = self.linear1(src2)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)

        src = src + self.dropout2(src2)

        return src, consistent_feature

    def forward(self, src, shape,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None
    ):
        if self.normalize_before:
            return self.forward_pre(src, shape, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, shape, src_mask, src_key_padding_mask,pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(
        f"activation should be relu/gelu, not {activation}."
    )
