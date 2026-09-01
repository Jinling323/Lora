import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class LoRAExpert(nn.Module):
    """A low-rank residual branch, ``scale * B(A(x))``."""

    def __init__(self, in_features, out_features, rank=4, alpha=4.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be greater than zero")

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = Parameter(torch.empty(rank, in_features))
        self.lora_B = Parameter(torch.empty(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        # The LoRA branch starts at zero, so adding it does not disturb the
        # original Linear layer at initialization.
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class TopKRouter(nn.Module):
    """Token-wise router with auxiliary-loss-free load balancing."""

    def __init__(self, input_dim, num_experts=4, top_k=2):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        self.proj = nn.Linear(input_dim, num_experts) # 算出每個expert的router score

        # 用來修正每個expert的router score的bias，它是從routing統計數據更新，而不是通過反向傳播更新
        self.register_buffer("expert_bias", torch.zeros(num_experts))   
        # 這個是用來統計每個expert在當前batch中被選中的次數，並且不會被保存到模型的state_dict中。
        self.register_buffer(
            "_batch_load", torch.zeros(num_experts), persistent=False
        ) 

    # 每個optimizer step之前清除路由統計數據，否則舊的誤差會被重複使用，導致bias更新不正確
    def reset_load(self): 
        """Clear routing counts before processing a new optimizer step."""
        self._batch_load.zero_()

    #更新每個expert的bias，根據每個expert的負載情況進行調整。
    #它計算平均負載，然後將每個expert的負載與平均負載相減，計算出負載誤差
    #並根據update_rate更新bias。
    @torch.no_grad()   # 不用建立梯度計算紀錄
    def update_expert_bias(self, update_rate):
        """Update bias proportionally to each expert's load violation."""
        average_load = self._batch_load.mean()  #計算平均負載
        if average_load.item() == 0:
            return 0.0

        load_error = average_load - self._batch_load    #計算每個expert的負載誤差（最重要）
        self.expert_bias.add_(load_error, alpha=update_rate) # expert_bias += update_rate * load_error
        max_violation = (self._batch_load.max() - average_load) / average_load #負載平衡指標（觀察用） （最忙的 expert比平均負載高出多少比例）
        return max_violation.item()

    #加上動態bias的分數選擇top-k，再用原始logits計算softmax
    def forward(self, x):   
        logits = self.proj(x)     # x：token feature、self.proj：Router 的 Linear、logits：每個 token 對所有 experts 的原始分數（偏好程度）
        selection_logits = logits + self.expert_bias.to(dtype=logits.dtype)   # 原始分數加上 bias
        # 選擇的專家
        top_indices = torch.topk(
            selection_logits, self.top_k, dim=-1
        ).indices
        
        # 根據 top_indices 記錄的 expert 編號，從 logits 中取出這些 expert 的原始分數
        top_logits = logits.gather(dim=-1, index=top_indices) 
        # 用原始分數計算選中 experts 的融合權重
        top_weights = F.softmax(top_logits.float(), dim=-1).to(logits.dtype)

        if self.training:  # 訓練模式才執行下面的負載統計
            #統計每個 expert ID被選到的次數  
            load = torch.bincount(    # 統計每個 ID 出現幾次
                top_indices.detach().reshape(-1), minlength=self.num_experts
            ).to(self._batch_load.dtype)
            self._batch_load.add_(load)
        return top_indices, top_weights # 每個 token 選了哪些 experts 和那些 experts 的權重


class LoRAMoELinear(nn.Module):
    """Original Linear plus a sparse, router-weighted bank of LoRA experts.

    Each token is sent only to the selected experts.  With the defaults this
    creates four LoRA branches and selects two of them (Top-k=2).
    """

    def __init__(self, in_features, out_features, bias=True,
                 num_experts=4, top_k=2, lora_rank=4, lora_alpha=4.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.lora_enabled = True    #是否啟用LoRA

        self.base_linear = nn.Linear(in_features, out_features, bias=bias)  #原始Linear
        self.router = TopKRouter(in_features, num_experts, top_k)
        self.lora_experts = nn.ModuleList([
            LoRAExpert(in_features, out_features, lora_rank, lora_alpha)
            for _ in range(num_experts)
        ])

    @property   #取得原本linear的權重
    def weight(self):
        """Keep the most useful part of the nn.Linear interface."""
        return self.base_linear.weight

    @property   #取得原本linear的bias
    def bias(self):
        return self.base_linear.bias

    def forward(self, x):
        base_output = self.base_linear(x)
        if not self.lora_enabled:   #如果沒有啟用LoRA就直接回傳原本的linear輸出
            return base_output

        top_indices, top_weights = self.router(x)

        x_flat = x.reshape(-1, self.in_features)
        indices_flat = top_indices.reshape(-1, self.top_k)
        weights_flat = top_weights.reshape(-1, self.top_k)
        lora_output = x_flat.new_zeros(x_flat.size(0), self.out_features)

        # Compute an expert only for the tokens assigned to it.  index_add
        # also handles the weighted sum when a token has two selected experts.
        for expert_id, expert in enumerate(self.lora_experts):
            token_ids, topk_slots = torch.where(indices_flat == expert_id)  #確認哪些token被分配到這個expert
            if token_ids.numel() == 0:  #如果沒有token被分配到這個expert就跳過
                continue
            expert_output = expert(x_flat.index_select(0, token_ids)) #將被分配到這個expert的token送進去計算
            expert_output = expert_output * weights_flat[token_ids, topk_slots].unsqueeze(-1) #將expert的輸出乘上對應的權重
            lora_output.index_add_(0, token_ids, expert_output) #index_add_會自動處理同一個token被分配到多個expert的情況

        lora_output = lora_output.reshape(*x.shape[:-1], self.out_features) 
        return base_output + lora_output

    def enable_lora(self, enabled=True):    #啟用或禁用LoRA experts和router
        """Enable or disable both the LoRA experts and their router."""
        self.lora_enabled = enabled


class LearnableGlobalLocalMultiheadAttention(nn.Module):
    NUM_WEIGHTS = 9
    def __init__(
            self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.bias_k = self.bias_v = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)


    # global
    def in_proj_global_q(self, query):
        return self._in_proj(query, start=0, end=self.embed_dim)

    def in_proj_global_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_global_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim, end=3 * self.embed_dim)

    # local left
    def in_proj_local_left_q(self, query):
        return self._in_proj(query, start=3 * self.embed_dim, end=4 * self.embed_dim)

    def in_proj_local_left_k(self, key):
        return self._in_proj(key, start=4 * self.embed_dim, end=5 * self.embed_dim)

    # local right
    def in_proj_local_right_q(self, query):
        return self._in_proj(query, start=5 * self.embed_dim, end=6 * self.embed_dim)

    def in_proj_local_right_k(self, key):
        return self._in_proj(key, start=6 * self.embed_dim, end=7 * self.embed_dim)

    # local right
    def in_proj_local_q(self, query):
        return self._in_proj(query, start=7 * self.embed_dim, end=8 * self.embed_dim)

    def in_proj_local_k(self, key):
        return self._in_proj(key, start=8 * self.embed_dim, end=9 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def prepare_local_masking(self, q_left, k_left, q_right, k_right, shape):

        left_attn_weights = torch.bmm(q_left, k_left.transpose(1, 2))
        right_attn_weights = torch.bmm(q_right, k_right.transpose(1, 2))

        left_size = left_attn_weights.size()
        src_len = left_size[2]

        triu = torch.ones(src_len, src_len, device=q_left.device, dtype=q_left.dtype).triu_()
        mini_triu = torch.ones(shape[1], shape[1], device=q_left.device, dtype=q_left.dtype).triu_()
        mini_triu = mini_triu.repeat(shape[0], shape[0])
        triu = (triu * mini_triu).unsqueeze_(0)

        left_softmax = F.softmax(left_attn_weights, dim=-1)
        right_softmax = F.softmax(right_attn_weights, dim=-1)

        local_mask = self.compute_lrmask2localmask(left_softmax, right_softmax, triu)

        return local_mask

    def compute_lrmask2localmask(self, left_softmax, right_softmax, triu):
        triu_t = triu.transpose(1,2)
        left_mask = torch.matmul(left_softmax, triu)
        right_mask = torch.matmul(right_softmax, triu_t)
        bw_left_mask = torch.matmul(left_softmax, triu_t)
        bw_right_mask = torch.matmul(right_softmax, triu)

        fw_mask = left_mask * right_mask
        bw_mask = bw_left_mask * bw_right_mask
        local_mask = fw_mask + bw_mask
        return local_mask

    def forward(self, query, key, shape, value):

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        q = self.in_proj_global_q(query)
        k = self.in_proj_global_k(key)
        v = self.in_proj_global_v(value)
        q_left = self.in_proj_local_left_q(query)
        k_left = self.in_proj_local_left_k(key)
        q_right = self.in_proj_local_right_q(query)
        k_right = self.in_proj_local_right_k(key)
        q_local = self.in_proj_local_q(query)
        k_local = self.in_proj_local_k(key)

        q = q*self.scaling
        q_local = q_local * self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_local = q_local.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k_local = k_local.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k_left = k_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k_right = k_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_left = q_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_right = q_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        global_attn_weights = torch.bmm(q, k.transpose(1, 2))
        local_attn_weights = torch.bmm(q_local, k_local.transpose(1, 2))

        local_att_mask = self.prepare_local_masking(q_left, k_left, q_right, k_right, shape)
        masked_local_attn_weights = local_attn_weights * local_att_mask

        attn_weights = 0.1 * global_attn_weights + masked_local_attn_weights

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        local_att_mask = local_att_mask.contiguous().view(bsz, self.num_heads, tgt_len, tgt_len)
        consistent_mask = torch.sum(local_att_mask, dim=1)
        consistent_mask = consistent_mask.permute(1,2,0).unsqueeze(-1)

        return attn, consistent_mask
