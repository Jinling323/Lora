import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
from .transformer_cosine_multibatch import TransformerEncoder, TransformerEncoderLayer
from .module_consistency_multibatch import LoRAMoELinear


__all__ = ['vgg19_trans']
model_urls = {'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

class VGG_Trans(nn.Module):
    def __init__(self, features, num_experts=4, top_k=2, lora_rank=4,
                 lora_alpha=4.0, freeze_vgg=True):
        super(VGG_Trans, self).__init__()
        self.features = features
        self.freeze_vgg = freeze_vgg

        d_model = 512
        nhead = 2
        num_layers = 2
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                num_experts, top_k, lora_rank,
                                                lora_alpha)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_layers, if_norm)
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        rh = int(h) // 16
        rw = int(w) // 16
        # VGG is the only frozen part during full LoRA-MoE model training.
        if self.freeze_vgg:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x, features = self.encoder(x, (h,w))   # transformer
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        #
        x = F.interpolate(x, size=(rh, rw), mode='bilinear', align_corners=False)
        x = self.reg_layer_0(x)   # regression head
        return torch.relu(x), features

    #告知哪些參數是LoRA experts和router的參數
    def _lora_parameter_ids(self):  
        parameter_ids = set()
        for module in self.modules():
            if isinstance(module, LoRAMoELinear):
                parameter_ids.update(id(p) for p in module.router.parameters())
                parameter_ids.update(id(p) for p in module.lora_experts.parameters())
        return parameter_ids

    #base model總共有多少參數
    def base_parameters(self):  
        """Parameters belonging to the original network (not LoRA/router)."""
        lora_parameter_ids = self._lora_parameter_ids()
        return [p for p in self.parameters() if id(p) not in lora_parameter_ids]

    #lora和router總共有多少參數
    def lora_parameters(self):  
        """Parameters belonging only to the LoRA experts and routers."""
        lora_parameter_ids = self._lora_parameter_ids()
        return [p for p in self.parameters() if id(p) in lora_parameter_ids]

    #啟用或禁用LoRA experts和router
    def enable_lora(self, enabled=True):    
        """Control whether LoRA/router branches participate in inference."""
        for module in self.modules():
            if isinstance(module, LoRAMoELinear):
                module.enable_lora(enabled)

    #每個optimizer step之前清除路由統計數據
    def reset_router_loads(self): 
        """Clear every router's load counter for the next training step."""
        for module in self.modules():
            if isinstance(module, LoRAMoELinear):
                module.router.reset_load()

    #不同的router獨立更新bias，並返回它們的平均MaxVio（觀察用）
    def update_router_biases(self, update_rate): 
        """Update each router independently and return their mean MaxVio."""
        max_violations = []
        for module in self.modules():
            if isinstance(module, LoRAMoELinear):
                max_violations.append(
                    module.router.update_expert_bias(update_rate)
                )
        #只是觀察使用，不影響訓練
        if not max_violations:
            return 0.0
        return sum(max_violations) / len(max_violations) 

    #set_training_stage方法根據訓練階段選擇可訓練的參數組
    def set_training_stage(self, stage):    
        """Select the trainable parameter group for the two-stage schedule."""
        if stage not in ('base', 'lora'):
            raise ValueError("stage must be either 'base' or 'lora'")

        lora_parameter_ids = self._lora_parameter_ids()
        train_lora = stage == 'lora'
        for parameter in self.parameters():
            is_lora_parameter = id(parameter) in lora_parameter_ids
            parameter.requires_grad_(is_lora_parameter == train_lora)

        # LoRA must not alter base-model pretraining predictions.
        self.enable_lora(train_lora)
        # Avoid retaining the frozen VGG activation graph in the LoRA stage.
        self.freeze_vgg = train_lora


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def vgg19_trans(num_experts=4, top_k=2, lora_rank=4, lora_alpha=4.0,
                freeze_vgg=True):  # 是否訓練 VGG 由 freeze_vgg 控制
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_Trans(
        make_layers(cfg['E']),
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        freeze_vgg=freeze_vgg,
    )
    # VGG 固定以 ImageNet 預訓練權重作為初始值。
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    if freeze_vgg:
        for parameter in model.features.parameters():
            parameter.requires_grad_(False)
    return model
