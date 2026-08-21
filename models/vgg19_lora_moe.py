import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from .transformer_lora_moe import TransformerEncoder, TransformerEncoderLayer

__all__ = ["vgg19_lora_moe"]

model_urls = {"vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"}

class VGG_Trans(nn.Module):
    """
    VGG19 + Transformer Encoder + Haze-Adaptive LoRA-MoE FFN
    + Regression Head.

    Transformer FFN:
        Linear 1
            ├── original Linear output
            └── Router -> Top-k=2 -> 2/4 LoRA experts
                         ↓
                       weighted sum
        original Linear1 + LoRA-MoE
                    ↓
                  ReLU
                    ↓
        Linear 2
            ├── original Linear output
            └── Router -> Top-k=2 -> 2/4 LoRA experts
                         ↓
                       weighted sum
        original Linear2 + LoRA-MoE
                    ↓
              Transformer residual
                    ↓
              Regression Head
    """

    def __init__(self,
        features,
        d_model=512,
        nhead=2,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        num_experts=4,
        top_k=2,
        lora_rank=4,
        lora_alpha=4.0,
        freeze_base_linear=False,
    ):
        super(VGG_Trans, self).__init__()

        self.features = features

        # ============================================================
        # Transformer Encoder
        # ============================================================
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,

            # Haze-Adaptive LoRA-MoE settings
            num_experts=num_experts,
            top_k=top_k,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base_linear=freeze_base_linear,
        )

        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers,
            if_norm,
        )

        # ============================================================
        # Regression Decoder / Regression Head
        #
        # Encoder output:
        #     [B, 512, H, W]
        #
        # After upsampling:
        #     [B, 512, H/16, W/16]
        #
        # Regression head:
        #     512 -> 256 -> 128 -> 1
        # ============================================================
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        # ============================================================
        # 1. VGG feature extraction
        # ============================================================
        b, c, h, w = x.shape

        # VGG19 has a total downsampling factor of 32.
        # The original model uses H/16 and W/16 as the final density
        # map resolution.
        rh = int(h) // 16
        rw = int(w) // 16

        x = self.features(x)

        # x: [B, 512, H_vgg, W_vgg]
        bs, c, h, w = x.shape

        # ============================================================
        # 2. Convert CNN feature map to Transformer sequence
        #
        # [B, C, H, W]
        #       ↓ flatten spatial dimensions
        # [B, C, H*W]
        #       ↓ permute
        # [H*W, B, C]
        # ============================================================
        x = x.flatten(2).permute(2, 0, 1)

        # ============================================================
        # 3. Transformer Encoder
        #
        # Inside each EncoderLayer:
        #
        # Linear1
        #   + Top-k LoRA-MoE (4 experts, select 2)
        #       ↓
        #     ReLU
        #       ↓
        # Linear2
        #   + Top-k LoRA-MoE (4 experts, select 2)
        #
        # Then normal Transformer residual + LayerNorm.
        # ============================================================
        x, features = self.encoder(x, (h, w))

        # ============================================================
        # 4. Convert Transformer sequence back to feature map
        #
        # [H*W, B, C]
        #       ↓
        # [B, C, H*W]
        #       ↓
        # [B, C, H, W]
        # ============================================================
        x = x.permute(1, 2, 0).view(bs, c, h, w)

        # ============================================================
        # 5. Upsampling
        # ============================================================
        x = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)
        
        # ============================================================
        # 6. Regression Decoder / Head
        # ============================================================
        x = self.reg_layer_0(x)

        # Density map must be non-negative.
        x = torch.relu(x)

        return x, features


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == "M":
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
    "E": [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, 256, "M",
        512, 512, 512, 512, "M",
        512, 512, 512, 512, "M",
    ]
}


def vgg19_lora_moe(
    pretrained_vgg=True,
    freeze_base_linear=False,
    num_experts=4,
    top_k=2,
    lora_rank=4,
    lora_alpha=4.0):
    """
    VGG19 + Transformer + Haze-Adaptive LoRA-MoE model.

    Parameters
    ----------
    pretrained_vgg : bool
        Load ImageNet-pretrained VGG19 weights into self.features.

    freeze_base_linear : bool
        If True, freeze the original Transformer FFN Linear1/Linear2
        weights and train only LoRA + Router parameters.

    num_experts : int
        Number of LoRA experts for EACH Transformer Linear layer.
        Default = 4.

    top_k : int
        Number of LoRA experts selected by EACH Router.
        Default = 2.

    lora_rank : int
        Rank of each LoRA expert.

    lora_alpha : float
        LoRA scaling alpha.
    """

    model = VGG_Trans(make_layers(cfg["E"]),
        d_model=512,
        nhead=2,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,

        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        freeze_base_linear=freeze_base_linear)

    # Only load the VGG19 ImageNet weights.
    #
    # The VGG state_dict contains classifier weights as well, but
    # strict=False allows the feature extractor to load the matching
    # "features.*" parameters while ignoring unrelated parameters.
    if pretrained_vgg:
        pretrained_dict = model_zoo.load_url(model_urls["vgg19"])

        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)

        print("Loaded ImageNet VGG19 weights.")
        print("Missing keys (expected for Transformer/LoRA/Head):")
        print(missing)

        if unexpected:
            print("Unexpected keys from VGG checkpoint:")
            print(unexpected)

    return model
