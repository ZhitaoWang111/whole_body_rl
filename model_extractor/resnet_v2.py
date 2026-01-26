import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


class ResNet(nn.Module):
    def __init__(self, sample_obs: dict):
        super().__init__()

        # ==== 配置 ====
        self.feature_size = 0
        self.image_keys = [k for k in ["top_rgb", "left_wrist_rgb", "right_wrist_rgb"] if k in sample_obs]
        proj_out = 128
        state_proj_out = 128
        tgt_proj_out = 256

        # ==== 共享 backbone ====
        backbone_model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT,
            norm_layer=FrozenBatchNorm2d
        )
        # 只取 layer4 的输出
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # ==== 为每个图像键单独的 projection head ====
        self.projection_heads = nn.ModuleDict({
            key: nn.Conv2d(512, proj_out, kernel_size=1) for key in self.image_keys
        })

        # ==== state 编码器 ====
        self.state_encoder = None
        if "state" in sample_obs:
            state_in = sample_obs["state"].shape[-1]
            self.state_encoder = nn.Linear(state_in, state_proj_out)
        # ==== target delta 编码器 ====
        self.tgt_pos_encoder = None
        if "tgt_delta" in sample_obs:
            tgt_in = sample_obs["tgt_delta"].shape[-1]
            self.tgt_pos_encoder = nn.Sequential(
                nn.Linear(tgt_in, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, tgt_proj_out),
            )

        # ==== 计算拼接后的 feature_size ====
        # 用 sample_obs 推理一次，确定展平后的维度
        with torch.no_grad():
            # 对每个图像键跑一次（假设所有图像输入 HxW 一致）
            for key in self.image_keys:
                # [B, H, W, C] -> [B, C, H, W], 并归一化到 [0,1]
                sample_img = sample_obs[key].float().permute(0, 3, 1, 2).cpu() / 255.0
                feats = self.backbone(sample_img)["feature_map"]               # [B, 512, h, w]
                proj  = self.projection_heads[key](feats)                      # [B, 256, h, w]
                self.feature_size += proj.shape[1] * proj.shape[2] * proj.shape[3]

            if self.state_encoder is not None:
                self.feature_size += self.state_encoder.out_features
            if self.tgt_pos_encoder is not None:
                self.feature_size += tgt_proj_out

    def forward(self, observations: dict) -> torch.Tensor:
        encoded = []

        # 一次性处理所有图像：每个样本/键各自过共享backbone，再接对应projection
        for key in self.image_keys:
            x = observations[key].float().permute(0, 3, 1, 2) / 255.0
            feats = self.backbone(x)["feature_map"]              # 共享 backbone
            proj  = self.projection_heads[key](feats)            # 键专属 projection
            encoded.append(torch.flatten(proj, start_dim=1))

        if self.state_encoder is not None and "state" in observations:
            encoded.append(self.state_encoder(observations["state"]))
        if self.tgt_pos_encoder is not None and "tgt_delta" in observations:
            encoded.append(self.tgt_pos_encoder(observations["tgt_delta"]))

        return torch.cat(encoded, dim=1)
