import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

class ResNet(nn.Module):
    def __init__(self, sample_obs: dict):
        super().__init__()

        extractors = {}
        self.feature_size = 0
        in_channels = 3 
        feature_size = 256 

        image_keys = ["rgb", "left_wrist_rgb", "right_wrist_rgb"]
        for key in image_keys:
            if key in sample_obs:
                backbone_model = models.resnet18(
                    weights=models.ResNet18_Weights.DEFAULT, 
                    norm_layer=FrozenBatchNorm2d
                )
                backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
                projection_head = nn.Conv2d(512, feature_size, kernel_size=1)
                extractors[key] = nn.ModuleList([backbone, projection_head])
                with torch.no_grad():
                    sample_img = sample_obs[key].float().permute(0, 3, 1, 2).cpu()
                    features = backbone(sample_img)['feature_map']
                    projected_features = projection_head(features)
                    output_shape = projected_features.shape
                    flattened_size = output_shape[1] * output_shape[2] * output_shape[3]
                    self.feature_size += flattened_size
        
        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            state_feature_size = 256
            extractors["state"] = nn.Linear(state_size, state_feature_size)
            self.feature_size += state_feature_size

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: dict) -> torch.Tensor:
        encoded_tensor_list = []
        
        for key, extractor in self.extractors.items():
            obs = observations[key]
            
            if key in ["rgb", "left_wrist_rgb", "right_wrist_rgb"]:
                obs = obs.float().permute(0, 3, 1, 2)
                obs = obs / 255.0
                features = extractor[0](obs)['feature_map']     # backbone
                projected_features = extractor[1](features)     # projection_head
                encoded_tensor_list.append(torch.flatten(projected_features, start_dim=1))
            
            elif key == "state":
                encoded_tensor_list.append(extractor(obs))

        return torch.cat(encoded_tensor_list, dim=1)
