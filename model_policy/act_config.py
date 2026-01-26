from dataclasses import dataclass

@dataclass
class ACTConfig:
    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 1
    n_action_steps: int = 100

    # Architecture
    # Vision backbone
    image_features = True
    robot_state_feature = True
    env_state_feature = False
    robot_state_feature_shape = 6
    env_state_feature_shape = None
    action_feature_shape = 7
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # Transformer layers
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: original ACT has 7 decoder layers, but due to a bug only the first is used
    n_decoder_layers: int = 1

    # VAE
    use_vae: bool = False
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
