import torch
import torch.nn as nn

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.feature_size = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
        fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.feature_size += feature_size

        # Add wrist camera processing with same architecture
        if "left_wrist_rgb" in sample_obs:
            wrist_in_channels = sample_obs["left_wrist_rgb"].shape[-1]
            wrist_cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=wrist_in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
            
            with torch.no_grad():
                n_flatten_wrist = wrist_cnn(sample_obs["left_wrist_rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            wrist_fc = nn.Sequential(nn.Linear(n_flatten_wrist, feature_size), nn.ReLU())
            
            extractors["left_wrist_rgb"] = nn.Sequential(wrist_cnn, wrist_fc)
            self.feature_size += feature_size

        # Add right wrist camera processing with same architecture
        if "right_wrist_rgb" in sample_obs:
            right_wrist_in_channels = sample_obs["right_wrist_rgb"].shape[-1]
            right_wrist_cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=right_wrist_in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
            
            with torch.no_grad():
                n_flatten_right_wrist = right_wrist_cnn(sample_obs["right_wrist_rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            right_wrist_fc = nn.Sequential(nn.Linear(n_flatten_right_wrist, feature_size), nn.ReLU())
            
            extractors["right_wrist_rgb"] = nn.Sequential(right_wrist_cnn, right_wrist_fc)
            self.feature_size += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.feature_size += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key in ["rgb", "left_wrist_rgb", "right_wrist_rgb"]:
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)
    
# class NatureCNN(nn.Module):
#     def __init__(self, sample_obs):
#         super().__init__()
#         self.feature_size = 0
#         feature_size = 256

#         def make_nature_cnn(in_ch):
#             return nn.Sequential(
#                 nn.Conv2d(in_ch, 32, 8, 4, 0), nn.ReLU(),
#                 nn.Conv2d(32, 64, 4, 2, 0),    nn.ReLU(),
#                 nn.Conv2d(64, 64, 3, 1, 0),    nn.ReLU(),
#                 nn.Flatten(),
#             )

#         extractors = {}
#         device = sample_obs["rgb"].device
#         H, W, C = sample_obs["rgb"].shape[1:4]

#         # rgb
#         cnn = make_nature_cnn(C)
#         with torch.no_grad():
#             n_flatten = cnn(torch.zeros(1, C, H, W, device=device)).shape[1]
#         fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
#         extractors["rgb"] = nn.Sequential(cnn, fc)
#         self.feature_size += feature_size

#         # left wrist
#         if "left_wrist_rgb" in sample_obs:
#             Hw, Ww, Cw = sample_obs["left_wrist_rgb"].shape[1:4]
#             wrist_cnn = make_nature_cnn(Cw)
#             with torch.no_grad():
#                 n_flatten_wrist = wrist_cnn(torch.zeros(1, Cw, Hw, Ww, device=device)).shape[1]
#             wrist_fc = nn.Sequential(nn.Linear(n_flatten_wrist, feature_size), nn.ReLU())
#             extractors["left_wrist_rgb"] = nn.Sequential(wrist_cnn, wrist_fc)
#             self.feature_size += feature_size

#         # right wrist
#         if "right_wrist_rgb" in sample_obs:
#             Hr, Wr, Cr = sample_obs["right_wrist_rgb"].shape[1:4]
#             right_cnn = make_nature_cnn(Cr)
#             with torch.no_grad():
#                 n_flatten_r = right_cnn(torch.zeros(1, Cr, Hr, Wr, device=device)).shape[1]
#             right_fc = nn.Sequential(nn.Linear(n_flatten_r, feature_size), nn.ReLU())
#             extractors["right_wrist_rgb"] = nn.Sequential(right_cnn, right_fc)
#             self.feature_size += feature_size

#         if "state" in sample_obs:
#             state_size = sample_obs["state"].shape[-1]
#             extractors["state"] = nn.Sequential(nn.Linear(state_size, 256), nn.ReLU())
#             self.feature_size += 256

#         self.extractors = nn.ModuleDict(extractors)

#     def forward(self, observations):
#         outs = []
#         for key in ("rgb", "left_wrist_rgb", "right_wrist_rgb", "state"):
#             if key not in self.extractors or key not in observations:
#                 continue
#             x = observations[key]
#             if key.endswith("rgb"):
#                 x = x.float().permute(0, 3, 1, 2).contiguous() / 255.0
#             outs.append(self.extractors[key](x))
#         return torch.cat(outs, dim=1)
