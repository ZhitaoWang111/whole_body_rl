import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP_Agent(nn.Module):
    def __init__(self, action_space, sample_obs, Extractor, device):
        super().__init__()
        self.feature_net = Extractor
        latent_size = self.feature_net.feature_size

        self.n_bins = 32
        self.n_actions = action_space.shape[0]
        self.action_low = torch.tensor(action_space.low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_space.high, dtype=torch.float32, device=device)
        self.temperature = 1.0

        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )

        self.trunk = nn.Sequential(
            nn.Linear(latent_size, 512), nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512), nn.GELU()
        )
        self.head = nn.Linear(512, self.n_actions * self.n_bins)
    
    def forward(self, x):
        x = self.feature_net(x)
        x = self.trunk(x)
        logits = self.head(x)
        return logits.view(-1, self.n_actions, self.n_bins)


    def get_features(self, x):
        return self.feature_net(x)
    
    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x.detach())
    
    def get_action(self, x, greedy=False):
        logits = self.forward(x)                    # (batch_size, n_actions, n_bins)
        logits = logits / max(self.temperature, 1e-6)
        log_probs = F.log_softmax(logits, dim=-1)   # (batch_size, n_actions, n_bins)
        probs = torch.exp(log_probs)                # (batch_size, n_actions, n_bins)
        if greedy:
            idx = torch.argmax(probs, dim=-1)         # (batch_size, n_actions)
        else:
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()                       # (batch_size, n_actions)
        action = idx.float()
        ratio = action / (self.n_bins - 1)
        action = self.action_low + (self.action_high - self.action_low) * ratio

        return action
    

    def get_action_and_value(self, x, action=None):
        logits = self.forward(x)                    # (batch_size, n_actions, n_bins)
        logits = logits / max(self.temperature, 1e-6)
        log_probs = F.log_softmax(logits, dim=-1)   # (batch_size, n_actions, n_bins)
        probs = torch.exp(log_probs)                # (batch_size, n_actions, n_bins)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            idx = dist.sample()                       # (batch_size, n_actions)
            action = idx.float()
            ratio = action / (self.n_bins - 1)
            action = self.action_low + (self.action_high - self.action_low) * ratio
            log_prob_idx = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)  # (batch_size, n_actions)
            log_prob_total = log_prob_idx.sum(1)  # (batch_size,)
        else:
            ratio = (action - self.action_low) / (self.action_high - self.action_low)
            idx = torch.round(ratio.clamp(0,1) * (self.n_bins - 1)).long()
            log_prob_idx = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            log_prob_total = log_prob_idx.sum(1)  # (batch_size,)

        return action, log_prob_total, dist.entropy().sum(1), self.get_value(x)
