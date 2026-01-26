"""Simple script to train a RGB PPO policy in simulation (wandb logging).

To train with rendering (single environment only):
python train_ppo_rgb.py \
  --ppo.total-timesteps 10000 \
  --ppo.render-training \
  --ppo.num-envs 1 \
  --ppo.num-eval-envs 0

To train without rendering (faster, multiple environments):
python train_ppo_rgb.py \
  --ppo.total-timesteps 100000000 \
  --ppo.num-envs 35 \
  --ppo.num-eval-envs 4 \
  --ppo.learning-rate 3e-4 \
  --ppo.num-minibatches 16 \
  --ppo.gs_render \
  --ppo.checkpoint runs/PiperEnv__ppo_rgb__1__1757156602/ckpt_301.pt

To resume training from a checkpoint:
python train_ppo_rgb.py \
  --ppo.total-timesteps 50000000 
  --ppo.num-envs 70 
  --ppo.num-eval-envs 8 
  --ppo.learning-rate 8e-5 
  --ppo.num-minibatches 16 
  --ppo.checkpoint runs/PiperEnv__ppo_rgb__1__1756392623/ckpt_726.pt

To fine-tune from a checkpoint (start fresh but with pretrained weights):
python train_ppo_rgb.py \
  --ppo.num-envs 50 
  --ppo.num-eval-envs 8 
  --ppo.num-minibatches 16 
  --ppo.checkpoint runs/PiperEnv__ppo_rgb__1__1754212457/ckpt_226.pt 
  --ppo.total-timesteps 2000000 
  --ppo.learning-rate 1e-5

Note: For large num_envs (>50), reduce num_steps to maintain reasonable batch sizes.
Recommended: batch_size = num_envs * num_steps should be 1000-8000 for optimal performance.
"""

from dataclasses import dataclass, field
import json
from typing import Optional
import tyro

# from mujoco_sim2real.ppo_rgb import PPOArgs, train
from ppo_rgb import PPOArgs, train

@dataclass
class Args:
    env_id: str = "PiperEnv"
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""
    ppo: PPOArgs = field(default_factory=PPOArgs)
    """PPO training arguments"""

def main(args: Args):
    # Sync env id into PPO args
    args.ppo.env_id = args.env_id

    # Optional env kwargs from json
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs = json.load(f)
        args.ppo.env_kwargs = env_kwargs
    else:
        print("No env kwargs json path provided, using default env kwargs with default settings")
    
    train(args=args.ppo)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
