#!/usr/bin/env python3

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch

from mobile_robot_env import PiperEnv


def test_policy(
    policy_path,
    num_episodes=3,
    deterministic=True,
    max_steps=80,
    render=True,
    gs_render=False,
    use_top_rgb=False,
    use_left_wrist_rgb=False,
    use_right_wrist_rgb=False,
    use_state=True,
    use_target=True,
    encoder="resnet_v2",
):
    """
    Test trained left arm policy by loading checkpoint and running episodes.
    Handles RGB + state observation format used in training.
    
    Args:
        policy_path: Path to policy checkpoint
        num_episodes: Number of episodes to run
        deterministic: Use deterministic policy
        max_steps: Maximum steps per episode 
        render: Enable rendering
        gs_render: Use Gaussian Splatting rendering
    """
    # Check if checkpoint file exists
    print(f"  Policy: {policy_path}")
    if not os.path.exists(policy_path):
        print(f"Error: Policy file not found: {policy_path}")
        return

    print(f"Running {num_episodes} episodes with {'deterministic' if deterministic else 'stochastic'} policy")
    print(f"Rendering: {'enabled' if render else 'disabled'}")
    print("-" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    # Create environment with rendering if requested
    env = PiperEnv(
        visualization=render,
        gs_render=gs_render,
        max_episode_length=max_steps,
        use_top_rgb=use_top_rgb,
        use_left_wrist_rgb=use_left_wrist_rgb,
        use_right_wrist_rgb=use_right_wrist_rgb,
    )
    
    # Load checkpoint and create agent
    def load_agent_from_checkpoint(checkpoint_path, env):
        """Helper function to load an agent from a checkpoint."""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print(f"✓ Checkpoint loaded: {os.path.basename(checkpoint_path)}")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            return None
        
        # Get sample observation for agent initialization
        obs, _ = env.reset()
        print('reser_over')
        
        # Import agent from the main PPO file (match training)
        if encoder == "state_target":
            from model_extractor.state_target_encoder import StateTargetEncoder as Extractor
        else:
            from model_extractor.resnet_v2 import ResNet as Extractor
        from model_policy.mlp_agent import MLP_Agent as TrainedAgent
        
        # Create a sample observation for network initialization
        sample_obs = {}
        if use_top_rgb and 'top_rgb' in obs:
            sample_obs['top_rgb'] = torch.tensor(obs['top_rgb']).unsqueeze(0)
        if use_state and 'state' in obs:
            sample_obs['state'] = torch.tensor(obs['state']).unsqueeze(0)
        if use_left_wrist_rgb and 'left_wrist_rgb' in obs:
            sample_obs['left_wrist_rgb'] = torch.tensor(obs['left_wrist_rgb']).unsqueeze(0)
        if use_right_wrist_rgb and 'right_wrist_rgb' in obs:
            sample_obs['right_wrist_rgb'] = torch.tensor(obs['right_wrist_rgb']).unsqueeze(0)
        if use_target and 'target' in obs:
            sample_obs['target'] = torch.tensor(obs['target']).unsqueeze(0)
        
        # Create agent
        visual_net = Extractor(sample_obs=sample_obs).to(device)
        agent = TrainedAgent(env.action_space, sample_obs=sample_obs, Extractor=visual_net, device=device).to(device)

        
        # Load state dict (print only mismatched layers)
        model_dict = agent.state_dict()
        compatible = {}
        mismatched = []
        for key, value in checkpoint.items():
            if key in model_dict and model_dict[key].shape == value.shape:
                compatible[key] = value
            else:
                ckpt_shape = tuple(value.shape) if hasattr(value, "shape") else None
                model_shape = tuple(model_dict[key].shape) if key in model_dict else None
                mismatched.append((key, ckpt_shape, model_shape))

        if mismatched:
            print("⚠ Checkpoint architecture mismatch detected; loading compatible layers only.")
            for key, ckpt_shape, model_shape in mismatched:
                print(f"  Mismatch: {key}  ckpt={ckpt_shape}  model={model_shape}")

        model_dict.update(compatible)
        agent.load_state_dict(model_dict, strict=False)
        print(f"✓ Loaded {len(compatible)}/{len(checkpoint)} compatible layers")
        
        agent.eval()
        return agent

    # Load the agent
    agent = load_agent_from_checkpoint(policy_path, env)
    if agent is None:
        return

    def get_action_from_obs(obs_dict, agent):
        """Convert observation dict to tensor format and get action."""
        converted = {}
        for key, value in obs_dict.items():
            if key == "top_rgb" and use_top_rgb:
                converted[key] = torch.tensor(value, dtype=torch.uint8, device=device).unsqueeze(0)
            elif key == "left_wrist_rgb" and use_left_wrist_rgb:
                converted[key] = torch.tensor(value, dtype=torch.uint8, device=device).unsqueeze(0)
            elif key == "right_wrist_rgb" and use_right_wrist_rgb:
                converted[key] = torch.tensor(value, dtype=torch.uint8, device=device).unsqueeze(0)
            elif key == "state" and use_state:
                converted[key] = torch.tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
            elif key == "target" and use_target:
                converted[key] = torch.tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            return agent.get_action(converted, deterministic=deterministic).cpu().numpy().squeeze()

    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    # Run episodes
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        print("  Testing LEFT arm fruit picking")
            
        obs, info = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        for step in range(max_steps):
            action = get_action_from_obs(obs, agent)
            
            next_obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            done = terminated or truncated
            
            if step % 10 == 0 or done:
                gripper_info = ""
                print(f"  Step {step:2d}: reward={reward:+6.3f}, total_reward={episode_reward:+7.3f}{gripper_info}")
                if info:
                    penalty_info = info.get("penalty_info", {}) or {}
                    print(
                        f"    is_success={info.get('is_success')}, "
                        f"object_fell={penalty_info.get('object_fell')}, "
                        f"object_tipped={penalty_info.get('object_tipped')}, "
                        f"table_contact={penalty_info.get('table_contact')}, "
                        f"terminated_reasons={info.get('terminated_reasons')}"
                    )
            
            if done:
                print(f"  Episode finished: {'terminated' if terminated else 'truncated'}")
                break
        
        episode_success = info.get("is_success", False)
        
        # Episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode_success:
            success_count += 1
            print(f"  ✓ Episode successful!")
        
        print(f"  Episode {episode + 1} stats: reward={episode_reward:.3f}, length={episode_length}")

    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Episodes run: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min/Max reward: {np.min(episode_rewards):.3f} / {np.max(episode_rewards):.3f}")
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")

    env.close()
    print("\nDone!")


def _str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Test trained PPO left arm policy")
    parser.add_argument(
        "--policy",
        type=str,
        default="/home/wzt/wzt/mycode/whole_body_rl/runs/whole_body_RL_try4/50.pt",
        help="Path to policy checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--deterministic", type=_str2bool, default=False,
                        help="Use deterministic policy (no action noise)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument("--render", type=_str2bool, default=True,
                        help="Enable rendering")
    parser.add_argument("--gs_render", type=_str2bool, default=False,
                        help="Use Gaussian Splatting rendering instead of MuJoCo rendering")
    parser.add_argument("--use-top-rgb", type=_str2bool, default=False,
                        help="Use top_rgb observation")
    parser.add_argument("--use-left-wrist-rgb", type=_str2bool, default=False,
                        help="Use left_wrist_rgb observation")
    parser.add_argument("--use-right-wrist-rgb", type=_str2bool, default=False,
                        help="Use right_wrist_rgb observation")
    parser.add_argument("--use-state", type=_str2bool, default=True,
                        help="Use state observation")
    parser.add_argument("--use-target", type=_str2bool, default=True,
                        help="Use target observation")
    parser.add_argument("--encoder", type=str, default="state_target",
                        choices=["resnet_v2", "state_target"],
                        help="Encoder type (must match training)")
    
    # explicit defaults are set in each argument above
    args = parser.parse_args()
    
    # Test the policy
    test_policy(
        policy_path     = args.policy,
        num_episodes    = args.episodes,
        deterministic   = args.deterministic,
        max_steps       = args.max_steps,
        render          = args.render,
        gs_render       = args.gs_render,
        use_top_rgb     = args.use_top_rgb,
        use_left_wrist_rgb = args.use_left_wrist_rgb,
        use_right_wrist_rgb = args.use_right_wrist_rgb,
        use_state       = args.use_state,
        use_target      = args.use_target,
        encoder         = args.encoder,
    )


if __name__ == "__main__":
    main()
