from collections import defaultdict
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from tqdm import trange

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None

from model_extractor.resnet_v2 import ResNet as Extractor
# chose only one of agents
from model_policy.mlp_agent import MLP_Agent as Agent


# ManiSkill specific imports - removed, using custom environment
# Import our custom environment
# from mujoco_sim2real.mobile_robot_env import PiperEnv
from mobile_robot_env import PiperEnv


def _make_env(render: bool, gs_render: bool, max_episode_length: int) -> gym.Env:
    render_mode = "human" if render else None
    env = PiperEnv(render_mode=render_mode, gs_render=gs_render, max_episode_length=max_episode_length)
    return gym.wrappers.RecordEpisodeStatistics(env)


def _convert_obs(obs_dict, device):
    """Convert observation dict to torch tensors on the target device."""
    converted = {}
    for key, value in obs_dict.items():
        if torch.is_tensor(value):
            if key in ["rgb", "left_wrist_rgb", "right_wrist_rgb"]:
                converted[key] = value.to(dtype=torch.uint8, device=device)
            elif key in ["state", "tgt_delta"]:
                converted[key] = value.to(dtype=torch.float32, device=device)
        else:
            if key in ["rgb", "left_wrist_rgb", "right_wrist_rgb"]:
                converted[key] = torch.as_tensor(value, dtype=torch.uint8, device=device)
            elif key in ["state", "tgt_delta"]:
                converted[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return converted


def _init_wandb(args, run_name: str):
    if not args.track:
        return
    if wandb is None:
        raise ImportError("wandb is not installed. Please `pip install wandb` or set --ppo.track False.")
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
    )


def _log_wandb(metrics: dict, args, step: int):
    if args.track and wandb is not None:
        wandb.log(metrics, step=step)

@dataclass
class PPOArgs:
    exp_name: Optional[str] = "whole_body_RL"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    track: bool = True
    """if toggled, enable wandb logging"""
    wandb_project_name: str = "PiperEnv-RGB-PPO"
    """wandb project name"""
    wandb_entity: Optional[str] = None
    """wandb entity (team)"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_training: bool = False
    """if toggled, render the environment during training"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_kwargs: dict = field(default_factory=dict)
    """extra environment kwargs to pass to the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    # === 常用训练超参（与启动命令对应，统一放一起便于修改）===
    total_timesteps: int = 2000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel environments"""
    num_eval_envs: int = 0
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 40
    """the number of steps to run in each environment per policy rollout
    NOTE: batch_size = num_envs * num_steps. For large num_envs (>50), consider reducing num_steps to 10-50
    to maintain reasonable batch sizes and update frequencies."""
    num_eval_steps: int = 80
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = None
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.95
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatch: int = 16
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.3
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    num_episodes_per_eval_env: int = 3
    """number of episodes to run per evaluation environment during evaluation"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    gs_render: bool = False



class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)





def train(args: PPOArgs):
    '''
    需要给入参数: num_envs 环境数、num_steps 每个环境的步数、num_minibatch 小批量数
    '''
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatch)

    if args.minibatch_size < 2:
        args.num_minibatch = max(1, args.batch_size // 2)
        args.minibatch_size = int(args.batch_size // args.num_minibatch)
        print(f"Adjusted num_minibatch to {args.num_minibatch} to ensure minibatch_size >= 2")
    args.num_iterations = args.total_timesteps // args.batch_size

    # === experiment name ===  
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # === seeding ===
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # === env setup ===
    train_render = args.render_training and args.num_envs == 1
    if args.render_training and args.num_envs > 1:
        print(f"WARNING: Rendering requested but num_envs={args.num_envs} > 1. Rendering disabled to avoid multiple windows.")
        print("To enable rendering, use --ppo.num-envs 1")
        train_render = False
    elif args.render_training:
        print("Rendering enabled for training environment")
    
    # 初始化环境
    max_episode_length = args.num_steps
    '''
    num_steps < max_episode_steps: 更快更新/更省显存
    num_steps ≥ max_episode_steps: rollout 内看到完整 episode、减少 bootstrap 偏差
    '''

    if args.num_envs > 0:
        envs = gym.vector.SyncVectorEnv(
            [lambda: _make_env(train_render, args.gs_render, max_episode_length) for _ in range(args.num_envs)]
        )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


    _init_wandb(args, run_name)
    # === rollout buffers ===
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    final_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # === rollout start ===
    global_step = 0
    start_time = time.time()
    if args.num_envs > 0:
        next_obs, _ = envs.reset(seed=args.seed)
    next_obs = _convert_obs(next_obs, device)
    next_done = torch.zeros(args.num_envs, device=device)
    
    # 打印 训练 信息
    print(f"==== training config INFO ====")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"=================================")
    
    # 打印 observation space 信息
    print("==== OBSERVATION SPACE INFO ====")
    print(f"Environment observation space: {envs.single_observation_space}")
    print(f"Sample observation keys: {list(next_obs.keys())}")
    for key, value in next_obs.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    print("=================================")

    # === agent / optimizer ===
    Encoder = Extractor(sample_obs=next_obs).to(device)
    agent = Agent(envs.single_action_space, sample_obs=next_obs, Extractor=Encoder, device=device).to(device)

    # 若存在视觉backbone，则冻结其权重
    for p in agent.feature_net.backbone.parameters():
        p.requires_grad = False
    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=1e-4, eps=1e-5)

    # 是否加载 checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        agent.load_state_dict(torch.load(args.checkpoint))
        print("✓ Resuming training from checkpoint")


    cumulative_times = defaultdict(float)
    for iteration in trange(1, args.num_iterations + 1, desc="Training Iterations"):

        # Reset bootstrap buffer each iteration to avoid stale values
        final_values.zero_()
        rollout_time = time.perf_counter()

        for step in range(0, args.num_steps):

            global_step += args.num_envs

            # save obs done
            obs[step] = next_obs
            dones[step] = next_done

            # 1. model forward
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            # action logP
            actions[step] = action
            logprobs[step] = logprob

            # 2. Env step
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # np to tensor
            next_obs = _convert_obs(next_obs, device)
            terminations = torch.tensor(terminations, dtype=torch.bool, device=device)
            truncations = torch.tensor(truncations, dtype=torch.bool, device=device)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            
            # reward shaping
            reward_tensor = torch.tensor(reward, device=device, dtype=torch.float32).reshape(args.num_envs)
            rewards[step] = reward_tensor * args.reward_scale

            # 3. 记录每个 step 的 reward 细节（wandb）
            if 'reward_dict' in infos:
                reward_info = infos['reward_dict']
                not_done_rewards = [r for r in reward_info if r is not None]

                if len(not_done_rewards) > 0:
                    keys = set()
                    for r_dict in not_done_rewards:
                        keys.update(r_dict.keys())
                    metrics = {}
                    for k in sorted(keys):
                        vals = [r_dict[k] for r_dict in not_done_rewards if k in r_dict]
                        if len(vals) > 0:
                            metrics[f"reward/{k}"] = float(np.mean(vals))
                    if metrics:
                        _log_wandb(metrics, args, global_step)

            # final_info  有两种情况会出现
            # 1. terminations 终止 任务失败
            # 2. truncations 截断  达到horizon
            # 不是只在最后会出现 final_info, 每个 env 只要 done 就会出现
            # 如果 done 了，next_obs是reset后的obs， final_obs是done后step出的obs
            # final_obs 中包含这一 step 所有（不管done了几个）的obs
            if "final_info" in infos:
                final_info = infos["final_info"]
                final_observation = infos["final_observation"]     
                done_mask = infos["_final_info"]

                done_infos = np.asarray(final_info, dtype=object)[np.asarray(done_mask, dtype=bool)].tolist()
                done_obs = np.asarray(final_observation, dtype=object)[np.asarray(done_mask, dtype=bool)].tolist()

                # 4. 统计每个episode的 长度， mean_reward， 时间 
                r_list = np.array([fi["episode"]["r"] for fi in done_infos], dtype=np.float32)
                l_list = np.array([fi["episode"]["l"] for fi in done_infos], dtype=np.float32)
                t_list = np.array([fi["episode"]["t"] for fi in done_infos], dtype=np.float32)
                _log_wandb(
                    {
                        "train/episode_reward": r_list.mean(),
                        "train/episode_length": l_list.mean(),
                        "train/episode_time": t_list.mean(),
                    },
                    args,
                    global_step,
                )

                # 5. 计算 final_values
                final_obs = {}
                for key in done_obs[0].keys():
                    final_obs[key] = torch.from_numpy(np.stack([obs[key] for obs in done_obs], axis=0)).to(device)
                final_obs = _convert_obs(final_obs)
                with torch.no_grad():
                    bootstrap = agent.get_value(final_obs)
                    
                    # terminations 和 truncations处的bootstrap都要保留
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] =  bootstrap.view(-1)

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        

        # === GAE / returns ===
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            last_gae = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done  # if not done, next_not_done=1
                    nextvalues = next_value
                else: 
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                # done :        real_next_values = final_values
                # not done :    real_next_values = nextvalues


                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = last_gae = delta + args.gamma * args.gae_lambda * next_not_done * last_gae # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # === flatten rollout ===
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # === PPO update ===
        agent.train()
        b_idxs = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_idxs)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_idxs = b_idxs[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_idxs], b_actions[mb_idxs])

                # 概率比率
                logratio = newlogprob - b_logprobs[mb_idxs]
                ratio = logratio.exp()

                # 统计量： 两种近似KL， ratio裁剪
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # if args.target_kl is not None and approx_kl > args.target_kl:
                #     break

                # Policy loss
                mb_advantages = b_advantages[mb_idxs]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                loss_1 = mb_advantages * ratio
                loss_2 = mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                policy_loss = -torch.min(loss_1, loss_2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_idxs]) ** 2
                    v_clipped = b_values[mb_idxs] + torch.clamp(
                        newvalue - b_values[mb_idxs],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_idxs]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_idxs]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # PPO loss
                loss = policy_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # if args.target_kl is not None and approx_kl > args.target_kl:
            #     break

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # 解释方差-->尺度无关
        # EV≈1：预测几乎解释了全部方差（很好）。
        # EV≈0：预测与常数基线差不多（不好）。
        # EV<0：预测比用𝑦当常数预测还差（很差）
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # === logging ===
        _log_wandb(
            {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": policy_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
            },
            args,
            global_step,
        )

        print("SPS:", int(global_step / (time.time() - start_time)))
        _log_wandb(
            {
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "time/step": global_step,
                "time/update_time": update_time,
                "time/rollout_time": rollout_time,
                "time/rollout_fps": args.num_envs * args.num_steps / rollout_time,
            },
            args,
            global_step,
        )
        for k, v in cumulative_times.items():
            _log_wandb({f"time/total_{k}": v}, args, global_step)
        _log_wandb(
            {"time/total_rollout+update_time": cumulative_times["rollout_time"] + cumulative_times["update_time"]},
            args,
            global_step,
        )
        
        if args.save_model and not args.evaluate and iteration % 40 == 0:
            model_path = f"runs/{run_name}/{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"Final model saved to {model_path}")

    envs.close()
    if args.track and wandb is not None:
        wandb.finish()
