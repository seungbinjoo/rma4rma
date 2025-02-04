from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Import classes and methods needed for algorithm
from algo.models import *
from algo.misc import DictArray, calculate_flattened_dim
from task.pick_single_ycb_rma import PickSingleYCBEnvRMA

# Arguments for the experiment in ManiSkill3
@dataclass
class Args: # TODO: get rid of args that are not needed for adaptation training
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    base_policy_checkpoint: Optional[str] = None
    """path to trained base policy checkpoint"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Algorithm specific arguments
    phase: str = "AdaptationTraining"
    """"whether we are in 'PolicyTraining', 'AdaptationTraining', or 'Evaluation' phase"""
    env_id: str = "PickSingleYCBRMA-v1"
    """the id of the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    eval_freq: int = 1e5
    """evaluation frequency in terms of steps"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.phase == "AdaptationTraining"

    # name experiment
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(obs_mode="rgbd", render_mode=args.render_mode, sim_backend="gpu", phase=args.phase)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # For multi-agent settings, action space is a dictionary, so we must flatten the action spaces
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    
    # Configure saving videos
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    
    # ManiSkill-specific wrapper for vectorized environments
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Print env setup details
    envs.unwrapped.print_sim_details()
    
    # Logging in tensorboard
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger = Logger(log_wandb=args.track, tensorboard=writer)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    eps_returns = torch.zeros(args.num_envs, dtype=torch.float, device=device)

    # instantiate new agent that includes adaptation_net
    observation_space_dim = calculate_flattened_dim(envs.single_observation_space)
    action_space_dim = calculate_flattened_dim(envs.single_action_space)
    agent = Agent(envs, observation_space_dim, action_space_dim, env_id=args.env_id, phase=args.phase).to(device)
    # load trained env_encoder and policy (from base policy training) into agent
    if args.base_policy_checkpoint:
        agent.load_state_dict(torch.load(args.base_policy_checkpoint))
    # freeze env_encoder and policy network (anything that is not adaptation_net)
    for name, param in agent.named_parameters():
        if "adaptation_net" not in name:
            param.requires_grad = False
    # set optimizer to minimize only over adaptation_net parameters
    optimizer = optim.Adam(agent.adaptation_net.parameters(), lr=args.learning_rate, eps=1e-5)

    action_space_low, action_space_high = torch.from_numpy(envs.single_action_space.low).to(device), torch.from_numpy(envs.single_action_space.high).to(device)
    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)
    
    # adaptation training
    step = 0
    while step < args.total_timesteps:
        # periodically test the agent in the evaluation environment to measure performance
        if step % args.eval_freq == 1:
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if args.evaluate:
                break
        if args.save_model and step % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{step}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        # insert action into buffer
        action  = agent.get_action(next_obs)
        next_obs = agent.action_to_proprio(next_obs)
        # get ground truth and predicted env embeddings from frozen base policy and adaptation module, respectively
        env_embedding_gt, env_embedding_pred = agent.get_env_embeddings(next_obs)
        # compute adaptation loss and backpropagate
        loss = torch.mean((env_embedding_pred - env_embedding_gt) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # next step for environment
        next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
        # log adaptation training loss
        logger.add_scalar("adaptation_training_loss", loss, step)
        step += 1
