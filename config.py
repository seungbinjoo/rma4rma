# Import required packages
import argparse
import os
import os.path as osp
from functools import partial

import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

from stable_baselines3.common.utils import set_random_seed, get_latest_run_id

def parse_args():
    """
    This function is used to parse command-line arguments provided when the script is run.
    This allows users to specify options for the RL training or evaluation run without modifying the code directly.
    Uses argparse library to define a set of possible command-line options and returns a structured 'args'
    object that contains these arguments as attributes.

    EXAMPLE:
    Suppose the script is run with the following command:
    python train.py --env-id PickCube-v1 --n-envs 10 --use_depth_base --batch_size 1024
    Then the 'args' would look like:
    args.env_id           # 'PickCube-v1'
    args.n_envs           # 10
    args.use_depth_base   # True
    args.batch_size       # 1024
    args.seed             # None (since not provided in the command line)
    args.total_timesteps  # 100_000_000_000_000 (default value)
    args.log_dir          # '/users/joo/4yp/rma4rma/logs' (default value)

    Then, the 'args' object is structured as follows:
    Namespace(
        env_id='PickCube-v1',
        n_envs=10,
        use_depth_base=True,
        batch_size=1024,
        seed=None,
        total_timesteps=100_000_000_000_000,
        log_dir='/users/joo/4yp/rma4rma/logs',
        other arguments with their default values or as provided
    )
    """

    parser = argparse.ArgumentParser(
        description="script demonstrating the use of Stable Baselines 3 with ManiSkill and RGBD observations"
    )
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="PickCube-v1",
        help="-e or --env-id specifies the environment ID, defaulting to PickCube-v1"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="panda",
        choices=["panda", "xarm7", "xmate3_robotiq"],
        help="robot to use in environment, defaulting to panda arm"
    )
    parser.add_argument(
        "--obs_mode",
        type=str,
        default="state_dict",
        help="observation mode defines the observation space; can be stat_dict, sensor_data, etc."
    )
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=50,
        help="-n or --n-envs sets the number of parallel environments, it defaults to 50 envs"
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=5000,
        help="""batch size for training:
        Agent interacts with the environment to collect experiences in the form of (state, action, reward, next_state) tuples
        which are stored in a memory buffer (replay buffer). Then, the agent updates its policy or value function (using
        algorithms like PPO, DQN, etc.). Agent samples a subset of experiences, called a batch, from the buffer to train on"""
    )
    parser.add_argument(
        "-rs",
        "--rollout_steps",
        type=int,
        default=2000,
        help="""rollout steps per env:
        Rollout steps define how long the agent interacts with the environment during one session before the collected data,
        i.e. (state, action, reward, next_state) tuples, is used for training"""
    )
    parser.add_argument(
        "-kl",
        "--target_kl",
        type=float,
        default=.05,
        help="""upper bound for the KL divergence (for PPO implementation in SB3):
        Limit the KL divergence between updates, because the clipping is not enough to prevent large update"""
    )
    parser.add_argument(
        "-cr",
        "--clip_range",
        type=float,
        default=.2,
        help="clip range for PPO (implementation in SB3)"
    )
    parser.add_argument(
        "-nep",
        "--n_epochs",
        type=int,
        default=10,
        help="number of epoch when optimizing the surrogate loss in PPO (see SB3 implementation of PPO)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to initialize training with"
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=50,
        help="max steps per episode before truncating them"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100_000_000_000_000,
        help="Total timesteps for training"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/users/joo/4yp/rma4rma/logs",
        help="directory path for where logs, checkpoints, and videos are saved"
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="PPO",
        help="model name, e.g., PPO, PPO-pc0-bs400_1, ..."
        # specify log_name in --continue_training argument to resume the logging
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        help="path to SB3 model for evaluation"
        # specify log_name to continue training from checkpoint
        # e.g., model_320000_steps.zip, latest.zip
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="whether to only evaluate policy"
        # The action="store_true" parameter in parser.add_argument() means that the argument
        # acts as a flag, and when this flag is provided in the command line,
        # the corresponding value in the parsed arguments (args) will be set to True
    )
    parser.add_argument(
        "-ct",
        "--continue_training",
        action="store_true",
        help="continue training from checkpoint"
        # for continue training, specify:
        # - log_name for logging to the correct dir,
        # - ckpt_name for loading the correct model
    )
    parser.add_argument(
        "--policy_arch",
        type=list,
        default=[512, 256, 128], # config in hora
        help="policy network architecture"
    )
    parser.add_argument(
        "--randomized_training",
        action="store_true",
        help="whether to randomize the training environment"
    )
    parser.add_argument(
        "-on",
        "--obs_noise",
        action="store_true",
        help="whether to add noise to the observations."
    )
    parser.add_argument(
        "--lr_schedule",
        default=0,
        type=int,
        help="whether to use learning rate schedule, if not specified"
    )
    parser.add_argument(
        "--clip_range_schedule",
        default=1,
        type=int,
        help="whether to use clip range schedule (for PPO), if not specified"
    )
    parser.add_argument(
        "-ae",
        "--anneal_end_step",
        type=float,
        default=1e7,
        help="step number where annealing of learning rate and clip range ends"
    )
    parser.add_argument(
        "--adaptation_training",
        action="store_true",
        help="""perform stage 2, i.e. adaptation training, when this tag is specified
        when using this tag, `log_dir`, `log_name`, `ckpt_name` must be specified"""
    )
    parser.add_argument(
        "--transfer_learning",
        action="store_true",
        help="""Perform transfer learning on another env specified by env-id
        When used, specify `log_dir`, `log_name`, `ckpt_name` to choose the base model"""
    )
    parser.add_argument(
        "--use_depth_adaptation",
        action="store_true",
        help="""use depth information in the observation.
        This entails using rgbd observation and having a CNN feature extractor."""
    )
    parser.add_argument(
        "--use_depth_base",
        action="store_true",
        help="doesn't use object position and privileged information."
    )
    parser.add_argument(
        "--use_prop_history_base",
        action="store_true",
        help="doesn't use object position and privileged information."
    )
    parser.add_argument(
        "--ext_disturbance",
        action="store_true",
        help="whether to add external disturbance force to the environment."
    )
    parser.add_argument(
        "--inc_obs_noise_in_priv",
        action="store_true",
        help="add obs noise as part of the privileged observation."
    )
    parser.add_argument(
        "--expert_adapt",
        action="store_true",
        help="whether we use trained adaptation module (?)"
    )
    parser.add_argument(
        "--without_adapt_module",
        action="store_true",
        help="no adaptation module used"
    )
    parser.add_argument(
        "--only_DR",
        action="store_true",
        help="only domain randomization"
    )
    parser.add_argument(
        "--sys_iden",
        action="store_true",
        help="system identification: whether we want to predict the priviliged information (?)"
    )
    parser.add_argument(
        "--auto_dr",
        action="store_true",
        help="autmatic domain randomization"
    )
    parser.add_argument(
        "--obj_emb_dim",
        default=32,
        type=int,
        help="object embedding dimension"
    )
    parser.add_argument(
        "--eval_model_id",
        default="002_master_chef_can",
        help="The model to eval the model on"
    )
    parser.add_argument(
        "--compute_adaptation_loss",
        action="store_true",
        help="""perform stage 2, i.e. adaptation training, when the tag is specified
        when using this, `log_dir`, `log_name`, `ckpt_name` must be specified"""
    )

    # return structured 'args' object that contains all above arguments as attributes
    args = parser.parse_args()
    return args

# Dictionary mapping from environment name to its abbreviation (used in config_log_path method below)
env_name_to_abbrev = {
    'PickCube-v0': 'pc0',
    'PickCube-v1': 'pc',
    'StackCube-v1': 'sc',
    'PickSingleYCB-v1': 'ps',
    'PegInsertionSide-v1': 'pi',
    'TurnFaucet-v1': 'tf',
}

# Configuring paths and logging: set up the paths where logs, checkpoints, and videos for training runs will be saved.
# Method below uses the parameters provided in the 'args' object to customize the directory structure and file naming.
def config_log_path(args):
    # config save, load path
    log_dir = args.log_dir
    ckpt_path = None

    # if we are continuing training (rather than starting from scratch)
    if args.continue_training:
        log_name = f"{args.log_name}"
        ckpt_path = osp.join(log_dir, log_name, 'ckpt', args.ckpt_name)

    # if we are training adaptation module or transfer learning or evaluating policy
    elif args.adaptation_training or args.transfer_learning or args.eval:
        log_name = f"{args.log_name}"
        ckpt_path = osp.join(log_dir, log_name, 'ckpt', args.ckpt_name)
        if args.adaptation_training:
            log_name = f"{args.log_name}_stage2"
            if args.use_depth_adaptation:
                log_name += "_dep"
        elif args.transfer_learning:
            log_name = f"{args.log_name}_to-{env_name_to_abbrev[args.env_id]}"
        latest_run_id = get_latest_run_id(args.log_dir, log_name)
        log_name = f"{log_name}_{latest_run_id + 1}"
    
    # if we are training and starting from scratch (rather than continuing training)
    else:
        log_name = f"{args.log_name}-{env_name_to_abbrev[args.env_id]}"+\
                    f"-bs{args.batch_size}-rs{args.rollout_steps}"+\
                    f"-kl{args.target_kl}-neps{args.n_epochs}"+\
                    f"-cr{args.clip_range}-lr_scdl{args.lr_schedule}"+\
                    f"-cr_scdl{args.clip_range_schedule}"+\
                    f"-ms{args.max_episode_steps}"+\
                    f"-incObsN{int(args.inc_obs_noise_in_priv)}"
        if args.use_depth_base: log_name += "-dep"
        if args.use_prop_history_base: log_name += "-prop"
        if args.only_DR: log_name += "-onlyDR"
        if args.auto_dr: log_name += "-ADR"
        if args.sys_iden: log_name += "-SyId"
        latest_run_id = get_latest_run_id(args.log_dir, log_name)
        log_name = f"{log_name}_{latest_run_id + 1}"
    print(f"## Saving to {log_name}")

    # record dir is: <log_dir>/<log_name>/video
    record_dir = osp.join(log_dir, log_name, "video")
    ckpt_dir = osp.join(log_dir, log_name, "ckpt")
    tb_path_root = osp.join(log_dir, log_name)
    if not osp.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    return record_dir, ckpt_dir, ckpt_path, tb_path_root

# Configuring and initializing the environments: sets up both the training and evaluation environments
def config_envs(args, record_dir):
    # store required attributes from 'arg'
    env_id = args.env_id
    num_envs = args.n_envs
    max_episode_steps = args.max_episode_steps

    # configure observation mode (https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)
    if args.use_depth_adaptation or args.use_depth_base:
        obs_mode = "rgbd"
    else:
        obs_mode = "state_dict"

    # Configure controller (https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)
    # Summary: controller takes the relative movement of the end-effector as input, and uses inverse kinematics
    # to convert input actions to target positions of robot joints. The robot uses a PD controller to drive motors
    # to achieve target joint positions
    control_mode = "pd_ee_delta_pose"
    
    # Dense reward: the reward function provides signals at each time step or action taken,
    # often guiding the agent by reflecting incremental progress toward the goal
    # Sparse reward: the agent receives feedback only at specific milestones or at the end of an
    # episode, making the reward function more binary or less informative.
    reward_mode = "normalized_dense"

    # set random seed if there isn't a random see provided
    if args.seed is not None:
        set_random_seed(args.seed)

    # EVALUATION ENVIRONMENT SETUP
    # if we are in policy evaluation phase, the record_dir is updated to include an eval subdirectory
    if args.eval:
        record_dir = osp.join(record_dir, "eval")

    # the model to eval the model on
    model_ids = args.eval_model_id

    # eval_env_kwargs: dictionary of eval environment parameters
    eval_env_kwargs = dict(
        randomized_training=args.randomized_training,
        robot=args.robot,
        obs_noise=args.obs_noise,
        ext_disturbance=args.ext_disturbance,
        inc_obs_noise_in_priv=args.inc_obs_noise_in_priv,
        test_eval=args.eval,
        sim_freq=120, # TODO: what should the sim freq be?
    )
    
    # eval environment setup
    eval_env = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        reward_mode=reward_mode,
        num_envs=1,
        # **eval_env_kwargs
    )

    # FlattenRGBDObservationWrapper: concatenates all the RGB and Depth images into a single image with combined channels,
    # and concatenates all state data into a single array so that the observation space becomes a simple dictionary
    # composed of a state key and a rgbd key.
    if args.use_depth_adaptation or args.use_depth_base:
        eval_env = FlattenRGBDObservationWrapper(eval_env)

    # # FlattenActionSpaceWrapper: flatten a dictionary action space into a flat array action space
    # eval_env = FlattenActionSpaceWrapper(eval_env)

    # ManiSkillSB3VectorEnv: a wrapper to make ManiSkill parallel simulation compatible with SB3 VecEnv
    # and auto adds the monitor wrapper
    eval_env = ManiSkillSB3VectorEnv(eval_env)

    eval_env.seed(args.seed)
    eval_env.reset()

    # env_kwargs: dictionary of training environment parameters
    env_kwargs = dict(
        randomized_training=args.randomized_training,
        robot=args.robot,
        auto_dr=args.auto_dr,
        obs_noise=args.obs_noise,
        ext_disturbance=args.ext_disturbance,
        inc_obs_noise_in_priv=args.inc_obs_noise_in_priv,
        test_eval=args.eval,
        sim_freq=120,
    )

    if args.eval:
        env = eval_env

    # TRAINING ENVIRONMENT SETUP
    else:
        # Create vectorized environments for training
        env = gym.make(
            env_id,
            num_envs=num_envs,
            obs_mode=obs_mode,
            control_mode=control_mode,
            max_episode_steps=max_episode_steps,
            # **env_kwargs
        )
        
        # FlattenRGBDObservationWrapper: concatenates all the RGB and Depth images into a single image with combined channels,
        # and concatenates all state data into a single array so that the observation space becomes a simple dictionary
        # composed of a state key and a rgbd key.
        if args.use_depth_adaptation or args.use_depth_base:
            env = FlattenRGBDObservationWrapper(env)

        # # FlattenActionSpaceWrapper: flatten a dictionary action space into a flat array action space
        # env = FlattenActionSpaceWrapper(env)

        # ManiSkillSB3VectorEnv: a wrapper to make ManiSkill parallel simulation compatible with SB3 VecEnv
        # and auto adds the monitor wrapper
        env = ManiSkillSB3VectorEnv(env)
        
        env.seed(args.seed)
        env.reset()

    return env, eval_env
