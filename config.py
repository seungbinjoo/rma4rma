# Import required packages
import argparse
import os
import os.path as osp
from functools import partial

from stable_baselines3.common.utils import set_random_seed, get_latest_run_id
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from mani_skill2.vector.wrappers.sb3 import SB3VecEnvWrapper
from mani_skill2.vector import VecEnv
from mani_skill2.vector import make as make_vec_env
from mani_skill2.utils.wrappers.sb3 import ContinuousTaskWrapper,\
                                           SuccessInfoWrapper

from algo.misc import make_env, ManiSkillRGBDVecEnvWrapper

# PURPOSE of config.py file: serves as a centralized configuration script for defining and managing various parameters and settings for training and running reinforcement learning (RL) experiments
# 1. Parsing Command-Line Arguments
# The parse_args() function in config.py is used to parse command-line arguments provided when the script is run. This allows users to specify options for the RL training or evaluation run without modifying the code directly.
# 2. Defining Default Values and Parameter Options
# The argparse.ArgumentParser is set up with various argument options, including defaults for each parameter. This ensures that even if a user does not specify certain parameters, the script uses pre-defined defaults.
# 3. Improving Code Modularity
# By keeping the configuration separate in a file like config.py, the codebase is more modular. This approach prevents clutter in the main training and evaluation scripts (train.py, evaluate.py, etc.), making them easier to read and maintain.
# 4. Configuring Paths and Logging
# The config.py file may include functions (like config_log_path()) that help set up paths for saving logs, checkpoints, and results. This ensures that experiments can be organized in a consistent structure
# 5. Creating environment configuations

# this function is called in main.py script
# purpose: parse the command-line arguments passed when running the script
# e.g. 'python main.py -n 50 -bs 5000 -rs 2000 ...' --> convert arugments here to objects we can work with
# this function uses Python’s argparse library to define a set of possible command-line options
# and returns a structured args object that contains these arguments as attributes
# note: parsing means reading these raw strings from the command line (--env-id PickCube-v1, --n-envs 10), analyzing them, and converting them into a Python object (in this case, args) that makes it easy to access the data as attributes
def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Simple script demonstrating how to use Stable Baselines 3 with ManiSkill2 and RGBD Observations"
    )
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1") # -e or --env-id specifies the environment ID, defaulting to "PickCube-v1"
    # CHANGE TO BIMANUAL ROBOT
    parser.add_argument("--robot",
                        type=str,
                        default="panda",
                        choices=["panda", "xarm7", "xmate3_robotiq"])
    # parser.add_argument("-e", "--env-id", type=str, default="PickSingleYCB-v1")
    parser.add_argument("--obs_mode", type=str, default="state_dict")
    parser.add_argument( # e.g. -n or --n-envs sets the number of parallel environments --> it defaults to 50 and helps the user run training across multiple environments simultaneously
        "-n",
        "--n-envs",
        type=int,
        default=50,
        help="number of parallel envs to run.",
    )
    # In RL, the agent interacts with the environment to collect experiences in the form of (state, action, reward, next_state) tuples.
    # These experiences are stored in a memory buffer (replay buffer) or collected in real-time
    # When the agent updates its policy or value function (using algorithms like PPO, DQN, etc.), it doesn’t always use all the data it has collected at once.
    # Instead, it samples a subset of experiences, called a batch, from the buffer to train on.
    # This parameter specifies how many of those experiences are used in one training iteration
    # For example, if batch_size = 32, then each time the agent updates its model, it samples 32 experiences from the replay buffer or from recent rollouts and uses them to compute the loss and update the model’s weights.
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=5000,
        help="batch size for training",
    )
    # The agent starts in a given state and performs actions in the environment, transitioning from state to state while collecting data such as the current state, action taken, reward received, and the next state.
    # Rollout steps define how long the agent interacts with the environment during one session before the collected data is used for training
    # For example, if rollout_steps = 2000, the agent will take 2000 actions and collect 2000 experiences in the form of (state, action, reward, next_state) tuples before stopping to process and learn from them.
    # fter collecting data for the specified number of rollout steps, the data is used to update the agent’s policy or value function. This batch of collected experiences can be used directly or stored in a replay buffer for sampling later.
    parser.add_argument(
        "-rs",
        "--rollout_steps",
        type=int,
        default=2000,
        help="rollout steps per env",
    )
    parser.add_argument(
        "-kl",
        "--target_kl",
        type=float,
        default=.05,
        help="upper bound for the KL divergence",
    )
    parser.add_argument(
        "-cr",
        "--clip_range",
        type=float,
        default=.2,
        help="clip range for PPO",
    )
    parser.add_argument(
        "-nep",
        "--n_epochs",
        type=int,
        default=10,
        help=
        "number of parallel envs to run. Note that increasing this does not increase rollout size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=50,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100_000_000_000_000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/users/joo/4yp/rma4rma/logs",
        help="path for where logs, checkpoints, and videos are saved",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="PPO",
        help="model name, e.g., PPO, PPO-pc0-bs400_1, ..."
        # specify log_name in continue training to resume the logging
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        help="path to sb3 model for evaluation"
        # specify log_name to continue training from checkpoint
        # e.g., model_320000_steps.zip, latest.zip
    )
    parser.add_argument("--eval",
                        action="store_true",
                        help="whether to only evaluate policy")
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
        # "--policy_arch", type=list, default=[256, 256],
        "--policy_arch",
        type=list,
        default=[512, 256, 128],  # config in hora
        help="policy network architecture")
    # The action="store_true" parameter in parser.add_argument() means that the argument acts as a flag, and when this flag is provided in the command line,
    # the corresponding value in the parsed arguments (args) will be set to True
    parser.add_argument("--randomized_training",
                        action="store_true",
                        help="whether to randomize the training environment")
    parser.add_argument("-on",
                        "--obs_noise",
                        action="store_true",
                        help="whether to add noise to the observations.")
    parser.add_argument(
        "--lr_schedule",
        default=0,
        type=int,
        help="whether to use learning rate schedule, if not specified")
    parser.add_argument(
        "--clip_range_schedule",
        default=1,
        type=int,
        help="whether to use learning rate schedule, if not specified")
    parser.add_argument(
        "-ae",
        "--anneal_end_step",
        type=float,
        default=1e7,
        help="end step for annealing learning rate and clip range",
    )
    parser.add_argument(
        "--adaptation_training", action="store_true",
        help="perform stage 2, adaptation training when the tag is specified"+\
        "when using this, `log_dir`, `log_name`, `ckpt_name` must be specified"
    )
    parser.add_argument(
        "--transfer_learning", action="store_true",
        help="perform transfer learning on another env specified by env-id."+\
        "When used, specify `log_dir`, `log_name`, `ckpt_name` to choose the"+\
        "base model."
    )
    parser.add_argument(
        "--use_depth_adaptation", action="store_true",
        help="use depth information in the observation. This entails using rgbd"+\
             "observation and have a CNN feature extractor."
    )
    parser.add_argument(
        "--use_depth_base",
        action="store_true",
        help="doesn't use object position and privileged information.")
    parser.add_argument(
        "--use_prop_history_base",
        action="store_true",
        help="doesn't use object position and privileged information.")
    parser.add_argument(
        "--ext_disturbance",
        action="store_true",
        help="whether to add external disturbance force to the environment.")
    parser.add_argument(
        "--inc_obs_noise_in_priv",
        action="store_true",
        help="add obs noise as part of the privileged observation.")
    parser.add_argument(
        "--expert_adapt",
        action="store_true",
    )
    parser.add_argument(
        "--without_adapt_module",
        action="store_true",
    )
    parser.add_argument(
        "--only_DR",
        action="store_true",
    )
    parser.add_argument(
        "--sys_iden",
        action="store_true",
    )
    parser.add_argument(
        "--auto_dr",
        action="store_true",
    )
    parser.add_argument("--obj_emb_dim", default=32, type=int, help="")
    parser.add_argument("--eval_model_id",
                        default="002_master_chef_can",
                        help="The model to eval the model on")
    parser.add_argument(
        "--compute_adaptation_loss", action="store_true",
        help="perform stage 2, adaptation training when the tag is specified"+\
        "when using this, `log_dir`, `log_name`, `ckpt_name` must be specified"
    )

    # Suppose the script is run with the following command: python train.py --env-id PickCube-v1 --n-envs 10 --use_depth_base --batch_size 1024
    # Then the 'args' would look like:
    # args.env_id           # 'PickCube-v1'
    # args.n_envs           # 10
    # args.use_depth_base   # True
    # args.batch_size       # 1024
    # args.seed             # None (since not provided in the command line)
    # args.total_timesteps  # 100_000_000_000_000 (default value)
    # args.log_dir          # '/users/joo/4yp/rma4rma/logs' (default value)

    # The 'args' object is structured as follows:
    # Namespace(
    #     env_id='PickCube-v1',
    #     n_envs=10,
    #     use_depth_base=True,
    #     batch_size=1024,
    #     seed=None,
    #     total_timesteps=100_000_000_000_000,
    #     log_dir='/users/joo/4yp/rma4rma/logs',
    #     # Other arguments with their default values or as provided
    # )

    args = parser.parse_args()
    return args


env_name_to_abbrev = {
    'PickCube-v0': 'pc0',
    'PickCube-v1': 'pc',
    'StackCube-v1': 'sc',
    'PickSingleYCB-v1': 'ps',
    'PegInsertionSide-v1': 'pi',
    'TurnFaucet-v1': 'tf',
}

# The config_log_path function is responsible for setting up the paths where logs, checkpoints, and videos for training runs will be saved.
# It uses the parameters provided in the args object to customize the directory structure and file naming.
def config_log_path(args):
    # ---- config save, load path
    log_dir = args.log_dir
    ckpt_path = None
    if args.continue_training:
        log_name = f"{args.log_name}"
        ckpt_path = osp.join(log_dir, log_name, 'ckpt', args.ckpt_name)
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

# The config_envs function is responsible for configuring and initializing the environments used in the reinforcement learning (RL) experiment.
# It customizes the environment based on the parameters passed through args and sets up both the training and evaluation environments.
def config_envs(args, record_dir):
    env_id = args.env_id
    num_envs = args.n_envs
    max_episode_steps = args.max_episode_steps
    # more info on observation modes --> https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html
    if args.use_depth_adaptation or args.use_depth_base:
        # This observation mode has the same data format as the sensor_data mode, but all sensor data from cameras are replaced with the following structure
        # sensor_data:
        # {sensor_uid}:
        # If the data comes from a camera sensor:
        # rgb: [H, W, 3], torch.uint8, np.uint8. RGB.
        # depth: [H, W, 1], torch.int16, np.uint16. The unit is millimeters. 0 stands for an invalid pixel (beyond the camera far).
        obs_mode = "rgbd"
    else:
        # The observation is a dictionary of states. It usually contains privileged information such as object poses. It is not supported for soft-body tasks.
        # agent: robot proprioception (return value of a task’s _get_obs_agent function)
        # qpos: [nq], current joint positions. nq is the degree of freedom.
        # qvel: [nq], current joint velocities
        # controller: controller states depending on the used controller. Usually an empty dict.
        # extra: a dictionary of task-specific information, e.g., goal position, end-effector position. This is the return value of a task’s _get_obs_extra function
        obs_mode = "state_dict"

    # more info on controllers here --> https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html
    # Summary: controller takes the relative movement of the end-effector as input, and uses inverse kinematics to convert input actions to target positions of robot joints.
    # The robot uses a PD controller to drive motors to achieve target joint positions
    control_mode = "pd_ee_delta_pose"
    
    # dense reward:
    # The agent receives frequent and detailed feedback throughout its interaction with the environment.
    # The reward function provides signals at each time step or action taken, often guiding the agent by reflecting incremental progress toward the goal
    # sparse reward:
    # The agent receives feedback only at specific milestones or at the end of an episode, making the reward function more binary or less informative.
    # The reward often only indicates success or failure (e.g., achieving a goal or not).
    # examples:
    # Dense Reward: The arm receives a reward based on how close it moves to the object, how well it grips the object, and how stable it holds it. Each step that brings the arm closer to success adds a small reward.
	# Sparse Reward: The arm only receives a reward if it successfully picks up the object. There is no feedback until the task is completed.
    reward_mode = "normalized_dense"

    # set random seed if there isn't a random see provided
    if args.seed is not None:
        set_random_seed(args.seed)

    # CREATE EVAL ENVIRONMENT
    # If args.eval is True (indicating an evaluation phase), the record_dir is updated to include an eval subdirectory
    if args.eval:
        record_dir = osp.join(record_dir, "eval")

    # The model to eval the model on
    model_ids = args.eval_model_id

    # eval_env_kwargs is a dictionary containing various environment configurations like whether the training is randomized (randomized_training),
    # whether external disturbances are included (ext_disturbance), and so on
    eval_env_kwargs = dict(
        randomized_training=args.randomized_training,
        # auto_dr=args.auto_dr,
        robot=args.robot,
        obs_noise=args.obs_noise,
        ext_disturbance=args.ext_disturbance,
        inc_obs_noise_in_priv=args.inc_obs_noise_in_priv,
        test_eval=args.eval,
        sim_freq=120,
        # seed=args.seed,
        # model_ids=model_ids if model_ids else [],
    )
    
    # begin
    if args.use_depth_adaptation or args.use_depth_base:
        # A vectorized environment is created using make_vec_env, wrapping it with ManiSkillRGBDVecEnvWrapper and then SB3VecEnvWrapper
        # Create vectorized environments for training
        # env :: mani_skill2.vector.vec_env.VecEnv
        # RGBDVecEnv(<ContinuousTaskWrapper<TimeLimit<PickSingleYCBRMA<PickSingleYCB-v1>>>>)
        eval_env = make_vec_env(
            env_id,
            num_envs=1,
            # record_dir=record_dir,
            obs_mode=obs_mode,
            control_mode=control_mode,
            reward_mode=reward_mode,
            wrappers=[partial(SuccessInfoWrapper)],
            **eval_env_kwargs)
        # ManiSkillRGBDVecEnvWrapper(<ContinuousTaskWrapper<TimeLimit<PickSingleYCBRMA<PickSingleYCB-v1>>>>)
        eval_env = ManiSkillRGBDVecEnvWrapper(eval_env)
        # <mani_skill2.vector.wrappers.sb3.SB3VecEnvWrapper object at 0x150c7a9c1750>
        eval_env = SB3VecEnvWrapper(eval_env)

    # If depth adaptation isn’t used, a vectorized environment is created using SubprocVecEnv with the same environment settings as for evaluation but with num_envs environments running in parallel
    # SubprocVecEnv is a type of vectorized environment provided by stable_baselines3 and its predecessor libraries. Its primary purpose is to allow parallel simulation of multiple instances of an environment using separate processes.
    # Parallel Environment Execution: SubprocVecEnv runs multiple environments in parallel across separate processes. This parallelization improves data throughput and allows for faster data collection, as each environment runs independently and simultaneously.
    # Efficient Sample Collection: By running environments in parallel, SubprocVecEnv can collect more experiences (e.g., state transitions, rewards) at a higher rate compared to running a single environment sequentially. This is especially beneficial for RL algorithms that require a lot of interaction with the environment (e.g., PPO, A2C).
    else:
        # <stable_baselines3.common.vec_env.subproc_vec_env.SubprocVecEnv object at 0x154358125490>
        eval_env = SubprocVecEnv([
            make_env(env_id,
                     record_dir=record_dir,
                     obs_mode=obs_mode,
                     control_mode=control_mode,
                     reward_mode=reward_mode,
                     **eval_env_kwargs) for _ in range(1)
        ])
    # end new
    # old
    # eval_env = SubprocVecEnv(
    #                 [make_env(env_id, record_dir=record_dir, obs_mode=obs_mode,
    #                 control_mode=control_mode, reward_mode=reward_mode,
    #                 **eval_env_kwargs
    #             ) for _ in range(1)])
    # end old
    eval_env = VecMonitor(
        eval_env)  # attach this so SB3 can log reward metrics
    eval_env.seed(args.seed)
    eval_env.reset()

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

    # CREATE TRAINING ENVIRONMENT
    else:
        if args.use_depth_adaptation or args.use_depth_base:
            # Create vectorized environments for training
            # env :: mani_skill2.vector.vec_env.VecEnv
            # RGBDVecEnv(<ContinuousTaskWrapper<TimeLimit<PickSingleYCBRMA<PickSingleYCB-v1>>>>)
            env: VecEnv = make_vec_env(env_id,
                                       num_envs=num_envs,
                                       obs_mode=obs_mode,
                                       control_mode=control_mode,
                                       wrappers=[
                                           partial(ContinuousTaskWrapper),
                                           partial(SuccessInfoWrapper)
                                       ],
                                       max_episode_steps=max_episode_steps,
                                       **env_kwargs)
            # ManiSkillRGBDVecEnvWrapper(<ContinuousTaskWrapper<TimeLimit<PickSingleYCBRMA<PickSingleYCB-v1>>>>)
            env = ManiSkillRGBDVecEnvWrapper(env)
            # <mani_skill2.vector.wrappers.sb3.SB3VecEnvWrapper object at 0x150c7a9c1750>
            env = SB3VecEnvWrapper(env)

        else:
            # <stable_baselines3.common.vec_env.subproc_vec_env.SubprocVecEnv object at 0x154358125490>
            env = SubprocVecEnv([
                make_env(env_id,
                         max_episode_steps=max_episode_steps,
                         obs_mode=obs_mode,
                         control_mode=control_mode,
                         reward_mode=reward_mode,
                         **env_kwargs) for _ in range(num_envs)
            ], )

        # In both cases (training and evaluation), the environment is wrapped with VecMonitor, which tracks the rewards and logs them for later analysis
        env = VecMonitor(env)
        env.seed(args.seed)
        env.reset()
    return env, eval_env
