# Import required packages
import json
import os.path as osp

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import parse_args, config_log_path, config_envs
from algo.models import FeaturesExtractorRMA
from algo.ppo_rma import PPORMA
from algo.adaptation import ProprioAdapt
from algo.policy_rma import ActorCriticPolicyRMA
from task import gym_task_map
from algo.evaluate_policy import evaluate_policy
from algo.callbacks_rma import EvalCallbackRMA, CheckpointCallbackRMA
from algo.misc import linear_schedule


def main():
    # use parse_args() function from config.py to parse arguments from command line and store them into 'args' object
    args = parse_args()
    print("args:", args)

    num_envs = args.n_envs
    log_dir = args.log_dir
    rollout_steps = args.rollout_steps

    # configure paths where logs will be kept as described by config_log_path() function in config.py
    record_dir, ckpt_dir, ckpt_path, tb_path_root = config_log_path(args)

    # dictionary which contains the model configurations (all the parameters)
    # stored in a JSON file in the appropriate directory within log directory
    args_dict = vars(args)
    with open(ckpt_dir + '/args.json', 'w') as f:
        json.dump(args_dict, f, indent=4)

    # ---- Config the environments
    env, eval_env = config_envs(args, record_dir)

    # ---- policy configuration and algorithm configuration
    env_name, env_version = args.env_id.split("-")

    # kwargs is short for keyword arguments and is used when passing a variable number of keyword (named) arguments to a function
    # When defined as **kwargs in a function signature, it allows the function to receive a dictionary of keyword arguments.
    # kwargs is typically seen as part of function calls or parameter setups where additional keyword arguments are passed without explicitly naming them all.
    # For instance, features_extractor_kwargs=dict(...) is a dictionary that can be passed as **features_extractor_kwargs to another function or class.
    # This means that each key-value pair in the dictionary is passed as a keyword argument to the called function or class
    # def example_function(*args, **kwargs):
    # for arg in args:
    #     print(arg)
    # for key, value in kwargs.items():
    #     print(f"{key}: {value}")

    # example_function(1, 2, 3, a=4, b=5)

    # The code sets up a dictionary, policy_kwargs, which specifies the configuration for the policy network used by the PPO algorithm
    policy_kwargs = dict(
        net_arch={
            "pi": args.policy_arch,
            "vf": args.policy_arch
        },
        n_envs=num_envs,
        sys_iden=args.sys_iden,
        inc_obs_noise_in_priv=args.inc_obs_noise_in_priv,
        object_emb_dim=args.obj_emb_dim,
        use_depth_adaptation=args.use_depth_adaptation,
        env_name=env_name,

        # features_extractor_kwargs: A nested dictionary for configuring the FeaturesExtractorRMA class, with parameters such as object_emb_dim, use_depth_base, and sys_iden.
        # This allows detailed customization of the feature extraction module in the policy.
        features_extractor_kwargs=dict(
            object_emb_dim=args.obj_emb_dim,
            env_name=env_name,
            # use_depth_adaptation=args.use_depth_adaptation,
            use_depth_base=args.use_depth_base,
            use_prop_history_base=args.use_prop_history_base,
            only_dr=args.only_DR or args.auto_dr,
            sys_iden=args.sys_iden,
            without_adapt_module=args.without_adapt_module,
            inc_obs_noise_in_priv=args.inc_obs_noise_in_priv,
        ))

    # If the environment version (env_version) is either 'v1' or 'v2', the features_extractor_class key is added to policy_kwargs,
    # specifying that the policy should use the FeaturesExtractorRMA class for feature extraction. 
    if env_version in ['v1', 'v2']:
        policy_kwargs.update(features_extractor_class=FeaturesExtractorRMA)

    end_step = args.anneal_end_step

    # If args.lr_schedule is set, a learning rate schedule is created using the linear_schedule function.
    # This function linearly decreases the learning rate from 3e-4 to 1e-5 over the course of args.total_timesteps, stopping at end_step
    if args.lr_schedule:
        lr = linear_schedule(3e-4,
                             1e-5,
                             total_steps=args.total_timesteps,
                             end_step=end_step)
    # If args.lr_schedule is not set, the learning rate is fixed at 3e-4
    else:
        lr = 3e-4

    # Similar to the learning rate schedule, the clip_range (used in PPO to limit policy updates for stability) is set up based on args.clip_range_schedule
    # If a schedule is specified, linear_schedule decreases the clip_range from args.clip_range to 0.05 over args.total_timesteps, stopping at end_step
    if args.clip_range_schedule:
        clip_range = linear_schedule(args.clip_range,
                                     0.05,
                                     total_steps=args.total_timesteps,
                                     end_step=end_step)
    # If args.clip_range_schedule is not set, the clip_range is fixed at args.clip_range
    else:
        clip_range = args.clip_range

    # A PPORMA (custom PPO implementation for Rapid Motor Adaptation) model is instantiated with the ActorCriticPolicyRMA policy class
    # and configured with hyperparameters and training settings
    model = PPORMA(
        ActorCriticPolicyRMA,
        env,
        auto_dr=args.auto_dr,
        use_prop_history_base=args.use_prop_history_base,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
        clip_range=clip_range,
        verbose=2,
        n_steps=rollout_steps,
        batch_size=args.batch_size,
        gamma=0.85,
        n_epochs=args.n_epochs,
        tensorboard_log=tb_path_root,
        target_kl=args.target_kl,
        eval=args.eval,
    )
    # creating a boolean variable exclude_adaptor_net based on several conditions related to the training configuration.
    # The exclude_adaptor_net variable is used to determine whether the adaptation network/module should be excluded
    # during certain phases of training or evaluation
    exclude_adaptor_net = (
        (args.adaptation_training and not args.continue_training)
        or  # just starting adaptor training
        (not args.adaptation_training) and (not args.eval) or  # base training
        args.only_DR or args.auto_dr or args.use_depth_base
        or args.expert_adapt)  # DR methods
    
    # if we only intend to evaulate policy
    if args.eval:
        # if args.ckpt_name is None:
        #     ckpt_path = osp.join(log_dir, "best_model")
        # The model.load() method loads the pre-trained model, setting the environment, tensorboard log path,
        # and policy keyword arguments. The exclude_adaptor_net flag ensures specific components are not loaded if needed
        model = model.load(
            ckpt_path,
            env=env,
            tensorboard_log=tb_path_root,
            policy_kwargs=policy_kwargs,
            exclude_adaptor_net=exclude_adaptor_net,
        )
        model.policy.use_depth_base = args.use_depth_base
        model.policy.use_prop_history_base = args.use_prop_history_base
        model.policy.use_depth_adaptation = args.use_depth_adaptation
        model.policy.adapt_tconv.use_depth = args.use_depth_adaptation
    
    # If the script is not in evaluation mode, it checks whether training should continue or start from a checkpoint.
    # This can be for regular training, adaptation training, or transfer learning
    else:
        # If args.continue_training, args.adaptation_training, or args.transfer_learning are True, it attempts to
        # load the existing model checkpoint from ckpt_path. If the file does not exist, a warning is printed for debugging purposes
        if (args.continue_training or args.adaptation_training
                or args.transfer_learning):
            # if the file specified by ckpt_path doesn't exist
            if not osp.exists(ckpt_path):
                print(
                    f"###  Warning: ckpt_path {ckpt_path} doesn't exist -- for debugging only"
                )
            else:
                model = model.load(
                    ckpt_path,
                    env=env,
                    tensorboard_log=tb_path_root,
                    policy_kwargs=policy_kwargs,
                    exclude_adaptor_net=exclude_adaptor_net,
                )

                assert model.policy.adapt_tconv.use_depth == args.use_depth_adaptation

                # If the model is loaded successfully, its observation space and the policy’s observation space
                # are updated to match the current environment
                model.observation_space = env.observation_space
                model.policy.observation_space = env.observation_space
                reset_num_timesteps = False
        
        # If no continuation or adaptation is needed, reset_num_timesteps is set to True, indicating a fresh start for training
        else:
            reset_num_timesteps = True

        env.env_method('set_step_counter', model.num_timesteps // num_envs)
        eval_env.env_method('set_step_counter',
                            model.num_timesteps // num_envs)

        # CALLBACKS:
        # In machine learning and deep reinforcement learning, callbacks are functions or classes that are executed at certain stages
        # during training or evaluation. They allow you to monitor, control, and customize the training process without directly altering the core training loop.
        # Callbacks can be used to:
        # Monitor Performance: Track metrics like rewards, losses, or accuracy during training and save them for analysis
        # Save Checkpoints: Periodically save the state of the model, ensuring progress isn’t lost and enabling resumption from a certain point if training is interrupted.
        # Evaluate the Model: Run evaluations at regular intervals to assess the model’s performance on validation or test environments
        # Early Stopping: Stop training automatically if certain conditions are met, such as no improvement in performance over a specified period
        # Adjust Hyperparameters: Dynamically modify parameters like learning rates or exploration strategies based on the current training state.

        # EXAMPLES:
        # EvalCallbackRMA: Evaluates the model periodically during training, saving the best version and recording logs to the specified directory.
        # CheckpointCallbackRMA: Saves the current model state and relevant data at regular intervals, providing checkpoint files that can be used for resuming training or analysis

        # functions used here are from callbacks_rma.py file
        # define callbacks to periodically save our model and evaluate it to
        # help monitor training the below freq values will save every 10
        # rollouts.
        # the callback is called every rollout_steps * n_envs steps
        eval_freq = args.rollout_steps
        save_freq = args.rollout_steps * 10
        eval_callback = EvalCallbackRMA(
            eval_env,
            best_model_save_path=ckpt_dir,
            log_path=ckpt_dir,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            num_envs=num_envs,
        )
        checkpoint_callback = CheckpointCallbackRMA(
            save_path=ckpt_dir,
            save_freq=save_freq,
            name_prefix="model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        # If args.adaptation_training is True, the ProprioAdapt class is used to handle adaptation-specific training.
        # A SummaryWriter is created for tensorboard logging.
        # Train an agent with PPO for args.total_timesteps interactions
        if args.adaptation_training:
            # adaptation training assumes the policy has been trained
            # create a tensorboard write
            # SummaryWriter() creates a file writer that logs metrics, images, graphs, and other visual data to a specified directory.
            # This data can then be visualized in real-time using TensorBoard, which helps in understanding the training process and debugging the model.
            writer = SummaryWriter(log_dir=tb_path_root)
            # instantiate adapter class
            algo = ProprioAdapt(
                model=model,
                env=env,
                writer=writer,
                save_dir=ckpt_dir,
            )
            # If args.compute_adaptation_loss is set, it computes and prints the mean adaptation loss; 
            # otherwise, the algo.learn() method runs the adaptation training loop.
            if args.compute_adaptation_loss:
                print("Computing adaptation loss")
                adaptation_loss = algo.compute_mean_adaptor_loss()
            else:
                # Train adapter
                algo.learn()
        else:
            # Base policy training:
            # If adaptation training is not specified, the model trains using the model.learn() method
            # with the defined callbacks and specified total timesteps
            model.learn(
                args.total_timesteps,
                callback=[checkpoint_callback, eval_callback], # callbacks are passed to the learn() function
                reset_num_timesteps=False,
                tb_log_name='',
            )
        # Save the final model
        model.save(osp.join(log_dir, "latest_model"))

    # Both env and eval_env have their step counters reset to a large value (1e8),
    # which is a way to signal the end of training or switch the mode of operation in the environments
    env.env_method('set_step_counter', 1e8)
    eval_env.env_method('set_step_counter', 1e8)

    # This block evaluates the performance of the trained model on the environment using evaluate_policy()
    if not args.compute_adaptation_loss:
        # n_eval_eps = 1
        n_eval_eps = 100
        # observations = env.reset()
        # from stable_baselines3.common.utils import obs_as_tensor
        # model.policy(obs_as_tensor(observations,"cuda"), adapt_trn=True)
        returns, ep_lens = evaluate_policy(
            model,
            # observations,
            eval_env,
            deterministic=True,
            render=False,
            return_episode_rewards=True,
            n_eval_episodes=n_eval_eps,
            test_mode=args.eval,
            expert_adapt=args.expert_adapt,
            only_dr=args.only_DR or args.auto_dr,
            without_adapt_module=args.without_adapt_module,
        )
        print("Returns", returns)
        print("Episode Lengths", ep_lens)
        mean_ep_lens = np.mean(np.array(ep_lens))
        success = np.array(ep_lens) < 200
        success_rate = success.mean()
        print("Mean Episode Lengths", mean_ep_lens)
        print("Success Rate:", success_rate)

    # save the results in the logs/eval_results.csv
    # create the file if it doesn't exist
    # eval_results_path = osp.join(log_dir, "eval_results.csv")
    eval_results_path = osp.join(log_dir, "eval_results_finegrained.csv")
    if not osp.exists(eval_results_path):
        with open(eval_results_path, "w") as f:
            # create the header if it's the first time creating the file
            f.write("log_name,expert_adapt,inc_obs_noise_in_priv,only_DR,"+\
                "without_adapt_module,sys_iden,success_rate,"+\
                "mean_ep_len,n_eval_eps,env_name,model_id,adapt_loss\n")
    
    # The evaluation results (e.g., success_rate, mean_ep_len, n_eval_eps) are appended to the CSV file
    # save the log_name, args.expert_adapt, args.inc_obs_noise_in_priv
    # args.only_DR, args.without_adapt_module, args.sys_iden
    log_model_id = args.eval_model_id if args.eval_model_id else "All"
    with open(eval_results_path, "a") as f:
        if args.compute_adaptation_loss:
            f.write(f"{args.log_name},{args.expert_adapt},"+\
                f"{args.inc_obs_noise_in_priv},{args.only_DR},"+\
                f"{args.without_adapt_module},{args.sys_iden},"+\
                f",,"+\
                f",{env_name},{log_model_id},{adaptation_loss}\n")
        else:
            f.write(f"{args.log_name},{args.expert_adapt},"+\
                f"{args.inc_obs_noise_in_priv},{args.only_DR},"+\
                f"{args.without_adapt_module},{args.sys_iden},"+\
                f"{success_rate:.3f},{mean_ep_lens},"+\
                f"{n_eval_eps},{env_name},{log_model_id}\n")


if __name__ == "__main__":
    main()
