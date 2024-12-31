import json
from config import parse_args, config_log_path, config_envs

"""
Command line input:
python config_test.py -n 50 -bs 5000 -rs 2000 --randomized_training --ext_disturbance --obs_noise -e PickCube-v1
"""

def config_test():
    print('check')
    args = parse_args()
    print("args:", args)

    # store important parameters that have been parsed
    num_envs = args.n_envs
    log_dir = args.log_dir
    rollout_steps = args.rollout_steps

    # configure paths where logs will be kept as described by config_log_path() function in config.py
    record_dir, ckpt_dir, ckpt_path, tb_path_root = config_log_path(args)

    # dictionary which contains the model configurations (all the parameters) --> stored in a json file in the appropriate directory within log directory
    # ---- Save the dictionary to a JSON file
    args_dict = vars(args)
    with open(ckpt_dir + '/args.json', 'w') as f:
        json.dump(args_dict, f, indent=4)

    # ---- Config the environments
    env, eval_env = config_envs(args, record_dir)

    # Test action and observation spaces
    print(env.action_space)
    print(env.observation_space)
    print(eval_env.action_space)
    print(eval_env.observation_space)

if __name__ == "__main__":
    config_test()