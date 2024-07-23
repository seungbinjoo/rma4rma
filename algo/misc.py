import os
import random
from typing import Dict
import shlex
import subprocess

import torch as th
# import gym
# import gym.spaces as spaces
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from numpy.linalg import norm
import torch
# from omegaconf import DictConfig
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.utils.wrappers.sb3 import ContinuousTaskWrapper, \
                                           SuccessInfoWrapper
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from typing import Callable


class ManiSkillRGBDVecEnvWrapper(VecEnvObservationWrapper):

    def __init__(self, env):
        assert env.obs_mode == "rgbd"
        # we simply define the single env observation space. The inherited wrapper automatically computes the batched version
        single_observation_space = ManiSkillRGBDWrapper.init_observation_space(
            env.single_observation_space)
        super().__init__(env, single_observation_space)

    def observation(self, observation):
        return ManiSkillRGBDWrapper.convert_observation(observation)


class ManiSkillRGBDWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        assert env.obs_mode == "rgbd"
        self.observation_space = self.init_observation_space(
            env.observation_space)

    @staticmethod
    def init_observation_space(obs_space: spaces.Dict):
        # Concatenate all the image spaces
        image_shapes = []
        for cam_uid in obs_space["image"]:
            cam_space = obs_space["image"][cam_uid]
            # image_shapes.append(cam_space["rgb"].shape)
            image_shapes.append(cam_space["depth"].shape)
        image_shapes = np.array(image_shapes)
        assert np.all(image_shapes[0, :2] == image_shapes[:, :2]), image_shapes
        h, w = image_shapes[0, :2]
        c = image_shapes[:, 2].sum(0)
        rgbd_space = spaces.Box(0, np.inf, shape=(c, h, w))

        # Create the new observation space
        obs_space["image"] = rgbd_space
        return obs_space

    @staticmethod
    def convert_observation(observation):
        # Process images. RGB is normalized to [0, 1].
        images = []
        for cam_uid, cam_obs in observation["image"].items():
            # rgb = cam_obs["rgb"] / 255.0
            depth = cam_obs["depth"]
            # depth = np.transpose(depth, (2, 0, 1))
            if len(depth.shape) == 3:
                depth = np.transpose(depth, (2, 0, 1))
            elif len(depth.shape) == 4:
                depth = depth.permute(0, 3, 1, 2)
            else:
                raise NotImplementedError

            # NOTE: SB3 does not support GPU tensors, so we transfer them to CPU.
            # For other RL frameworks that natively support GPU tensors, this step is not necessary.
            # if isinstance(rgb, th.Tensor):
            #     rgb = rgb.to(device="cpu", non_blocking=True)
            if isinstance(depth, th.Tensor):
                depth = depth.to(device="cpu", non_blocking=True)

            # images.append(rgb)
            images.append(depth)

        # Concatenate all the images
        rgbd = np.concatenate(images, axis=-1)

        # Concatenate all the states
        # state = np.hstack(
        #     [
        #         flatten_state_dict(observation["agent"]),
        #         flatten_state_dict(observation["extra"]),
        #     ]
        # )
        observation['image'] = rgbd

        # return dict(rgbd=rgbd, state=state)
        return observation

    def observation(self, observation):
        return self.convert_observation(observation)


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def linear_schedule(initial_value: float,
                    final_value: float,
                    init_step: int = 0,
                    end_step: int = 2e7,
                    total_steps: int = None) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    anneal_percent goes from 0 (beginning) to 1 (end).
    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    when initial_value = 1, final_value = 2, the diff is -1
        when percent = 10%, the current_value = 1 - 0.1 * (-1) = 
        when percent = 99%, the current_value = 1 - 0.99 * (-1) = 1.99
    """

    def func(progress_remaining: float = None, elapsed_steps=None) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if elapsed_steps is None:
            elapsed_steps = total_steps * (1 - progress_remaining)
        if elapsed_steps < init_step:
            return initial_value
        anneal_percent = min(elapsed_steps / end_step, 1.0)
        return initial_value - anneal_percent * (initial_value - final_value)

    return func


class LatestCheckpointCallback(CheckpointCallback):

    def _checkpoint_path(self,
                         checkpoint_type: str = "",
                         extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"latest.{extension}")


def get_task_id(task_name: str) -> np.array:
    '''
    Get the task_id from task_name.
    '''
    name_to_id = {
        'PickCube': 0,
        'PickSingleYCB': 0,
        'StackCube': 1,
    }
    return np.array([name_to_id[task_name]])


def get_object_id(task_name: str,
                  model_id: str = None,
                  object_list: list = None) -> np.array:
    '''
    When task_id = 'PickCube', the object is always a cube, so model_id and 
        root_dir is not needed.
    When task_id = 'PickSingleYCB', the model_id and object_list is needed. And 
        the id is the model's position inside the root_dir.
    '''
    if task_name in ['PickCube', 'StackCube']:
        return np.array([1])
    elif task_name in ['PegInsertion']:
        return np.array([0])
    elif task_name in ['TurnFaucet'] and object_list == None:
        return np.array([0])
    elif task_name in ['TurnFaucet', 'PickSingleYCB', 'PickSingleEGAD']:
        assert model_id is not None and object_list is not None
        return np.array([object_list.index(model_id) + 2])
    else:
        raise NotImplementedError


class RecordEpisodeRandInfo(RecordEpisode):

    def flush_video(self,
                    suffix="",
                    verbose=False,
                    ignore_empty_transition=False):
        # save all the randomized environment parameters to a csv file
        # create a csv with name f"{self._episode_id}.csv"
        # with header parameter_name, low_range, top_range, current value
        if not self.save_video or len(self._render_images) == 0:
            return
        if ignore_empty_transition and len(self._render_images) == 1:
            return
        # csv_path = os.path.join(self.output_dir, f"{self._episode_id}.csv")
        # with open(csv_path, "w") as f:
        #     # f.write(f"{self.env.seedd}\n")
        #     f.write(f"{self.env.step_counter}\n")
        #     f.write("parameter_name,low_range,top_range,current_value\n")
        #     f.write(f"scale,{self.env.scl_scdl_l},{self.env.scl_scdl_h},{self.env.model_scale_mult}\n")
        #     f.write(f"density,{self.env.dens_scdl_l},{self.env.dens_scdl_h},{self.env.dens_mult}\n")
        #     f.write(f"friction,{self.env.fric_scdl_l},{self.env.fric_scdl_h},{self.env.obj_friction}\n")

        #     prop_norm_l = norm(np.full((9,), self.env.prop_scdl_l), ord=2)
        #     prop_norm_h = norm(np.full((9,), self.env.prop_scdl_h), ord=2)
        #     prop_norm_sam = norm(self.env.proprio_noise, ord=2)
        #     f.write(f"prop_nos,0,{prop_norm_h},{prop_norm_sam}\n")
        #     pos_norm_l = norm(np.full((3,), self.env.pos_scdl_l), ord=2)
        #     pos_norm_h = norm(np.full((3,), self.env.pos_scdl_h), ord=2)
        #     pos_norm_sam = norm(self.env.pos_noise, ord=2)
        #     f.write(f"pos_nos,0,{pos_norm_h},{pos_norm_sam}\n")
        #     f.write(f"rot_nos,{self.env.rot_scdl_l},{self.env.rot_scdl_h},{self.env.rot_ang}\n")

        #     f.write(f"dist_f,0,{self.force_scale_h},{self.env.max_force}\n")
        super().flush_video(suffix, verbose, ignore_empty_transition)


# define an SB3 style make_env function for evaluation
def make_env(env_id: str,
             obs_mode: str = "state",
             control_mode: str = "pd_ee_delta_pose",
             reward_mode: str = "normalized_dense",
             max_episode_steps: int = None,
             record_dir: str = None,
             **kwargs):

    def _init() -> gym.Env:
        # NOTE: Import envs here so that they are registered with gym in subprocesses
        import mani_skill2.envs
        # if env_id == "PickSingleYCB-v1":
        # model_id = ['065-a_cups', '065-b_cups', '065-c_cups', '065-d_cups',
        #             '065-e_cups', '065-f_cups', '065-g_cups', '065-h_cups',
        #             '065-i_cups', '065-j_cups']
        env = gym.make(env_id,
                       obs_mode=obs_mode,
                       reward_mode=reward_mode,
                       control_mode=control_mode,
                       renderer_kwargs={"offscreen_only": True},
                       render_mode="cameras",
                       max_episode_steps=max_episode_steps,
                       **kwargs)
        # else:
        #     env = gym.make(env_id, obs_mode=obs_mode,
        #                reward_mode=reward_mode,
        #                control_mode=control_mode,
        #                renderer_kwargs={"offscreen_only": True}, **kwargs)
        # For training, we regard the task as a continuous task with infinite
        # horizon. you can use the ContinuousTaskWrapper here for that
        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env)
        if obs_mode == "rgbd":
            env = ManiSkillRGBDWrapper(env)

        if record_dir is not None:
            env = SuccessInfoWrapper(env)
            env = RecordEpisodeRandInfo(
                # todo: turn the info off for plotting
                env,
                record_dir,
                info_on_video=False,
                save_trajectory=False,
            )
        return env

    return _init


def git_diff_config(name):
    cmd = f'git diff --unified=0 {name}'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return seed


# def omegaconf_to_dict(d: DictConfig)->Dict:
#     """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
#     ret = {}
#     for k, v in d.items():
#         if isinstance(v, DictConfig):
#             ret[k] = omegaconf_to_dict(v)
#         else:
#             ret[k] = v
#     return ret


class AverageScalarMeter(object):

    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        '''Compute running average within a window size.
        Before the number of values reaches the window size, the average is 
        computed with all the values we have seen so far.
        After reaching the window size, the average is computed with all the
        most recent values within the window size. 
        '''
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        # ite1
        # window_size = 100, size = 10, current_size = 0
        # old_size = 0
        # ite2
        # old_size = min(100 - 10, 10) = 10
        # ite11
        # old_size = min(100 - 10, 100) = 90
        old_size = min(self.window_size - size, self.current_size)
        # size_sum = 0 + 10 = 10
        # size_sum = 10 + 10 = 20
        # size_sum = 90 + 10 = 100
        size_sum = old_size + size
        # current_size = 10
        # current_size = 20
        # current_size = 100
        self.current_size = size_sum
        # mean = (0 * 0 + new_mean * 10) / 10 = new_mean
        # mean = (mean * 10 + new_mean * 10) / 20 = (mean + new_mean) / 2
        # mean = (mean * 90 + new_mean * 10) / 100
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean
