# See custom task building: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/index.html
# See simulation 101: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/simulation_101.html
# See PushCube environment example: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/push_cube.py
# See environment template: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/template.py
# See reproducability and RNG: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/rng.html
# See custom robots: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html
# See domain randomization: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html

import numpy as np
import torch
from collections import OrderedDict
from transforms3d.quaternions import axangle2quat, qmult
from algo.misc import linear_schedule

# Maniskill-specific imports
from mani_skill.envs.tasks.tabletop import PickSingleYCBEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import get_pairwise_contact_impulse

@register_env("PickSingleYCB-v1", max_episode_steps=50, asset_download_ids=["ycb"], override=True)
class PickSingleYCBEnvRMA(PickSingleYCBEnv):
    """
    PickSingleYCBEnv with:
        - domain randomization (with linear scheduling)
        - observation state_dict that includes keys ['observation', 'privileged_info', 'goal_info'] which is used in RMA training
    """
    def set_randomization_ranges(self):
        # Environment variables (object scale multiplier, object density multiplier, object coefficient of friction)
        if self.randomized_env:
            self.scale_low = 0.70
            self.scale_high = 1.20
            self.density_low = 0.50
            self.density_high = 5.00
            self.friction_low = 0.50
            self.friction_high = 1.10
        # External disturbance: value by which we scale the external disturbance force applied on object
        if self.ext_disturbance:
            self.force_scale_low = 0
            self.force_scale_high = 2.0
            self.force_decay = 0.8
            self.disturb_force = np.zeros(3)
        # Observation noise: account for inaccuracies in robot joint position, object position, rotation estimation reading
        if self.obs_noise:
            self.joint_pos_low = -0.005
            self.joint_pos_high = 0.005
            self.obj_pos_low = -0.005
            self.obj_pos_high = 0.005
            self.obj_rot_low = -np.pi * (10 / 180)
            self.obj_rot_high = np.pi * (10 / 180)

        # Linear scheduling for lows and highs of randomization ranges
        init_step, end_step = 1e6, 2e6
        if self.randomized_env:
            self.scale_low_scdl = linear_schedule(1.0, self.scale_low, init_step, end_step)
            self.scale_high_scdl = linear_schedule(1.0, self.scale_high, init_step, end_step)
            self.density_low_scdl = linear_schedule(1.0, self.density_low, init_step, end_step)
            self.density_high_scdl = linear_schedule(1.0, self.density_high, init_step, end_step)
            self.friction_low_scdl = linear_schedule(1.0, self.friction_low, init_step, end_step)
            self.friction_high_scdl = linear_schedule(1.0, self.friction_high, init_step, end_step)
        if self.ext_disturbance:
            self.force_scale_scdl = linear_schedule(self.force_scale_low, self.force_scale_high, init_step, end_step)
        if self.obs_noise:
            self.joint_pos_low_scdl = linear_schedule(0, self.joint_pos_low, init_step, end_step)
            self.joint_pos_high_scdl = linear_schedule(0, self.joint_pos_high, init_step, end_step)
            self.obj_pos_low_scdl = linear_schedule(0, self.obj_pos_low, init_step, end_step)
            self.obj_pos_high_scdl = linear_schedule(0, self.obj_pos_high, init_step, end_step)
            self.obj_rot_low_scdl = linear_schedule(0, self.obj_rot_low, init_step, end_step)
            self.obj_rot_high_scdl = linear_schedule(0, self.obj_rot_high, init_step, end_step)
    
    def __init__(
            self,
            *args,
            randomized_env=False,
            obs_noise=False,
            ext_disturbance=False,
            **kwargs
    ):
        self.randomized_env = randomized_env # TODO: what / do parms need to be batched? how to handle?
        self.obs_noise = obs_noise
        self.ext_disturbance = ext_disturbance
        self.set_randomization_ranges()
        self.step_counter = 0 # TODO: does this need to be batched?

    """
    Environment reset and reconfiguration Code:
    Below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general.
    Environment frozen after reconfiguration. Assets cannot be added or removed until reconfigure() is called again.
    In CPU simulation, for some tasks these may need to be called multiple times if you need to swap out object assets.
    In GPU simulation these will only ever be called once.
    """
    def reset(self, seed=None, options=None): # TODO: change to inheriting from _load_scene()
        # for 'panda' and 'panda_wristcam', 7 arm joints + 2 gripper joints
        self.joint_pos_dim = 9

        self.set_episode_rng(seed)

        if self.obs_noise:
            # noise to proprioception, i.e. robot joint positions (dim = 9)
            self.joint_pos_l = self.joint_pos_low_scdl(elapsed_steps=self.step_counter)
            self.joint_pos_h = self.joint_pos_high_scdl(elapsed_steps=self.step_counter)
            self.joint_pos_noise = self._episode_rng.uniform(self.joint_pos_l, self.joint_pos_h, self.joint_pos_dim)
            # noise to object position (dim = 3)
            self.obj_pos_l = self.obj_pos_low_scdl(elapsed_steps=self.step_counter)
            self.obj_pos_h = self.obj_pos_high_scdl(elapsed_steps=self.step_counter)
            self.obj_pos_noise = self._episode_rng.uniform(self.obj_pos_h, self.obj_pos_l, 3)
            # noise to object rotation (dim = 4)
            self.obj_rot_l = self.obj_rot_low_scdl(elapsed_steps=self.step_counter)
            self.obj_rot_h = self.obj_rot_high_scdl(elapsed_steps=self.step_counter)
            rot_axis = self._episode_rng.uniform(0, 1, 3)
            self.rot_ang = self._episode_rng.uniform(self.obj_rot_h, self.obj_rot_l)
            self.rot_noise = axangle2quat(rot_axis, self.rot_ang)
        
        # options dictionary stores 'reconfigure' key-value pair and other relevent things for task scene loading
        if options is None:
            options = dict()
        model_scale = options.pop("model_scale", 1.0)
        model_id = options.pop("model_id", None)
        reconfigure = options.pop("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure

        if self.randomized_env:
            self.scale_l = self.scale_low_scdl(elapsed_steps=self.step_counter)
            self.scale_h = self.scale_high_scdl(elapsed_steps=self.step_counter)
            self.scale_multiplier = self._episode_rng.uniform(self.scale_l, self.scale_h)
        options["model_scale"] = model_scale * self.scale_multiplier

        return super().reset(seed=self._episode_seed, options=options)

    """
    Episode Initialization Code
    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task. Note that these functions are given a env_idx variable.

    `env_idx` is a torch Tensor representing the indices of the parallel environments that are being initialized/reset. This is used
    to support partial resets where some parallel envs might be reset while others are still running (useful for faster RL and evaluation).
    Generally you only need to really use it to determine batch sizes via len(env_idx). ManiSkill helps handle internally a lot of masking
    you might normally need to do when working with GPU simulation. For specific details check out the push_cube.py code
    """

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    """
    Modifying observations, goal parameterization, and success conditions for your task:
    The code below all impact some part of `self.step` function
    """

    def _get_obs_extra(self, info):
        """
        Return a dict of additional observation data specific to the task:
        - This will be included as part of the observation in the "extra" key when obs_mode="state_dict" or any of the visual obs_modes 
        and included as part of a flattened observation when obs_mode="state".
        - Moreover, you have access to the info object which is generated by the `evaluate` function above

        Notes:
        - _get_obs_extra() is called within get_obs(), which in turn is called within step()
        - ext disturb force
        """
        # TODO: how to handle batches?
        # check whether object is grasped by gripper
        grasped = self.agent.is_grasping(self.obj)
        # add external disturbance force
        if self.ext_disturbance:
            # decay the previous disturbance force
            self.disturb_force *= self.force_decay
            # sample whether to apply new force with probablity 0.1
            if self._episode_rng.uniform() < 0.1: # TODO: switch to _batched_episode_rng???
                # sample 3D force for guassian distribution
                self.disturb_force = self._episode_rng.normal(0, 0.1, 3)
                self.disturb_force /= np.linalg.norm(self.disturb_force, ord=2)
                # sample value by which we scale the force (depends on linear schedule)
                self.fs_h = self.force_scale_scdl(elapsed_steps=self.step_counter)
                self.force_scale = self._episode_rng.uniform(0, self.fs_h)
                # scale by object mass and force scale value we just computed
                self.disturb_force *= self.obj.mass * self.force_scale
            if grasped:
                # only apply the force to object if it is grasped
                self.obj.apply_force(self.disturb_force, self.obj.pose.p) # TODO: update ManiSkill to support apply_force
        
        # Privileged environment information: magnitudes of the impulses applied by the left and right finger of the gripper
        contacts = self.obj.px.get_contacts()
        limpulse = np.linalg.norm(get_pairwise_contact_impulse(contacts, self.agent.finger1_link, self.obj), ord=2)
        rimpulse = np.linalg.norm(get_pairwise_contact_impulse(contacts, self.agent.finger2_link, self.obj), ord=2)

        if self.obs_noise:
            # noise to proprioception
            qpos = self.agent.robot.get_qpos() + self.proprio_noise
            proprio = np.concatenate([qpos, self.agent.robot.get_qvel()])
            # noise to obj position
            obj_pos = self.obj.pose.p
            obj_pos += self.pos_noise
            # noise to obj rotation
            obj_ang = self.obj.pose.q
            obj_ang = qmult(obj_ang, self.rot_noise)

        priv_info_dict = OrderedDict(
            obj_ang=obj_ang,  # 4
            bbox_size=self.model_bbox_size,  # 3
            obj_density=self.obj_density,  # 1
            obj_friction=self.obj_friction,  # 1
            limpulse=limpulse,  # 1
            rimpulse=rimpulse)  # 1

        return dict()

    """
    Miscellaneous code:
    """
    def step(self, action):
        self.step_counter += 1
        return super().step(action)