# See custom task building: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/index.html
# See simulation 101: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/simulation_101.html
# See PushCube environment example: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/push_cube.py
# See environment template: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/template.py
# See reproducability and RNG: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/rng.html
# See custom robots: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html
# See domain randomization: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html

import os
from typing import Dict, List
from pathlib import Path
import numpy as np
import torch
import sapien
from algo.misc import linear_schedule, get_ycb_builder_rma, get_object_id

# Maniskill-specific imports
from mani_skill import ASSET_DIR
from mani_skill.envs.tasks.tabletop import PickSingleYCBEnv
from mani_skill.utils.registration import register_env
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder

YCB_DATASET = dict()
WARNED_ONCE = False

@register_env("PickSingleYCBRMA-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PickSingleYCBEnvRMA(PickSingleYCBEnv):
    """
    PickSingleYCBEnv where:
        - Observation state_dict additionally includes privileged info and object id, which are used in RMA training
        - Domain randomization (with linear scheduling) is applied
            - Environment variables: object scale, object density, objeect friction
            - Disturbance: external force applied to object every step (with force decay)
            - Observation noise: joint position (proprioception) noise, object position noise, object rotation noise
    """
    def __init__(
            self,
            *args,
            randomized_env=True,
            obs_noise=True,
            ext_disturbance=True,
            **kwargs
    ):
        self.randomized_env = randomized_env
        self.obs_noise = obs_noise
        self.ext_disturbance = ext_disturbance
        self.step_counter = 0 # TODO: be able to pass current step, in case using checkpoint
        self.set_randomization_ranges()
        # get object instance and category lists, which will be used for computing object instance and category ids
        parent_folder = Path(ASSET_DIR) / "assets" / "mani_skill2_ycb" / "models"
        # E.g. ['038_padlock', '014_lemon, '065-e_cups', ...]
        self.obj_instance_list = [
            f for f in os.listdir(parent_folder)
            if os.path.isdir(os.path.join(parent_folder, f))
        ]
        # E.g. ['banana', 'bottle', 'mug', ...]
        self.obj_category_list = [
            f.rsplit("_", 1)[1] for f in os.listdir(parent_folder)
            if os.path.isdir(os.path.join(parent_folder, f))
        ]
        self.obj_category_list = list(set(self.obj_category_list))
        super().__init__(*args, **kwargs)
    
    # TODO: log these values in TensorBoard?
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
        # Observation noise: account for inaccuracies in robot joint position, object position, rotation estimation reading
        if self.obs_noise:
            self.joint_pos_low = -0.005
            self.joint_pos_high = 0.005
            self.obj_pos_low = -0.005
            self.obj_pos_high = 0.005
            self.obj_rot_low = -np.pi * (10 / 180)
            self.obj_rot_high = np.pi * (10 / 180)

        # Linear scheduling for lows and highs of randomization ranges
        init_step, end_step = 0, 1e6
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

    """
    Reconfiguration Code:
    Below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general.
    Environment frozen after reconfiguration. Assets cannot be added or removed until reconfigure() is called again.
    In CPU simulation, for some tasks these may need to be called multiple times if you need to swap out object assets.
    In GPU simulation these will only ever be called once.
    """

    # NOTE: variables are batched
    def _load_scene(self, options: dict):
        """
        Same as parent class but:
            - We load YCB object and randomize its scale, density, friction
            - We get object id and object type id (which will be used as additional observations in _get_obs_extra())
        """
        if self.randomized_env:
            # object scale
            self.scale_l = self.scale_low_scdl(elapsed_steps=self.step_counter)
            self.scale_h = self.scale_high_scdl(elapsed_steps=self.step_counter)
            self.scale_mults = self._batched_episode_rng.uniform(self.scale_l, self.scale_h) # [N]
            # object density
            self.density_h = self.density_low_scdl(elapsed_steps=self.step_counter)
            self.density_l = self.density_high_scdl(elapsed_steps=self.step_counter)
            self.density_mults = self._batched_episode_rng.uniform(self.density_l, self.density_h) # [N]
            # object coefficient of friction
            self.friction_high = self.friction_high_scdl(elapsed_steps=self.step_counter)
            self.friction_low = self.friction_low_scdl(elapsed_steps=self.step_counter)
            self.obj_friction = torch.from_numpy(self._batched_episode_rng.uniform(self.friction_low, self.friction_high)).to(self.device) # [N]

        global WARNED_ONCE
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # if not existent, initialize tensor to hold object instance ids and object category ids
        if not hasattr(self, 'obj_instance_id'):
            self.obj_instance_id = torch.zeros(self.num_envs, device=self.device)
            self.obj_category_id = torch.zeros(self.num_envs, device=self.device)

        if not hasattr(self, 'obj_density'):
            self.obj_density = torch.zeros(self.num_envs, device=self.device)
            self.model_bbox_size = torch.zeros(self.num_envs, device=self.device)

        # randomize the list of all possible models in the YCB dataset
        # then sub-scene i will load model model_ids[i % number_of_ycb_objects]
        model_ids = self._batched_episode_rng.choice(self.all_model_ids, replace=True)
        if (
            self.num_envs > 1
            and self.num_envs < len(self.all_model_ids)
            and self.reconfiguration_freq <= 0
            and not WARNED_ONCE
        ):
            WARNED_ONCE = True
            print(
                """There are less parallel environments than total available models to sample.
                Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
                or set reconfiguration_freq to be >= 1."""
            )

        # randomize YCB object, object scale, object density
        self._objs: List[Actor] = []
        self.obj_heights = []
        for i, model_id in enumerate(model_ids):
            # env randomization multiplers
            scale_mult = self.scale_mults[i]
            density_mult = self.density_mults[i]
            # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
            id=f"ycb:{model_id}"
            splits = id.split(":")
            actor_id = ":".join(splits[1:])
            builder, self.obj_density[i], self.model_bbox_size[i] = get_ycb_builder_rma(
                scene=self.scene,
                id=actor_id,
                add_collision=True,
                add_visual=True,
                scale_mult=scale_mult,
                density_mult=density_mult
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
            # save object instance and category id for each env in parallal envs
            self.obj_instance_id[i] = get_object_id(
                task_name="PickSingleYCB",
                model_id=model_id,
                object_list=self.obj_instance_list
            )
            self.obj_category_id[i] = get_object_id(
                task_name="PickSingleYCB",
                model_id=model_id.rsplit("_", 1)[1],
                object_list=self.obj_category_list
            )
        # self.obj represents all objects across parallel environments
        self.obj = Actor.merge(self._objs, name="ycb_object")
        self.add_to_state_dict_registry(self.obj)

        # NOTE: friction not available yet? check ManiSkill documentation / version again
        # randomize object friction
        # for i, obj in enumerate(self._objs):
        #     for shape in obj.collision_shapes:
        #         shape.physical_material.dynamic_friction = self.obj_friction[i]
        #         shape.physical_material.static_friction = self.obj_friction[i]
        #         shape.physical_material.restitution = 0.1

        # goal position
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    """
    Episode Initialization Code:
    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task. Note that these functions are given a env_idx variable.

    `env_idx` is a torch Tensor representing the indices of the parallel environments that are being initialized/reset. This is used
    to support partial resets where some parallel envs might be reset while others are still running (useful for faster RL and evaluation).
    Generally you only need to really use it to determine batch sizes via len(env_idx). ManiSkill helps handle internally a lot of masking
    you might normally need to do when working with GPU simulation. For specific details check out the push_cube.py code
    """

    # NOTE: RNG state of torch is already seeded for you in this part of the code so can freely use torch.rand
    # without reproducibility concerns
    # NOTE: code is written to support partial resets
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """
        Same as parent class, but we initialize observation noise that will be applied throughout episode
        """
        with torch.device(self.device):
            # for 'panda' and 'panda_wristcam', 7 arm joints + 2 gripper joints
            self.joint_pos_dim = 9
            # use 'b' to support partial resets (env_idx is a list of environment IDs that need initialization)
            b = len(env_idx)
            # initialize variables to store observation noise of episode
            if not hasattr(self, 'joint_pos_noise'):
                self.joint_pos_noise = torch.zeros(self.num_envs, self.joint_pos_dim)
                self.obj_pos_noise = torch.zeros(self.num_envs, 3)
                self.obj_rot_noise = torch.zeros(self.num_envs, 4)
            # sample noise that will be applied throughout episode
            if self.obs_noise:
                # noise to proprioception, i.e. robot joint positions (dim = 9)
                joint_pos_l = self.joint_pos_low_scdl(elapsed_steps=self.step_counter)
                joint_pos_h = self.joint_pos_high_scdl(elapsed_steps=self.step_counter)
                joint_pos_noise_sampled = torch.rand((b, self.joint_pos_dim)) * (joint_pos_h - joint_pos_l) + joint_pos_l
                self.joint_pos_noise[env_idx, :] = joint_pos_noise_sampled
                # noise to object position (dim = 3)
                obj_pos_l = self.obj_pos_low_scdl(elapsed_steps=self.step_counter)
                obj_pos_h = self.obj_pos_high_scdl(elapsed_steps=self.step_counter)
                obj_pos_noise_sampled = torch.rand((b, 3)) * (obj_pos_h - obj_pos_l) + obj_pos_l
                self.obj_pos_noise[env_idx, :] = obj_pos_noise_sampled
                # noise to object rotation (dim = 4)
                obj_rot_l = self.obj_rot_low_scdl(elapsed_steps=self.step_counter)
                obj_rot_h = self.obj_rot_high_scdl(elapsed_steps=self.step_counter)
                obj_rot_noise_sampled = random_quaternions(n=b, bounds=[obj_rot_l, obj_rot_h]) # TODO: is this the correct amount of angle randomization compared to RMA^2?
                self.obj_rot_noise[env_idx, :] = obj_rot_noise_sampled
        
        super()._initialize_episode(env_idx, options)


    """
    Modifying observations, goal parameterization, and success conditions for your task:
    The code below all impact some part of `self.step` function
    """

    def step(self, action):
        self.step_counter += 1
        return super().step(action)

    def _get_obs_state_dict(self, info: Dict):
        """
        Get (ground-truth) state-based observations.
        Then apply proprioceptive observation noise.
        """
        # add noise to proprioceptive observations
        obs_agent = self._get_obs_agent()
        obs_agent['qpos'] = obs_agent['qpos'] + self.joint_pos_noise
        return dict(
            # Agent state info: get observations about the agent's state.
            # By default it is proprioceptive observations which include qpos and qvel.
            agent=obs_agent,
            # Get task-relevant extra observations.
            # Usually defined on a task by task basis
            extra=self._get_obs_extra(info),
        ) # when flattened, dim = [N, 56] (excluding obs_act_history)

    def _get_obs_extra(self, info):
        """
        Return a dict of additional observation data specific to the task:
        - This will be included as part of the observation in the "extra" key when obs_mode="state_dict" or any of the visual obs_modes 
        and included as part of a flattened observation when obs_mode="state".
        - Moreover, you have access to the info object which is generated by the `evaluate` function above

        Notes:
        - _get_obs_extra() is called within get_obs(), which in turn is called within step()
        - External disturbance force is applied
        - Compute left and right impulses for gripper joints
        - Dictionary containing privileged information is returned inside obs dict
        - obs dict also contains goal and object state info
        """
        # TODO: when MS3 is finished implement, activate external disturbance force / ask on discord when this will be implemented
        # NOTE: variables are batched
        # NOTE: slight difference in external disturbance implementation compared to RMA^2
        # if object is not grasped in env, disturbance force is not updated
        # if not hasattr(self, 'disturb_force'):
        #     self.disturb_force = torch.zeros(self.num_envs, 3, device=self.device)
        # # check whether object is grasped by gripper
        # grasped = self.agent.is_grasping(self.obj)
        # # add external disturbance force
        # if self.ext_disturbance:
        #     # decay the previous disturbance force
        #     self.disturb_force *= self.force_decay
        #     # sample whether to apply new force with probablity 0.1
        #     mask = self._batched_episode_rng.uniform() < 0.1
        #     mask_tensor = torch.from_numpy(mask).to(self.device)
        #     combined_mask = mask_tensor & grasped
        #     idx = torch.nonzero(combined_mask, as_tuple=True)[0]
        #     # sample 3D force for gaussian distribution
        #     if idx.numel() > 0:
        #         self.disturb_force[idx, :] = torch.from_numpy(self._batched_episode_rng.normal(0, 0.1, 3)).float().to(self.device)[idx, :]
        #         self.disturb_force[idx, :] /= torch.linalg.norm(self.disturb_force, ord=2, dim=-1)[idx]
        #         # sample value by which we scale the force (depends on linear schedule)
        #         self.force_scale_h = self.force_scale_scdl(elapsed_steps=self.step_counter)
        #         self.force_scale = torch.from_numpy(self._batched_episode_rng.uniform(0, self.force_scale_h)).to(self.device)
        #         # scale by object mass and force scale value we just computed
        #         self.disturb_force[idx, :] *= self.obj.mass.to(self.device)[idx] * self.force_scale[idx]
        #         # only apply the force to object if it is grasped
        #         self.obj.apply_force(self.disturb_force) # TODO: ManiSkill upgrade not working (so for now, function was manually added in actors.py), px.cuda_rigid_body_force not implemented yet

        # Privileged env info: magnitudes of the impulses applied by the left and right finger of the gripper
        limpulse = torch.linalg.norm(self.scene.get_pairwise_contact_impulses(self.agent.finger1_link, self.obj), ord=2, dim=-1)
        rimpulse = torch.linalg.norm(self.scene.get_pairwise_contact_impulses(self.agent.finger2_link, self.obj), ord=2, dim=-1)
        
        # Task-specific observations for pick_single_ycb
        obj_pose = self.obj.pose.raw_pose # [N, 7], where last dim represents 3D pose + 4D quaternion
        obj_pose += torch.cat((self.obj_pos_noise, self.obj_rot_noise), dim=-1)
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            is_grasped=info["is_grasped"],
        )
        # object state and goal info
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp.pose.p,
                obj_pose=obj_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,
            )

        # TODO: if obs_act_hist, add proprioceptive history to observation (for adaptation training and evaluation)
        # privileged info (NOTE: by default, obs_noise and ext_disturbance is not included here)
        obs.update(
            obj_ang=obj_pose[:, -4:],              # dim = [N, 4]
            bbox_size=self.model_bbox_size,        # dim = [N, 3]
            obj_density=self.obj_density,          # dim = [N]
            obj_friction=self.obj_friction,        # dim = [N]
            limpulse=limpulse,                     # dim = [N]
            rimpulse=rimpulse,                     # dim = [N]
            obj_instance_id=self.obj_instance_id,  # dim = [N]
            obj_category_id=self.obj_category_id,  # dim = [N]
        )
        return obs