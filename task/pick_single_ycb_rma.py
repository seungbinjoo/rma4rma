# See custom task building: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/index.html
# See simulation 101: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/simulation_101.html
# See PushCube environment example: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/push_cube.py
# See environment template: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/template.py
# See reproducability and RNG: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/rng.html
# See custom robots: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html
# See domain randomization: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html

import os
from typing import Dict, List, Union
from pathlib import Path
import numpy as np
import torch
import sapien
from algo.misc import linear_schedule, get_ycb_builder_rma, get_object_id, calculate_flattened_dim

# Maniskill-specific imports
from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from task.panda_wristcam_rma import PandaWristCam
from mani_skill.utils.common import flatten_state_dict
from mani_skill.sensors.camera import Camera, CameraConfig, parse_camera_configs, update_camera_configs_from_dict
from mani_skill.sensors.depth_camera import StereoDepthCamera, StereoDepthCameraConfig
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
    
    SUPPORTED_ROBOTS = ["panda", "fetch", "panda_wristcam"]
    agent: Union[Panda, Fetch, PandaWristCam]

    def __init__(
            self,
            *args,
            robot_uids="panda_wristcam",
            randomized_env=True,
            obs_noise=True,
            ext_disturbance=True,
            phase="PolicyTraining",
            **kwargs
    ):
        self.phase = phase
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

        if self.phase == "AdaptationTraining":
            self.proprio_hist_dim = 50
            self.action_space_dim = calculate_flattened_dim(self.single_action_space)
            self.proprio_features_dim = self._get_obs_agent().shape[-1] + self.action_space_dim
            self.proprio = torch.zeros(self.num_envs, self.proprio_hist_dim, self.proprio_features_dim, device=self.device)
            self.proprio_idx = torch.zeros(self.num_envs) # to keep track of timestep index within proprioceptive history, for each environment

        super().__init__(robot_uids=robot_uids, *args, **kwargs)
    
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
        init_step, end_step = 3e7, 5e7 # start randomization at 30M step, then linearly ramp until 50M steps # TODO: this should be passed in as args in base_policy.py
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
        #     for shape  in obj.collision_shapes:
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

    def _setup_sensors(self, options: dict):
        """
        Setup sensor configurations and the sensor objects in the scene. Called by `self._reconfigure`
        Same as parent class but we exclude 'base_camera' which is initialized using self._default_sensor_configs
        """

        # First create all the configurations
        self._sensor_configs = dict()

        # Add task/external sensors
        # self._sensor_configs.update(parse_camera_configs(self._default_sensor_configs))

        # Add agent sensors
        self._agent_sensor_configs = dict()
        if self.agent is not None:
            self._agent_sensor_configs = parse_camera_configs(self.agent._sensor_configs)
            self._sensor_configs.update(self._agent_sensor_configs)

        # Add human render camera configs
        self._human_render_camera_configs = parse_camera_configs(
            self._default_human_render_camera_configs
        )

        self._viewer_camera_config = parse_camera_configs(
            self._default_viewer_camera_configs
        )

        # Override camera configurations with user supplied configurations
        if self._custom_sensor_configs is not None:
            update_camera_configs_from_dict(
                self._sensor_configs, self._custom_sensor_configs
            )
        if self._custom_human_render_camera_configs is not None:
            update_camera_configs_from_dict(
                self._human_render_camera_configs,
                self._custom_human_render_camera_configs,
            )
        if self._custom_viewer_camera_configs is not None:
            update_camera_configs_from_dict(
                self._viewer_camera_config,
                self._custom_viewer_camera_configs,
            )
        self._viewer_camera_config = self._viewer_camera_config["viewer"]

        # Now we instantiate the actual sensor objects
        self._sensors = dict()

        for uid, sensor_config in self._sensor_configs.items():
            if uid in self._agent_sensor_configs:
                articulation = self.agent.robot
            else:
                articulation = None
            if isinstance(sensor_config, StereoDepthCameraConfig):
                sensor_cls = StereoDepthCamera
            elif isinstance(sensor_config, CameraConfig):
                sensor_cls = Camera
            self._sensors[uid] = sensor_cls(
                sensor_config,
                self.scene,
                articulation=articulation,
            )

        # Cameras for rendering only
        self._human_render_cameras = dict()
        for uid, camera_config in self._human_render_camera_configs.items():
            self._human_render_cameras[uid] = Camera(
                camera_config,
                self.scene,
            )

        self.scene.sensors = self._sensors
        self.scene.human_render_cameras = self._human_render_cameras

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
            
            # for envs that are resetting, set proprioceptive histories to zeros
            if self.phase == "AdaptationTraining":
                self.proprio[env_idx, ...] = torch.zeros(b, self.proprio_hist_dim, self.proprio_features_dim, device=self.device)
                self.proprio_idx[env_idx] = torch.zeros(b, device=self.device)
                
        
        super()._initialize_episode(env_idx, options)


    """
    Modifying observations, goal parameterization, and success conditions for your task:
    The code below all impact some part of `self.step` function
    """

    def step(self, action):
        self.step_counter += self.num_envs
        return super().step(action)

    def _get_obs_state_dict(self, info: Dict):
        """
        Get (ground-truth) state-based observations (used during policy training phase).
        Then apply proprioceptive observation noise.
        """
        # add noise to proprioceptive observations
        obs_agent = self._get_obs_agent()
        obs_agent['qpos'] = obs_agent['qpos'] + self.joint_pos_noise
        state, goal, env_params, obj_instance_id, obj_category_id = self._get_obs_extra(info)
        return dict(
            # Agent state info: get observations about the agent's state, by default it is proprioceptive observations which include qpos and qvel.
            agent=torch.cat((obs_agent['qpos'], obs_agent['qvel']), dim=-1), # [N, 18]
            # Get task-relevant extra observations, usually defined on a task by task basis
            state=state,
            goal=goal,
            env_params=env_params,
            obj_instance_id=obj_instance_id,
            obj_category_id=obj_category_id,
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
        - If in policy training phase:
            - Compute left and right impulses for gripper joints
            - Dictionary containing privileged information is returned inside obs dict
            - obs dict also contains goal and object state info
        - If in adaptation training or evaluation phase:
            - Returns state, goal, proprio (which is proprioceptive history)
        """
        # TODO: when MS3 is finished, implement; activate external disturbance force / ask on discord when this will be implemented
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
        if self.phase == "PolicyTraining":
            limpulse = torch.linalg.norm(self.scene.get_pairwise_contact_impulses(self.agent.finger1_link, self.obj), ord=2, dim=-1)
            rimpulse = torch.linalg.norm(self.scene.get_pairwise_contact_impulses(self.agent.finger2_link, self.obj), ord=2, dim=-1)
        
        # Task-specific observations for pick_single_ycb
        obj_pose = self.obj.pose.raw_pose # [N, 7], where last dim represents 3D pose + 4D quaternion
        obj_pose += torch.cat((self.obj_pos_noise, self.obj_rot_noise), dim=-1)
        state = torch.cat((                                     # state [N, 18]
            self.agent.tcp.pose.raw_pose,                       # tcp_pose [N, 7]
            info["is_grasped"].unsqueeze(-1),                   # grasp state [N, 1]
            obj_pose,                                           # object pose [N, 7]
            self.obj.pose.p - self.agent.tcp.pose.p             # tcp_to_obj_pos [N, 3]
        ), dim=-1)
        goal = torch.cat((                                      # goal [N, 9]
            self.goal_site.pose.p,                              # goal_pos [N, 3]
            self.goal_site.pose.p - self.agent.tcp.pose.p,      # tcp_to_goal_pos [N, 3]
            self.goal_site.pose.p - self.obj.pose.p             # obj_to_goal_pos [N, 3]
        ), dim=-1)
        # privileged info (NOTE: by default, obs_noise and ext_disturbance is not included here)
        if self.phase == "PolicyTraining":
            env_params = torch.cat((                             # env params [N, 9]
                obj_pose[:, -4:],                                # dim = [N, 4]
                self.model_bbox_size.unsqueeze(-1),              # dim = [N, 1]
                self.obj_density.unsqueeze(-1),                  # dim = [N, 1]
                self.obj_friction.unsqueeze(-1),                 # dim = [N, 1]
                limpulse.unsqueeze(-1),                          # dim = [N, 1]
                rimpulse.unsqueeze(-1)                           # dim = [N, 1]
            ), dim=-1)
            obj_instance_id = self.obj_instance_id.unsqueeze(-1) # dim = [N,]
            obj_category_id = self.obj_category_id.unsqueeze(-1) # dim = [N,]
            return state, goal, env_params, obj_instance_id, obj_category_id
        # add proprioceptive history to observation (for adaptation training and evaluation)
        if self.phase == "AdaptationTraining":
            # fill proprio buffer with observation; note that action will be filled in later in Agent class
            self.proprio[:, self.proprio_idx, :] = torch.cat((self._get_obs_agent, torch.zeros(self.num_envs, self.action_space_dim, device=self._render_device)), dim=-1)
            self.proprio_idx = self.proprio_idx + 1
            return state, goal, self.proprio, self.proprio_idx

    def _get_obs_with_sensor_data(self, info: Dict, apply_texture_transforms: bool = True) -> dict:
        """
        Get the observation with sensor data (used during adaptation training and evaluation phase)
        Same as parent class but:
        - we apply noise to agent proprioceptive data
        - we have more state, goal, env_params, obj_instance_id, obj_category_id, sensor_param, sensor_data keys in the observation dictionary
        - sensor_param is flattened
        - sensor_data is [N, 32, 32, 1], where 1 is for depth
        """
        # observations same as when obs_mode='state_dict', during base policy training phase
        obs_agent = self._get_obs_agent()
        obs_agent['qpos'] = obs_agent['qpos'] + self.joint_pos_noise
        state, goal, proprio, proprio_idx = self._get_obs_extra(info)

        # flatten camera parameters
        sensor_param = {}
        for k, v in self.get_sensor_params()['hand_camera'].items():
            sensor_param[k] = torch.flatten(v, start_dim=1)

        # depth image
        sensor_data = self._get_obs_sensor_data(apply_texture_transforms)['hand_camera']['depth']

        return dict(
            agent=torch.cat((obs_agent['qpos'], obs_agent['qvel']), dim=-1), # [N, 18]
            state=state,                                                     # [N, 18]
            goal=goal,                                                       # [N, 9]
            proprio=proprio,                                                 # FIXME: [N, 50, 36]
            proprio_idx=proprio_idx,                                         # FIXME: 
            sensor_param=flatten_state_dict(sensor_param, use_torch=True),   # [N, 37]
            sensor_data=sensor_data,                                         # [N, 32, 32, 1]
        )