# See custom task building: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/index.html
# See simulation 101: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/simulation_101.html
# See PushCube environment example: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/push_cube.py
# See environment template: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/template.py
# See reproducability and RNG: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/rng.html
# See custom robots: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html
# See domain randomization: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html

import os
from typing import Dict, List, Union, Tuple
from pathlib import Path
import numpy as np
import torch
import sapien
from algo.misc import linear_schedule, get_ycb_builder_rma, get_object_id, calculate_flattened_dim

# Maniskill-specific imports
from mani_skill import ASSET_DIR
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.utils import randomization
from task.panda_wristcam_rma import PandaWristCam
from mani_skill.utils.common import flatten_state_dict
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop import TwoRobotPickCube
from mani_skill.utils.registration import register_env
from mani_skill.utils.io_utils import load_json
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder

YCB_DATASET = dict()
WARNED_ONCE = False

@register_env("TwoRobotPickCubeRMA-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class TwoRobotPickCubeRMA(TwoRobotPickCube):
    """
    TwoRobotPickCube env where:
        - Observation state_dict additionally includes privileged info and object id, which are used in RMA training
        - Domain randomization (with linear scheduling) is applied
            - Environment variables: object scale, object density, objeect friction
            - Disturbance: external force applied to object every step (with force decay)
            - Observation noise: joint position (proprioception) noise, object position noise, object rotation noise
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None
        self.all_model_ids = np.array(
            list(
                load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
            )
        )
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    """
    Reconfiguration Code:
    Below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general.
    Environment frozen after reconfiguration. Assets cannot be added or removed until reconfigure() is called again.
    In CPU simulation, for some tasks these may need to be called multiple times if you need to swap out object assets.
    In GPU simulation these will only ever be called once.
    """

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

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
            # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
            id=f"ycb:{model_id}"
            splits = id.split(":")
            actor_id = ":".join(splits[1:])
            builder, _, _ = get_ycb_builder_rma(
                scene=self.scene,
                id=actor_id,
                add_collision=True,
                add_visual=True,
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        # self.cube represents all objects across parallel environments
        self.cube = Actor.merge(self._objs, name="ycb_object")
        self.add_to_state_dict_registry(self.cube)

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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.left_init_qpos = self.left_agent.robot.get_qpos()
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            # ensure cube is spawned on the left side of the table
            xyz[:, 1] = -0.15 - torch.rand((b,)) * 0.1 + 0.05
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            goal_xyz[:, 1] = 0.15 + torch.rand((b,)) * 0.1 - 0.05
            goal_xyz[:, 2] = torch.rand((b,)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))