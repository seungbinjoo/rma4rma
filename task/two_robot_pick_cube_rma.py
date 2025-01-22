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
from mani_skill.envs.tasks.tabletop import TwoRobotPickCube
from mani_skill.utils.registration import register_env
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder

YCB_DATASET = dict()
WARNED_ONCE = False

@register_env("PickSingleYCBRMA-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class TwoRobotPickCubeRMA(TwoRobotPickCube):
    """
    TwoRobotPickCube env where:
        - Observation state_dict additionally includes privileged info and object id, which are used in RMA training
        - Domain randomization (with linear scheduling) is applied
            - Environment variables: object scale, object density, objeect friction
            - Disturbance: external force applied to object every step (with force decay)
            - Observation noise: joint position (proprioception) noise, object position noise, object rotation noise
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "panda_wristcam"]
    agent: Union[Panda, Fetch, PandaWristCam]

    