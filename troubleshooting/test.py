# Import required packages
import gymnasium as gym
from tqdm.notebook import tqdm
import numpy as np
import mani_skill2.envs
import matplotlib.pyplot as plt

#@title 1.1 Choose an environment, observation mode, control mode, and reward
#@markdown Run this cell to display the action space of the chosen controller as well as the current view of the environment.
#@markdown The main part of the view is our view of the environment. The two views on the right are the RGB and Depth images from a third-person camera
#@markdown and a hand-mounted camera. The two views on the right are also the exact perspectives and orientations the robot gets in the rgbd and point cloud observation modes

# Can be any env_id from the list of Rigid-Body envs: https://haosulab.github.io/ManiSkill2/concepts/environments.html#rigid-body
# and Soft-Body envs: https://haosulab.github.io/ManiSkill2/concepts/environments.html#soft-body

# This tutorial allows you to play with 4 environments out of a total of 20 environments that ManiSkill provides
env_id = "PickCube-v0" #@param ['PickCube-v0', 'PegInsertionSide-v0', 'StackCube-v0', 'PlugCharger-v0']

# choose an observation type and space, see https://haosulab.github.io/ManiSkill2/concepts/observation.html for details
obs_mode = "rgbd" #@param can be one of ['pointcloud', 'rgbd', 'state_dict', 'state']

# choose a controller type / action space, see https://haosulab.github.io/ManiSkill2/concepts/controllers.html for a full list
control_mode = "pd_joint_delta_pos" #@param can be one of ['pd_ee_delta_pose', 'pd_ee_delta_pos', 'pd_joint_delta_pos', 'arm_pd_joint_pos_vel']

reward_mode = "dense" #@param can be one of ['sparse', 'dense']

# create an environment with our configs and then reset to a clean state
env = gym.make(env_id,
               obs_mode=obs_mode,
               reward_mode=reward_mode,
               control_mode=control_mode,
               enable_shadow=False)
obs, _ = env.reset()
print("Action Space:", env.action_space)

# take a look at the current state
img = env.unwrapped.render_cameras()
plt.figure(figsize=(10,6))
plt.title("Current State viewed through all RGB and Depth cameras")
plt.imshow(img)

# Save the rendered image to disk --> can view it on VS code
output_path = "/users/joo/4yp/rma4rma/troubleshooting/renders/test_render_two_robots.png"
plt.imsave(output_path, img)
print(f"Image saved to {output_path}")

env.close()