import gymnasium as gym
import mani_skill.envs
import torch
import time
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from task.pick_single_ycb_rma import PickSingleYCBEnvRMA

# recording episodes
# to make it look a little more realistic, we will enable shadows which make the default lighting cast shadows
env = gym.make("PickSingleYCBRMA-v1", num_envs=4, render_mode="rgb_array", enable_shadow=True)

# record episode wrapper
env = RecordEpisode(
    env,
    "./videos", # the directory to save replay videos and trajectories to
    # on GPU sim we record intervals, not by single episodes as there are multiple envs
    # each 100 steps a new video is saved
    max_steps_per_video=100
)

# step through the environment with random actions
obs, _ = env.reset(seed=0)
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # env.render_human() # will render with a window if possible
env.close()
from IPython.display import Video
Video("./videos/test_pick_single_ycb.mp4", embed=True, width=640) # Watch the replay