# Import required packages
import gymnasium as gym
import mani_skill.envs
import torch
import time
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper

# ---------------------------------------------------------------------------------------------------------------- #
# test rendering maniskill environment for state + visual RL --> test simulation FPS
num_envs = 512 # you can go up higher on better GPUs, this is mostly memory constrained
env = gym.make("TwoRobotPickCube-v1", num_envs=num_envs, obs_mode="rgbd")
env.unwrapped.print_sim_details()
obs, _ = env.reset(seed=0)
done = False
start_time = time.time()
total_rew = 0

# loop which executes actions until one of the environments reaches terminal state --> use to test out FPS of simulation
while not done:
    # note that env.action_space is now a batched action space
    # in "TwoRobotPickCube-v1" task, there are two panda robot arms
    # env.action_space.sample() returns a dictonary: {'panda_wristcam-0': (512, 8), 'panda_wristcam-1': (512, 8)} --> batched because first dimension represents batch of actions for 512 environments
    # to see more details of implementation see mani_skills.env --> .tasks --> .tabletop --> .two_robot_pick_cube
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    done = (terminated | truncated).any() # stop if any environment terminates/truncates
N = num_envs * info["elapsed_steps"][0].item()
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")

# ---------------------------------------------------------------------------------------------------------------- #
# visualize the image data from the environment and inspect the data
print(obs.keys())
print(obs['sensor_data'].keys())
print(obs['sensor_data']['base_camera'].keys())
print(obs['sensor_data']['base_camera']['rgb'].shape)
import matplotlib.pyplot as plt
plt.imshow(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy())

# Save the plot to a file
plot_path =  "/users/joo/4yp/rma4rma/quickstart_demo/renders/test_render_two_robots.png"
plt.savefig(plot_path, bbox_inches='tight')
plt.close()  # Close the plot to free up resources

print(f"Plot saved to {plot_path}")

# ---------------------------------------------------------------------------------------------------------------- #
# try recording episodes
# to make it look a little more realistic, we will enable shadows which make the default lighting cast shadows
env = gym.make("TwoRobotPickCube-v1", num_envs=4, render_mode="rgb_array", enable_shadow=True)

# action space is initially a dictionary, so flatten this into an array so it is compatible with RecordEpisode wrapper
env = FlattenActionSpaceWrapper(env)

# record episode wrapper
env = RecordEpisode(
    env,
    "./videos", # the directory to save replay videos and trajectories to
    # on GPU sim we record intervals, not by single episodes as there are multiple envs
    # each 100 steps a new video is saved
    max_steps_per_video=100
)

# step through the environment with random actions
obs, _ = env.reset()
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # env.render_human() # will render with a window if possible
env.close()
from IPython.display import Video
Video("./videos/test_record_episode_two_robots.mp4", embed=True, width=640) # Watch the replay