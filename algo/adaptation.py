import os
import time
from typing import Type, Dict

import numpy as np
import torch as th
import gymnasium.spaces as spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor, get_device
from stable_baselines3.common.type_aliases import GymEnv
from torch.utils.tensorboard import SummaryWriter

# from .model.models import ProprioAdaptTConv
from algo.misc import tprint, AverageScalarMeter

# ProprioAdapt class is a training wrapper designed to adapt a model (using a Proximal Policy Optimization (PPO) algorithm)
# in a reinforcement learning environment. The class includes methods for training the model, logging performance metrics,
# and saving model checkpoints
class ProprioAdapt(object):

    def __init__(
        self,
        model: Type[PPO],
        env: GymEnv,
        writer: SummaryWriter,
        save_dir: str,
    ):
        # whether to use the sys_iden approach to have the adaptation predict
        # the privileged information directly
        self.model = model
        self.num_timesteps = 0
        self.device = get_device("auto")
        self.nn_dir = save_dir

        self.policy = model.policy
        self.env = env
        self.logger = writer
        # self.eval_callback = eval_callback

        self.action_space = env.action_space
        self.step_reward = th.zeros(env.num_envs, dtype=th.float32)
        self.step_length = th.zeros(env.num_envs, dtype=th.float32)
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.best_rewards = -np.inf
        self.best_succ_rate = -np.inf
        # ---- Optim ----
        adapt_params = []
        for name, p in self.policy.named_parameters():
            if 'adapt_tconv' in name:
                adapt_params.append(p)
            else:
                p.requires_grad = False
        self.optim = th.optim.Adam(adapt_params, lr=1e-4
                                   # lr=1e-3
                                   )

    def get_env(self):
        return None

    # trains the adaptation module of the model in the provided reinforcement learning environment
    def learn(self, ):
        '''training the adaptation module
        '''

        # Initialization
        # method first checks if the model has the attribute adaptation_steps, which keeps track of the number of training steps
        # If this attribute exists, it initializes n_steps and best_succ_rate from the model;
        # otherwise, it sets n_steps to 0 and initializes best_succ_rate to 0
        # The success rate (succ_rate) is initialized to 0
        if hasattr(self.model, 'adaptation_steps'):
            n_steps = self.model.adaptation_steps
            self.best_succ_rate = self.model.best_succ_rate
        else:
            n_steps = 0
            self.model.best_succ_rate = 0
        self.succ_rate = 0

        # Time Tracking
        # These lines initialize the time tracking variables to monitor training progress and calculate the frames per second (FPS)
        _t = time.time()
        _last_t = time.time()

        # Reset Environment
        # The environment is reset, and the initial observation (_last_obs) is stored
        # The number of environments (n_envs) is retrieved from the environment
        self._last_obs = self.env.reset()
        n_envs = self.env.num_envs
        assert self._last_obs is not None, "No previous observation was provided"

        # Main Training Loop
        # The loop runs until n_steps exceeds 1 million, controlling the total number of training steps
        while n_steps <= 1e6:
            # Convert to pytorch tensor
            obs_tensor = obs_as_tensor(self._last_obs, self.device)

            # The .detach() function ensures that no gradients are tracked for the current observation (important for efficiency)
            for key, tensor in obs_tensor.items():
                obs_tensor[key] = tensor.detach()
            actions, _, _, e, e_gt = self.policy(obs_tensor, adapt_trn=True) # note: this is the actor critic policy defined in policy_rma (but here adapt_trn is set to True)

            # Loss = MSE between ground truth environment embedding vs. predicted environment embedding from adaptation module
            loss = ((e - e_gt.detach())**2).mean()

            # Backpropagation and Optimization
            # optimizer’s gradients are cleared, then the loss is backpropagated to update the model’s weights using the Adam optimizer
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Rescale and perform action
            # The predicted actions are detached from the computation graph and converted to NumPy arrays
            actions = actions.detach().cpu().numpy()

            # The actions are clipped to ensure they remain within the valid bounds of the action space
            # (important for continuous action spaces)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low,
                                          self.action_space.high)
            new_obs, rewards, dones, infos = self.env.step(clipped_actions)

            # Statistics and Performance Tracking
            # Rewards and dones (episode termination indicators) are converted into PyTorch tensors
            rewards, dones = th.tensor(rewards), th.tensor(dones)
            # reset proprio_buffer if the episode is finished
            self.policy.reset_buffer(dones=dones)

            # f any environment episode has ended (dones == 1), the success rate is calculated by
            # summing successes across the environments and dividing by the total number of environments (n_envs)
            if th.any(dones == 1):
                n_succ = sum([infos[i]['success'] for i in range(50)])
                self.succ_rate = n_succ / n_envs

            # Updating Step Statistics
            # Step-based rewards and lengths are updated for each environment.
            # The statistics for rewards and episode lengths are updated for non-done environments
            self.step_reward += rewards
            self.step_length += 1
            done_indices = dones.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])
            self.loss = loss.item()

            # Loss and Step Adjustments
            # These lines ensure that rewards and step lengths are reset to zero for environments
            # that have ended (dones), so they do not accumulate incorrectly
            not_dones = 1.0 - dones.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.n_steps = n_steps
            self.model.adaptation_steps = n_steps
            self.log_tensorboard()

            # Model Checkpointing
            if n_steps % 1e4 == 0:
                self.save(
                    os.path.join(self.nn_dir, f'{int(self.n_steps//1e4)}0K'))
                self.save(os.path.join(self.nn_dir, f'latest_model'))

            # Best Success Rate Tracking
            # If the current success rate is better than the previous best, the model is saved as the best model
            if self.succ_rate >= self.best_succ_rate:
                self.save(os.path.join(self.nn_dir, f'best_model'))
                self.best_succ_rate = self.succ_rate
                self.model.best_succ_rate = self.best_succ_rate

            # Logging and FPS Calculation
            # Calculates and prints the FPS for the entire run (all_fps) and for the most recent batch of steps (last_fps)
            # Prints key information such as the current step count, loss, success rate, and best success rate
            all_fps = self.num_timesteps / (time.time() - _t)
            last_fps = self.env.num_envs / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(n_steps // 1e3):04}k | FPS: {all_fps:.1f} | ' \
                          f'Current Loss: {loss.item():.5f} | ' \
                          f'Succ. Rate: {self.succ_rate:.5f} | ' \
                          f'Best Succ. Rate: {self.best_succ_rate:.5f}'
            tprint(info_string)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Environment Update
            self._last_obs = new_obs
            self.num_timesteps += self.env.num_envs
    
    # log_tensorboard method is responsible for logging training statistics to TensorBoard.
    # TensorBoard is a tool for visualizing various metrics during training,
    # which is particularly useful for monitoring and debugging machine learning models.
    def log_tensorboard(self):
        self.logger.add_scalar('adaptation_training/loss', self.loss,
                               self.n_steps)
        self.logger.add_scalar('adaptation_training/mean_episode_rewards',
                               self.mean_eps_reward.get_mean(), self.n_steps)
        self.logger.add_scalar('adaptation_training/mean_episode_length',
                               self.mean_eps_length.get_mean(), self.n_steps)
        self.logger.add_scalar('adaptation_training/success_rate',
                               self.succ_rate, self.n_steps)

    def save(self, name):
        self.model.save(name)

    def compute_mean_adaptor_loss(self, ):
        '''compute the mean adaptor loss during training
        '''
        # if hasattr(self.model, 'adaptation_steps'):
        #     n_steps = self.model.adaptation_steps
        #     self.best_succ_rate = self.model.best_succ_rate
        # else:
        n_steps = 0
        self.succ_rate = 0

        _t = time.time()
        _last_t = time.time()
        self._last_obs = self.env.reset()
        n_envs = self.env.num_envs
        losses = []
        assert self._last_obs is not None, "No previous observation was provided"
        while n_steps <= 200 * 10:
            # Convert to pytorch tensor
            obs_tensor = obs_as_tensor(self._last_obs, self.device)

            for key, tensor in obs_tensor.items():
                obs_tensor[key] = tensor.detach()
            actions, _, _, e, e_gt = self.policy(obs_tensor, adapt_trn=True)

            loss = ((e - e_gt.detach())**2).mean()
            losses.append(loss.item())

            # Rescale and perform action
            actions = actions.detach().cpu().numpy()
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low,
                                          self.action_space.high)
            new_obs, rewards, dones, infos = self.env.step(clipped_actions)

            # Statistics
            rewards, dones = th.tensor(rewards), th.tensor(dones)
            # reset proprio_buffer if the episode is finished
            self.policy.reset_buffer(dones=dones)

            if th.any(dones == 1):
                n_succ = sum([infos[i]['success'] for i in range(50)])
                self.succ_rate = n_succ / n_envs

            self.step_reward += rewards
            self.step_length += 1
            done_indices = dones.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])
            self.loss = loss.item()

            not_dones = 1.0 - dones.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.n_steps = n_steps
            self.model.adaptation_steps = n_steps

            if self.succ_rate >= self.best_succ_rate:
                self.save(os.path.join(self.nn_dir, f'best_model'))
                self.best_succ_rate = self.succ_rate
                self.model.best_succ_rate = self.best_succ_rate

            all_fps = self.num_timesteps / (time.time() - _t)
            last_fps = self.env.num_envs / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(n_steps // 1e3):04}k | FPS: {all_fps:.1f} | ' \
                          f'Current Loss: {loss.item():.5f} | ' \
                          f'Succ. Rate: {self.succ_rate:.5f} | ' \
                          f'Best Succ. Rate: {self.best_succ_rate:.5f}'
            tprint(info_string)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            self._last_obs = new_obs
            self.num_timesteps += self.env.num_envs
        return np.mean(losses)
