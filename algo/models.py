import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import wandb

# Neural network initialization method
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Define actor-critic Agent class
class Agent(nn.Module):
    def __init__(self, envs, obs_dict):
        super().__init__()
        # Actor and critic networks
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)
        
        # Networks used for feature extraction
        self.obj_emb_dim = 32
        # Policy training phase
        if envs.obs_mode == "state_dict":
            # Environment encoder
            input_dim, output_dim = compute_env_encoder_dims(envs.id, self.obj_emb_dim)
            self.env_encoder = EnvEncoder(input_dim, output_dim)
            # Object embedding networks (category and instance dictionaries)
            self.obj_instance_emb = torch.nn.Embedding(80, self.obj_emb_dim)
            self.obj_category_emb = torch.nn.Embedding(50, self.obj_emb_dim)
        # Adaptation training phase
        elif envs.obs_mode == "rgbd":
            pass
        # Evaluation phase
        else:
            pass

    def get_features(self, env_params, obs_dict):
        """
        For policy training phase:
            Input: environment parameters (env_params) and obs_dict = {'agent_state': [N,], 'object_state': [N,], 'goal_info': [N,]} where N = batch_size
            Output: vector input to the policy --> concatenation of agent state, object state, goal_info, env embedding
        """
        obj_instance_emb = self.obj_instance_emb(obs_dict['object1_id'])
        obj_category_emb = self.obj_category_emb(obs_dict['object1_type_id'])
        env_encoder_input = torch.cat(obj_instance_emb, obj_category_emb, env_params, dim=-1)
        env_embedding = self.env_encoder(env_encoder_input)
        input_to_policy = torch.cat(
            obs_dict['agent_state'],
            obs_dict['object_state'],
            obs_dict['goal_info'],
            env_embedding,
        )
        return input_to_policy
    def get_value(self, env_params, obs_dict):
        input_to_policy = self.get_features(env_params, obs_dict)
        return self.critic(input_to_policy)
    def get_action(self, env_params, obs_dict, deterministic=False):
        input_to_policy = self.get_features(env_params, obs_dict)
        action_mean = self.actor_mean(input_to_policy)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, env_params, obs_dict, action=None):
        input_to_policy = self.get_features(env_params, obs_dict)
        action_mean = self.actor_mean(input_to_policy)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(input_to_policy)

# Env encoder
class EnvEncoder(nn.Module):
    """
    Input: state and environment parameters
    Output: environment embedding (z_t)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.env_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU(),
        )
    def forward(self, x):
        return self.env_encoder(x)

# Logic for computing input and output dimension for env encoder network
def compute_env_encoder_dims(env_id, obj_emb_dim):
    if env_id == "PickSingleYCB-v1":
        input_dim = 4 + 3 + 4
        input_dim += obj_emb_dim * 2
        output_dim = input_dim - 4
    return input_dim, output_dim

# Logging of scalar values during training or evaluation (to WandB or TensorBoard)
class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()