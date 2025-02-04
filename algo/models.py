import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from mani_skill.utils.common import flatten_state_dict
import wandb
from einops import rearrange

# Neural network initialization method
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Define actor-critic Agent class
class Agent(nn.Module):
    def __init__(self, envs, observation_space_dim, action_space_dim, env_id="PickSingleYCBRMA-v1", phase="PolicyTraining"):
        super().__init__()
        self.phase = phase
        self.obj_emb_dim = 32
        self.env_params_dim = 4 + 5 # obj_pose, model_bbox_size, obj_friction, obj_density, limpulse, rimpulse
        self.action = torch.zeros(action_space_dim) # internally track action so we can store obs action hist
        self.action_space_dim = action_space_dim
        # Networks used for feature extraction
        # Environment encoder
        input_dim, output_dim = compute_env_encoder_dims(env_id, self.obj_emb_dim)
        self.env_encoder = EnvEncoder(input_dim, output_dim)
        # Object embedding networks (category and instance dictionaries)
        self.obj_instance_emb = torch.nn.Embedding(80, self.obj_emb_dim)
        self.obj_category_emb = torch.nn.Embedding(50, self.obj_emb_dim)
        # Policy training phase
        if self.phase == "PolicyTraining":
            features_dim = (
                observation_space_dim
                - (self.env_params_dim + 2) # privileged info
                + 2*self.obj_emb_dim + self.env_params_dim # object instance and category embeddings + env params
                - 4 # after env encoder
            )
        # Adaptation training or evaluation phase
        if self.phase == "AdaptationTraining" or "Evaluation":
            self.adaptation_net = AdaptationNet(proprio_hist_dim=50, adapt_out_dim=output_dim)
            features_dim = (
                observation_space_dim
                - 32*32 - 37 # minus sensor_data dim and sensor_param dim
                - 50*26 - 1 # minus proprio and proprio_idx
                + 2*self.obj_emb_dim + self.env_params_dim # object instance and category embeddings + env params
                - 4 # after env encoder
                - 11 # FIXME: make these calculations here clearer and more readable
            )
        else:
            raise ValueError("Phase must be policy training, adaptation training, or evaluation.")
        # Actor and critic networks
        self.critic = nn.Sequential(
            layer_init(nn.Linear(features_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(features_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_space_dim), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_space_dim) * -0.5)

    def get_features(self, obs):
        """
        For policy training phase:
            Input:  obs = [N, 56], which includes agent info, object state info, goal info, privileged info
            Output: vector input to the policy --> concatenation of agent state, object state, goal_info, env embedding
        """
        # to obtain env_embedding, pass random environment parameters and state info through env encoder
        if self.phase == "PolicyTraining":
            obj_instance_emb = self.obj_instance_emb(obs['obj_instance_id'].long()).squeeze(1)
            obj_category_emb = self.obj_category_emb(obs['obj_category_id'].long()).squeeze(1)
            env_params = obs['env_params']
            env_encoder_input = torch.cat((obj_instance_emb, obj_category_emb, env_params), dim=-1)    # [N, 73]
            env_embedding = self.env_encoder(env_encoder_input)                                        # [N, 69]
        # to obtain env_embedding, use proprioceptive history and depth CNN
        elif self.phase == "AdaptationTraining":
            env_embedding = self.adaptation_net(obs)
        # construct input to the policy
        input_to_policy = torch.cat((   # [N, 114]
            obs['agent'],               # agent state
            obs['state'],               # state (e.g. object state, tcp pose, etc.)
            obs['goal'],                # goal info
            env_embedding),             # privileged info (including object instance and category id)
            dim=-1
        )
        return input_to_policy
    
    def get_env_embeddings(self, obs):
        # get ground truth environment embedding (using base policy)
        obj_instance_emb = self.obj_instance_emb(obs['obj_instance_id'].long()).squeeze(1)
        obj_category_emb = self.obj_category_emb(obs['obj_category_id'].long()).squeeze(1)
        env_params = obs['env_params']
        env_encoder_input = torch.cat((obj_instance_emb, obj_category_emb, env_params), dim=-1)    # [N, 73]
        env_embedding_gt = self.env_encoder(env_encoder_input)                                     # [N, 69]
        # get predicted environment embedding (using adaptation module)
        env_embedding_pred = self.adaptation_net(obs)
        return env_embedding_gt, env_embedding_pred
    
    def action_to_proprio(self, obs):
        # insert previous action into proprioceptive (observation-action) history
        proprio_idx = obs['proprio_idx']
        env_idx = torch.arange(obs['proprio'].shape[0]) # [0, 1, 2, ..., 511]
        obs['proprio'][env_idx, proprio_idx, -self.action_space_dim:] = self.action
        return obs

    def get_value(self, obs):
        input_to_policy = self.get_features(obs)
        return self.critic(input_to_policy)
    def get_action(self, obs, deterministic=False):
        input_to_policy = self.get_features(obs)
        action_mean = self.actor_mean(input_to_policy)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        self.action = probs.sample()
        return self.action
    def get_action_and_value(self, obs, action=None):
        input_to_policy = self.get_features(obs)
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
    Input: object instance and category embeddings,
    and env params (obj_pose, model_bbox_size, obj_friction, obj_density, limpulse, rimpulse)
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
        return self.env_encoder(x.float())

# Logic for computing input and output dimension for env encoder network
def compute_env_encoder_dims(env_id, obj_emb_dim):
    if env_id == "PickSingleYCBRMA-v1":
        input_dim = 4 + 5
        input_dim += obj_emb_dim * 2
        output_dim = input_dim - 4
    return input_dim, output_dim

class AdaptationNet(nn.Module):
    def __init__(
            self,
            proprio_hist_dim=50, # 50 timesteps in each proprioceptive observation
            proprio_features_dim = 26, # feature dim is 26 for each proprioceptive observation
            adapt_out_dim=69, # by default 69, which is the env embedding dim for "PickSingleYCBRMA-v1" task
        ):
        super(AdaptationNet, self).__init__()
        depth_cnn_out_dim = 64
        camera_param_dim = 3*4 + 4*4 + 3*3 # extrinsic_cv, cam2world_gl, instrinsic_cv, dim = 37
        self.depth_cnn = DepthCNN(out_dim=depth_cnn_out_dim)
        self.proprio_cnn = ProprioCNN(proprio_hist_dim)
        self.fc1 = nn.Linear(2 * proprio_features_dim + camera_param_dim + depth_cnn_out_dim, adapt_out_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(adapt_out_dim, adapt_out_dim)

    def forward(self, x):
        # proprioception, sensor data (depth camera data), camera parameters
        proprio, sensor_data, sensor_param = x['proprio'], x['sensor_data'], x['sensor_param']
        proprio = self.proprio_cnn(proprio)
        obs = [proprio]
        sensor_data = self.depth_cnn(sensor_data)
        obs.extend([sensor_data, sensor_param])
        x = self.fc1(torch.cat(obs, dim=-1))
        x = self.fc2(self.relu(x))
        return x

class DepthCNN(nn.Module):
    def __init__(self, out_dim):
        super(DepthCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, out_dim)
    
    def forward(self, x):
        # FIXME: x has shape [num_envs, times, 1, h, w]
        x = x.to(torch.float32)
        x = rearrange(x, "b h w c -> b c h w")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.reshape(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x
    
def calc_activation_shape_1d(dim, ksize, stride=1, dilation=1, padding=0):

    def shape_each_dim():
        odim_i = dim + 2 * padding - dilation * (ksize - 1) - 1
        return (odim_i / stride) + 1

    return int(shape_each_dim())

class ProprioCNN(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        # FCN for refining individual feature dimensions without considering temporal structure
        # in_dim = 26 = agent state info dim + actions dim = agent proprio + base pose + tcp pose + actions dim
        self.channel_transform = nn.Sequential(
            nn.Linear(26, 26),
            nn.LayerNorm(26),
            nn.ReLU(inplace=True),
            nn.Linear(26, 26),
            nn.LayerNorm(26),
            nn.ReLU(inplace=True),
        )
        # NOTE: change CNN layer sizes here? probably excessive
        # 1D CNN for capturing temporal patterns in proprioceptive data
        # add layerNorm after each conv1d
        ln_shape = calc_activation_shape_1d(in_dim, 9, 2)
        ln1 = nn.LayerNorm((26, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 7, 2)
        ln2 = nn.LayerNorm((26, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 5, 1)
        ln3 = nn.LayerNorm((26, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 3, 1)
        ln4 = nn.LayerNorm((26, ln_shape))
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(26, 26, (9, ), stride=(2, )),
            ln1,
            nn.ReLU(inplace=True),
            nn.Conv1d(26, 26, (7, ), stride=(2, )),
            ln2,
            nn.ReLU(inplace=True),
            nn.Conv1d(26, 26, (5, ), stride=(1, )),
            ln3,
            nn.ReLU(inplace=True),
            nn.Conv1d(26, 26, (3, ), stride=(1, )),
            ln4,
            nn.ReLU(inplace=True),
        )

    def forward(self, proprio):
        proprio = self.channel_transform(proprio)       # [N, 50, 26] = [batch_size, sequence_length, in_channels or features per timestep]
        proprio = rearrange(proprio, "b c l -> b l c")  # [N, 26, 50]
        proprio = self.temporal_aggregation(proprio)    # [N, 26, 2]
        proprio = proprio.flatten(1)                    # [N, 52]
        return proprio

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