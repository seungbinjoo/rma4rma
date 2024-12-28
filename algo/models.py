import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# A helper class that reshapes input tensors into a 2D format (batch size, flattened features),
# useful for passing data through linear layers
class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)

# Inherits from BaseFeaturesExtractor from Stable-Baselines3
# Custom feature extractor designed to process different types of observations,
# with various configurations based on environment specifics (env_name) and input types (e.g., depth images, proprioceptive data).
# OUTPUT: the feature extractor network outputs a vector that contains everything which needs to be inputted into the policy network
# During base policy training, feature extractor:
    # Input: priviliged state information + environment parameters
    # Output: observation x_t + environment embedding z_t + goal g
# During adaptation training / policy evaluation, feature extractor:
    # Input: obs-action history + proprioceptive history
    # Output: observation x_t + environment embedding z_t + goal g
class FeaturesExtractorRMA(BaseFeaturesExtractor):

    def __init__(self,
                 observation_space,
                 env_name,
                 object_emb_dim=32,
                 use_depth_base: bool = False,
                 use_prop_history_base: bool = False,
                 only_dr=False,
                 sys_iden=False,
                 without_adapt_module=False,
                 inc_obs_noise_in_priv=False) -> None:

        # priv_info aka env_info
        self.use_priv_info = (not only_dr) and (not without_adapt_module)\
                        and (not use_depth_base) and (not use_prop_history_base)
        self.sys_iden = sys_iden

        # If use_priv_info is True, the class includes logic for encoding privileged information about the environment to
        # help the model adapt to different contexts
        if self.use_priv_info:
            if env_name in ['PickCube', 'PickSingleYCB', 'PickSingleEGAD']:
                priv_enc_in_dim = 4 + 3 + 4
            elif env_name in ['TurnFaucet']:
                priv_enc_in_dim = 4 + 1 + 4 + 3
            elif env_name in ['PegInsertionSide']:
                priv_enc_in_dim = 4 + 3
            priv_enc_in_dim += object_emb_dim * 2
            priv_env_out_dim = priv_enc_in_dim - 4

            if inc_obs_noise_in_priv:
                priv_enc_in_dim += 19  # 9 proprio, 7 obj, 3 ext. force
                priv_env_out_dim += 15
        else:
            priv_env_out_dim = 0

        if self.sys_iden:
            priv_env_out_dim = priv_enc_in_dim

        # the output dim of feature extractor
        features_dim = priv_env_out_dim
        for k, v in observation_space.items():
            if k in ['agent_state', 'object1_state', 'goal_info']:
                features_dim += v._shape[0]
        # if env_name in ['PickCube', 'PickSingleYCB', 'PickSingleEGAD']:
        #     features_dim = (9 + 9 + 7 + 7 + 6 + priv_env_out_dim + 3 * 3)
        # elif env_name in ['TurnFaucet']:
        #     features_dim = (9 + 9 + 7 + 7 + 6 + priv_env_out_dim + 1)
        # elif env_name in ['PegInsertionSide']:
        #     features_dim = (9 + 9 + 7 + 7 + 6 + priv_env_out_dim + 3 * 3 + 5)

        self.use_prop_history_base = use_prop_history_base
        self.use_depth_base = use_depth_base
        # if use depth than it's doesn't use object state and priv info
        if use_depth_base:
            cnn_output_dim = 64
            features_dim += cnn_output_dim + 41 - 6  # cam param + img embedding

        if use_prop_history_base:
            prop_cnn_out_dim = 16
            features_dim += prop_cnn_out_dim
        super().__init__(observation_space, features_dim)

        # instantiate neural networks
        if self.use_depth_base:
            self.img_cnn = DepthCNN(out_dim=cnn_output_dim)
        if use_prop_history_base:
            self.prop_cnn = nn.Sequential(ProprioCNN(in_dim=50), Flatten(),
                                          nn.Linear(39 * 2, prop_cnn_out_dim))
        if self.use_priv_info:
            self.priv_enc = MLP(units=[128, 128, priv_env_out_dim],
                                input_size=priv_enc_in_dim)
        self.obj_id_emb = nn.Embedding(80, object_emb_dim)
        self.obj_type_emb = nn.Embedding(50, object_emb_dim)

    def forward(self,
                obs_dict,
                use_pred_e: bool = False,
                return_e_gt: bool = False,
                pred_e: th.Tensor = None) -> th.Tensor:

        priv_enc_in = []

        # If self.use_priv_info is True, the method:
        # Extracts and embeds object1_type_id and object1_id using the nn.Embedding layers (obj_type_emb, obj_emb)

        if self.use_priv_info:
            obj_type_emb = self.obj_type_emb(
                obs_dict['object1_type_id'].int()).squeeze(1)
            
            # Embedding Layers: obj_type_emb and obj_emb transform categorical data (object type ID and object ID) into dense vectors.
            obj_emb = self.obj_id_emb(obs_dict['object1_id'].int()).squeeze(1)

            # Privileged Encoding (priv_enc_in): Provides additional context that can help the model adapt to different environments or situations
            priv_enc_in.extend([obj_type_emb, obj_emb])
            priv_enc_in.append(obs_dict['obj1_priv_info'])
            priv_enc_in = th.cat(priv_enc_in, dim=1)

            if self.sys_iden:
                e_gt = priv_enc_in
            else:
                e_gt = self.priv_enc(priv_enc_in)
            if use_pred_e:
                env_vec = pred_e
            else:
                env_vec = e_gt
            obs_list = [
                obs_dict['agent_state'], obs_dict['object1_state'], env_vec,
                obs_dict['goal_info']
            ]
        # If self.use_priv_info is False, sets e_gt to None
        else:
            e_gt = None

            # Modular Processing: The method conditionally includes depth-based or proprioceptive-based features based on the configuration
            if self.use_depth_base:
                obs_list = [obs_dict['agent_state'], obs_dict['goal_info']]
                img_emb = self.img_cnn(obs_dict['image'])
                obs_list.extend([img_emb, obs_dict['camera_param']])
            else:
                obs_list = [
                    obs_dict['agent_state'], obs_dict['object1_state'],
                    obs_dict['goal_info']
                ]
            if self.use_prop_history_base:
                prop = self.prop_cnn(obs_dict['prop_act_history'])
                obs_list.append(prop)
        
        # Tries to concatenate obs_list along the last dimension to form a complete observation vector obs
        try:
            obs = th.cat(obs_list, dim=-1)
        except:
            print("error")
            breakpoint()

        if return_e_gt:
            return obs, e_gt
        else:
            return obs

# A simple multi-layer perceptron (MLP) network that can be configured with a list of layer sizes
class MLP(nn.Module):

    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.LayerNorm(output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# A network designed for adaptation purposes, which can optionally include depth perception (use_depth)
# Uses convolutional networks (DepthCNN and ProprioCNN) to process depth and proprioceptive data
class AdaptationNet(nn.Module):

    def __init__(self,
                 observation_space=None,
                 in_dim=50,
                 out_dim=16,
                 use_depth=False):

        super(AdaptationNet, self).__init__()
        self.use_depth = use_depth

        dep_cnn_output_dim = 0
        camera_param_dim = 0

        if use_depth:
            dep_cnn_output_dim = 64
            camera_param_dim = 32 + 9  # 16 + 16 + 9
        else:
            dep_cnn_output_dim = 0
            camera_param_dim = 0
        self.perc_cnn = DepthCNN(out_dim=dep_cnn_output_dim)
        self.prop_cnn = ProprioCNN(in_dim)
        self.fc = nn.Linear(39 * 2 + camera_param_dim + dep_cnn_output_dim,
                            out_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    # The forward() method extracts features from various components
    # (e.g., proprioceptive data, perception data) and concatenates them for further processing
    def forward(self, x):
        # print x
        prop, perc, cparam = x['prop'], x['perc'], x['cparam']
        # print(f"prop mean {prop.mean():.3f} min {prop.min():.3f} max {prop.max()}")
        # print(f"perc mean {perc.mean():.3f} min {perc.min():.3f} max {perc.max()}")
        # print(f"cparam mean {cparam.mean():.3f} min {cparam.min():.3f} max {cparam.max()}")
        prop = self.prop_cnn(prop)
        # print(f"new prop mean {prop.mean():.3f} min {prop.min():.3f} max {prop.max()}")
        obs = [prop]
        if self.use_depth:
            perc = self.perc_cnn(perc)
            # print(f"new perc mean {prop.mean():.3f} min {perc.min():.3f} max {perc.max()}")
            obs.extend([perc, cparam])
        x = self.fc(th.cat(obs, dim=-1))
        x = self.fc2(self.relu(x))
        # print(f"pred_e mean {x.mean():.3f} min {x.min():.3f} max {x.max():.3f}")
        # print("")
        return x

# CNN for depth images
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
        # self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(256 * 4 * 4, 512)
        # self.relu4 = nn.ReLU()
        # self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        # x has shape [n_env, times, 1, h, w]
        x = x.squeeze(2)
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
        x = x.view(-1, 128 * 4 * 4)
        # x = x.view(-1, 256 * 4 * 4)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x

# Calculate the output shape (or dimension) of a 1D convolutional layer given the input size and parameters of the convolution operation
# (such as kernel size, stride, dilation, and padding).
# This is useful for understanding how an input tensor will be transformed by the convolution
def calc_activation_shape_1d(dim, ksize, stride=1, dilation=1, padding=0):

    def shape_each_dim():
        odim_i = dim + 2 * padding - dilation * (ksize - 1) - 1
        return (odim_i / stride) + 1

    return int(shape_each_dim())

# CNN for proprioceptive data
class ProprioCNN(nn.Module):

    def __init__(self, in_dim) -> None:
        super().__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(39, 39),
            nn.LayerNorm(39),
            nn.ReLU(inplace=True),
            nn.Linear(39, 39),
            nn.LayerNorm(39),
            nn.ReLU(inplace=True),
        )
        # add layerNorm after each conv1d
        ln_shape = calc_activation_shape_1d(in_dim, 9, 2)
        # ln_shape = 21
        ln1 = nn.LayerNorm((39, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 7, 2)
        # ln_shape = 17
        ln2 = nn.LayerNorm((39, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 5, 1)
        # ln_shape = 13
        ln3 = nn.LayerNorm((39, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 3, 1)
        # ln_shape = 11
        ln4 = nn.LayerNorm((39, ln_shape))
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(39, 39, (9, ), stride=(2, )),
            ln1,
            nn.ReLU(inplace=True),
            nn.Conv1d(39, 39, (7, ), stride=(2, )),
            ln2,
            nn.ReLU(inplace=True),
            nn.Conv1d(39, 39, (5, ), stride=(1, )),
            ln3,
            nn.ReLU(inplace=True),
            nn.Conv1d(39, 39, (3, ), stride=(1, )),
            ln4,
            nn.ReLU(inplace=True),
        )

    def forward(self, prop):
        prop = self.channel_transform(prop)  # (N, 50, 39)
        prop = prop.permute((0, 2, 1))  # (N, 39, 50)
        prop = self.temporal_aggregation(prop)  # (N, 39, 3)
        prop = prop.flatten(1)
        return prop

# Calculate the output shape (or dimension) of a 2D convolutional layer given the input size and parameters of the convolution operation
# (such as kernel size, stride, dilation, and padding).
# This is useful for understanding how an input tensor will be transformed by the convolution
def calc_activation_shape_2d(dim,
                             ksize,
                             dilation=(1, 1),
                             stride=(1, 1),
                             padding=(0, 0)):

    def shape_each_dim(i):
        odim_i = dim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
        return (odim_i / stride[i]) + 1

    return int(np.floor(shape_each_dim(0))), int(np.floor(shape_each_dim(1)))
