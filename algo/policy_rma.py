from typing import Tuple, Optional, Type, Union, Dict

import numpy as np
import torch as th
import gymnasium.spaces as spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.distributions import Distribution

from .models import AdaptationNet

# see https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html for documentation on A2C policy
class ActorCriticPolicyRMA(ActorCriticPolicy):

    def __init__(
            # extend the ActorCriticPolicy from Stable-Baselines3 with additional parameters
            self,
            *args,
            prop_buffer_size=50,
            perc_buffer_size=20,
            n_envs=2,
            sys_iden=False,
            object_emb_dim=32,
            env_name=None,
            inc_obs_noise_in_priv=False,
            use_depth_adaptation=False,
            # use_depth_base=False,
            **kwargs):

        # common Python pattern used in class inheritance
        # super(): This function is used to call methods from a parent class (also known as a superclass).
        # In this context, it allows the __init__() method of the parent class (ActorCriticPolicy) to be called within the __init__() method of the subclass (ActorCriticPolicyRMA).
        super().__init__(*args, **kwargs)

        # store some of the input into the function --> as class variables
        self.use_depth_adaptation = use_depth_adaptation
        # self.use_depth_base = use_depth_base
        self.sys_iden = sys_iden
        self.inc_obs_noise_in_priv = inc_obs_noise_in_priv
        self.prev_actions = th.zeros(1,
                                     self.action_space.shape[0],
                                     device=self.device)
        # proprio_dim: agent propriocetion dim + action space dim
        self.n_envs = n_envs

        # BUFFER:
        # a buffer refers to a data structure (usually a tensor or an array) that stores a sequence of past data points
        # to be used later in the model’s computation
        # buffers (prop_buffer, perc_buffer, cparam_buffer) are used to store recent proprioceptive data
        # (e.g., agent’s state and actions), perceptual data (e.g., images or observations), and camera parameters
        # These buffers help the policy model maintain a short-term memory of past experiences, which can be crucial
        # for making decisions that consider recent history, allowing for better adaptation to the environment
        # WHY USE BUFFERS:
        # In environments where the state is partially observable or where adaptation is needed (e.g., changing dynamics or conditions),
        # storing past observations helps the model infer hidden variables or adapt its behavior.
        # Buffers act as a sliding window, maintaining the most recent N observations/actions to feed into an adaptation
        # module or neural network (e.g., the AdaptationNet) for context-aware predictions
        self.prop_buffer_size = prop_buffer_size
        self.proprio_dim = self.observation_space['agent_state'].shape[0] +\
                            self.action_space.shape[0]
        self.perc_buffer_size = perc_buffer_size
        self.cparam_buffer_size = perc_buffer_size
        # self.perc_sample_idx = [0, 4, 9, 19]
        self.perc_sample_idx = [19]
        # self.cparam_sample_idx = [0, 4, 9, 19]
        self.cparam_sample_idx = [19]
        self.reset_buffer()

        # privileged information encoder input dimension (?)
        priv_enc_in_dim = 4 + 3 + 4
        if env_name == 'TurnFaucet':
            priv_enc_in_dim += 1  #+1 # maybe only for TF?
        priv_enc_in_dim += object_emb_dim * 2

        # privileged information encoder output dimension (?)
        priv_env_out_dim = priv_enc_in_dim - 4
        # else:
        #     priv_env_out_dim = 3

        if inc_obs_noise_in_priv:
            priv_enc_in_dim += 19  # 9 proprio, 7 obj, 3 ext. force
            priv_env_out_dim += 15

        if sys_iden:
            priv_env_out_dim = priv_enc_in_dim
        adapt_tconv_out_dim = priv_env_out_dim

        # Instantiate adaptation neural network with appropriate dimensions and parameters
        # This NN takes in obs-action history and output of depthCNN to give predicted environment embedding
        self.adapt_tconv = AdaptationNet(self.observation_space,
                                         in_dim=50,
                                         out_dim=adapt_tconv_out_dim,
                                         use_depth=use_depth_adaptation)
        self.test_mode = False
        self.only_dr = False
        self.expert_adapt = False
        self.without_adapt_module = False

    # Prepares the policy for evaluation by setting specific flags and reducing buffer sizes
    # Configures the policy to either use domain randomization (only_dr), expert adaptation (expert_adapt),
    # or bypass the adaptation module (without_adapt_module)
    def test_eval(self,
                  expert_adapt=False,
                  only_dr=False,
                  without_adapt_module=False):
        self.test_mode = True
        self.prop_buffer = self.prop_buffer[0:1]
        self.perc_buffer = self.perc_buffer[0:1]
        self.cparam_buffer = self.cparam_buffer[0:1]
        self.cam_intrinsics = self.cam_intrinsics[0:1]
        self.only_dr = only_dr
        self.expert_adapt = expert_adapt
        self.without_adapt_module = without_adapt_module
        # if expert_adapt:
        #     self.features_extractor.use_priv_info = False
        if self.without_adapt_module:
            # self.pred_e = self.adapt_tconv(
            self.pred_e = self.adapt_tconv({
                "prop":
                self.prop_buffer[:, -50:].to("cuda"),
                "perc":
                th.zeros(1, 1, 32, 32)
            })

    # More on feature extractor in SB3: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    # This function allows the policy to incorporate environment-specific information (pred_e) into the feature extraction
    # process, making the policy adaptable to different scenarios
    def extract_features(self,
                         obs: th.Tensor,
                         adapt_trn: bool = False,
                         pred_e: th.Tensor = None,
                         return_e_gt=False):
        # The preprocessed_obs variable is created by calling preprocess_obs(), which formats the input obs tensor
        # according to the observation space and applies normalization if needed. This step ensures that the input
        # data is compatible with the feature extractor
        preprocessed_obs = preprocess_obs(
            obs,
            self.observation_space,
            normalize_images=self.normalize_images)

        # The use_pred_e flag decides whether the pred_e vector (predicted environmental embedding)
        # should be included in the feature extraction process
        use_pred_e = False

        # By default, use_pred_e is set to True if the function is called during adaptation training (adapt_trn=True)
        # or in test mode (self.test_mode=True)
        if adapt_trn or self.test_mode:
            # default case
            use_pred_e = True

        # In specific cases like self.test_mode combined with self.expert_adapt or self.only_dr (domain randomization),
        # use_pred_e is set to False to avoid using the predicted environment vector
        if (self.test_mode and self.expert_adapt) or \
            (self.test_mode and self.only_dr):
            use_pred_e = False

        # if self.expert_adapt or not self.test_mode or self.only_dr:
        #     # in expert_adapation mode, e_gt is used instead of pred_e
        #     # in DR mode, we don't have the env encoder
        #     # in training mode, we use e_gt not e_pred
        #     use_pred_e = False

        # If self.without_adapt_module is True, the function forces use_pred_e to True and uses the pred_e vector set during initialization
        # or previously computed. This bypasses using a dynamic prediction and instead uses a fixed or initial prediction
        if self.without_adapt_module:
            # in this case, use the pred_e from the first timestep
            assert use_pred_e
            pred_e = self.pred_e

        if use_pred_e:
            try:
                assert pred_e is not None
            except:
                breakpoint()
        return self.features_extractor(preprocessed_obs,
                                       use_pred_e=use_pred_e,
                                       pred_e=pred_e,
                                       return_e_gt=return_e_gt)

    # Forward pass in all the networks (actor and critic)
    # Here, we add previous action to the observation
    # Used during training
    def forward(self,
                obs: th.Tensor,
                deterministic: bool = False,
                adapt_trn=False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        '''same as the parent's except this adds previous action to the 
        observation
        '''

        # Handling Adaptation Training (adapt_trn=True)
        # The method processes the prop_buffer, perc_buffer, and cparam_buffer and moves them to the device (e.g., GPU) to ensure compatibility with the computation
        if adapt_trn:
            # Process the prop_buffer
            n_envs = obs['agent_state'].shape[0]
            self.prev_actions = self.prev_actions.to(self.device)

            # repeat prev_actions if it's first dim equals 1.
            # Ensures self.prev_actions is expanded to match the number of environments (n_envs) if needed
            if self.prev_actions.shape[0] == 1:
                self.prev_actions = self.prev_actions.repeat(n_envs, 1)
            self.prop_buffer = self.prop_buffer.to(self.device)
            self.perc_buffer = self.perc_buffer.to(self.device)
            self.cparam_buffer = self.cparam_buffer.to(self.device)
            self.cam_intrinsics = self.cam_intrinsics.to(self.device)

            # Concatenates the current agent state (obs['agent_state']) with the previous actions (self.prev_actions)
            # to create a state_action_vec. This represents the agent’s proprioceptive state and past actions.
            # Concat act (7dim) to obs (32dim),
            state_action_vec = th.cat(
                [obs['agent_state'].to(th.float32), self.prev_actions], dim=1)
            
            # Updates the prop_buffer by shifting its elements and appending the new state_action_vec
            # update the observation buffer
            self.prop_buffer = th.cat(
                [self.prop_buffer[:, 1:],
                 state_action_vec.unsqueeze(1)],
                dim=1)
            
            # if self.use_depth_adaptation:
            #     self.perc_buffer = th.cat([self.perc_buffer[:, 1:],
            #                             obs.get('image').unsqueeze(1)], dim=1)
            #     self.cparam_buffer = th.cat([self.cparam_buffer[:, 1:],
            #                             obs.get('camera_param')[:,:32].unsqueeze(1)],
            #                             dim=1)
            #     self.cam_intrinsics = obs.get('camera_param')[:,32:]
            # in adapt_trn or test_eval, we use the predicted env vector instead
            # of the gt env vector.

            # The method passes the latest prop_buffer and other optional perception (perc) and
            # camera parameters (cparam) to self.adapt_tconv, a neural network module designed to
            # generate a predicted environment embedding (pred_e)
            pred_e = self.adapt_tconv({
                "prop":
                self.prop_buffer[:, -50:].detach(),
                "perc":
                obs.get('image'),
                "cparam":
                obs.get('camera_param')
            })
            # "perc": self.perc_buffer[:, self.perc_sample_idx].detach(),
            # "cparam": th.cat([
            #         self.cparam_buffer[:, self.cparam_sample_idx
            #                         ].detach().view(n_envs, -1),
            #         self.cam_intrinsics], dim=1)})

            # Calls extract_features() with adapt_trn=True, passing the pred_e to use it during feature extraction.
            # It returns the extracted features and the ground-truth embedding (e_gt).
            features, e_gt = self.extract_features(obs,
                                                   adapt_trn=True,
                                                   pred_e=pred_e,
                                                   return_e_gt=True)
        
        # No adaptation training--> Standard Forward Pass (when adapt_trn=False)
        # Calls extract_features() with adapt_trn=False, omitting the use of pred_e for environments where adaptation is not needed
        else:
            features = self.extract_features(obs, adapt_trn=False)

        # If self.share_features_extractor is True, extracts latent representations for both the policy (latent_pi)
        # and value function (latent_vf) using a shared feature extractor (self.mlp_extractor)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)

        # Otherwise, processes the extracted features separately for the actor (policy) and critic (value function)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        # Evaluate the values for the given observations
        # Uses the policy’s value_net to compute the values (values)
        values = self.value_net(latent_vf)

        # Generates the action distribution (distribution) from the latent_pi
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Samples actions (actions) from the distribution, with an option to be deterministic
        actions = distribution.get_actions(deterministic=deterministic)

        # Computes the log probabilities (log_prob) of the sampled actions
        log_prob = distribution.log_prob(actions)

        # Updates self.prev_actions to store the current actions for future use
        actions = actions.reshape((-1, *self.action_space.shape))
        self.prev_actions = actions.detach()

        # Return pred_e and e_gt depending on whether adaptation module is being trained or not
        if adapt_trn:
            return actions, values, log_prob, pred_e, e_gt
        else:
            return actions, values, log_prob

    # Evaluates actions based on the current policy given observations
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) ->\
                            Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    # Returns the action distribution given observations
    def get_distribution(self,
                         obs: th.Tensor,
                         pred_e: th.Tensor = None) -> Distribution:
        features = self.extract_features(obs, adapt_trn=False, pred_e=pred_e)
        latent_pi = self.mlp_extractor.forward_actor(features)

        return self._get_action_dist_from_latent(latent_pi)

    # Predicts the action to take based on the given observation, with options for recurrent policies
    # Used in policy evaluation
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Same as parent's except this update the action to self.prev_action_eval.
        This is used by evaluate_policy in evaluation.py by the Eval_Callback.

        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        # todo: debug
        # self.set_training_mode(False)
        # breakpoint()

        observation, vectorized_env = self.obs_to_tensor(observation)

        # end
        with th.no_grad():
            pred_e = None
            if self.test_mode and (not self.only_dr or not self.expert_adapt):
                # update the buffer and get pred_e as in forward()
                n_envs = observation['agent_state'].shape[0]
                self.prev_actions = self.prev_actions.to(self.device)
                # repeat prev_actions if it's first dim equals 1.
                if self.prev_actions.shape[0] == 1:
                    self.prev_actions = self.prev_actions.repeat(n_envs, 1)
                self.prop_buffer = self.prop_buffer.to(self.device)
                self.perc_buffer = self.perc_buffer.to(self.device)
                self.cparam_buffer = self.cparam_buffer.to(self.device)
                self.cam_intrinsics = self.cam_intrinsics.to(self.device)
                # Concat act (7dim) to obs (32dim),
                state_action_vec = th.cat([
                    observation['agent_state'].to(th.float32),
                    self.prev_actions
                ],
                                          dim=1)
                # update the observation buffer
                self.prop_buffer = th.cat(
                    [self.prop_buffer[:, 1:],
                     state_action_vec.unsqueeze(1)],
                    dim=1)
                # if self.use_depth_adaptation:
                #     self.perc_buffer = th.cat([self.perc_buffer[:, 1:],
                #                         observation.get('image').unsqueeze(1)], dim=1)
                #     self.cparam_buffer = th.cat([self.cparam_buffer[:, 1:],
                #                         observation.get('camera_param')[:,:32].unsqueeze(1)],
                #                         dim=1)
                #     self.cam_intrinsics = observation.get('camera_param')[:,32:]
                pred_e = self.adapt_tconv({
                    "prop":
                    self.prop_buffer[:, -50:].detach(),
                    "perc":
                    observation.get('image'),
                    "cparam":
                    observation.get('camera_param')
                })
                # "perc": self.perc_buffer[:, self.perc_sample_idx].detach(),
                # "cparam": th.cat([
                #         self.cparam_buffer[:, self.cparam_sample_idx
                #                         ].detach().view(n_envs, -1),
                #         self.cam_intrinsics], dim=1)})
            actions = self._predict(observation,
                                    deterministic=deterministic,
                                    pred_e=pred_e)
            self.prev_actions = actions.detach()

        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low,
                                  self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    # helper method to predict actions using the distribution generated from the observation and optionally pred_e
    def _predict(self,
                 observation: th.Tensor,
                 deterministic: bool = False,
                 pred_e: th.Tensor = None) -> th.Tensor:
        """modified to pass pred_e
        """
        return self.get_distribution(
            observation,
            pred_e=pred_e).get_actions(deterministic=deterministic)

    # Predicts the values for the given observations
    def predict_values(self,
                       obs: th.Tensor,
                       done_idx: int = None) -> th.Tensor:

        # Preprocess the observation if needed
        features = self.extract_features(obs, adapt_trn=False)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    # Resets the self.prev_actions tensor when episodes end or at specific indexes
    def reset_prev_action(self, done_idx: int = None):
        if done_idx is not None:
            self.prev_actions[done_idx] = th.zeros(self.action_space.shape[0],
                                                   device=self.device)
        else:
            self.prev_actions = th.zeros_like(self.prev_actions)

    # def reset_eval_prev_action(self):
    #     self.prev_actions_eval = th.zeros(1, self.action_space.shape[0],
    #                         device=self.device)

    # Resets the internal buffers that store past state and observation information for adaptation
    def reset_buffer(self, dones: int = None, n=None):
        if n is not None:
            n_envs = n
        else:
            n_envs = self.n_envs
        if dones is not None:
            self.prop_buffer[dones == 1] = 0
            self.perc_buffer[dones == 1] = 0
            self.cparam_buffer[dones == 1] = 0
        else:
            self.prop_buffer = th.zeros(n_envs,
                                        self.prop_buffer_size,
                                        self.proprio_dim,
                                        device=self.device)
            self.perc_buffer = th.zeros(n_envs,
                                        self.perc_buffer_size,
                                        1,
                                        32,
                                        32,
                                        device=self.device)
            self.cparam_buffer = th.zeros(n_envs,
                                          self.cparam_buffer_size,
                                          32,
                                          device=self.device)
            self.cam_intrinsics = th.zeros(n_envs, 9, device=self.device)
