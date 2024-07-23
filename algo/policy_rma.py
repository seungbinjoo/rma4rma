from typing import Tuple, Optional, Type, Union, Dict

import numpy as np
import torch as th
import gymnasium.spaces as spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.distributions import Distribution

from .models import AdaptationNet


class ActorCriticPolicyRMA(ActorCriticPolicy):

    def __init__(
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

        super().__init__(*args, **kwargs)
        self.use_depth_adaptation = use_depth_adaptation
        # self.use_depth_base = use_depth_base
        self.sys_iden = sys_iden
        self.inc_obs_noise_in_priv = inc_obs_noise_in_priv
        self.prev_actions = th.zeros(1,
                                     self.action_space.shape[0],
                                     device=self.device)
        # proprio_dim: agent propriocetion dim + action space dim
        self.n_envs = n_envs
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
        priv_enc_in_dim = 4 + 3 + 4
        if env_name == 'TurnFaucet':
            priv_enc_in_dim += 1  #+1 # maybe only for TF?
        priv_enc_in_dim += object_emb_dim * 2
        priv_env_out_dim = priv_enc_in_dim - 4
        # else:
        #     priv_env_out_dim = 3

        if inc_obs_noise_in_priv:
            priv_enc_in_dim += 19  # 9 proprio, 7 obj, 3 ext. force
            priv_env_out_dim += 15

        if sys_iden:
            priv_env_out_dim = priv_enc_in_dim
        adapt_tconv_out_dim = priv_env_out_dim

        self.adapt_tconv = AdaptationNet(self.observation_space,
                                         in_dim=50,
                                         out_dim=adapt_tconv_out_dim,
                                         use_depth=use_depth_adaptation)
        self.test_mode = False
        self.only_dr = False
        self.expert_adapt = False
        self.without_adapt_module = False

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

    def extract_features(self,
                         obs: th.Tensor,
                         adapt_trn: bool = False,
                         pred_e: th.Tensor = None,
                         return_e_gt=False):
        preprocessed_obs = preprocess_obs(
            obs,
            self.observation_space,
            normalize_images=self.normalize_images)

        use_pred_e = False
        if adapt_trn or self.test_mode:
            # default case
            use_pred_e = True
        if (self.test_mode and self.expert_adapt) or \
            (self.test_mode and self.only_dr):
            use_pred_e = False
        # if self.expert_adapt or not self.test_mode or self.only_dr:
        #     # in expert_adapation mode, e_gt is used instead of pred_e
        #     # in DR mode, we don't have the env encoder
        #     # in training mode, we use e_gt not e_pred
        #     use_pred_e = False

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

    def forward(self,
                obs: th.Tensor,
                deterministic: bool = False,
                adapt_trn=False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        '''same as the parent's except this adds previous action to the 
        observation
        '''
        if adapt_trn:
            # Process the prop_buffer
            n_envs = obs['agent_state'].shape[0]
            self.prev_actions = self.prev_actions.to(self.device)
            # repeat prev_actions if it's first dim equals 1.
            if self.prev_actions.shape[0] == 1:
                self.prev_actions = self.prev_actions.repeat(n_envs, 1)
            self.prop_buffer = self.prop_buffer.to(self.device)
            self.perc_buffer = self.perc_buffer.to(self.device)
            self.cparam_buffer = self.cparam_buffer.to(self.device)
            self.cam_intrinsics = self.cam_intrinsics.to(self.device)
            # Concat act (7dim) to obs (32dim),
            state_action_vec = th.cat(
                [obs['agent_state'].to(th.float32), self.prev_actions], dim=1)
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

            features, e_gt = self.extract_features(obs,
                                                   adapt_trn=True,
                                                   pred_e=pred_e,
                                                   return_e_gt=True)
        else:
            features = self.extract_features(obs, adapt_trn=False)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        self.prev_actions = actions.detach()

        if adapt_trn:
            return actions, values, log_prob, pred_e, e_gt
        else:
            return actions, values, log_prob

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

    def get_distribution(self,
                         obs: th.Tensor,
                         pred_e: th.Tensor = None) -> Distribution:
        features = self.extract_features(obs, adapt_trn=False, pred_e=pred_e)
        latent_pi = self.mlp_extractor.forward_actor(features)

        return self._get_action_dist_from_latent(latent_pi)

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

    def _predict(self,
                 observation: th.Tensor,
                 deterministic: bool = False,
                 pred_e: th.Tensor = None) -> th.Tensor:
        """modified to pass pred_e
        """
        return self.get_distribution(
            observation,
            pred_e=pred_e).get_actions(deterministic=deterministic)

    def predict_values(self,
                       obs: th.Tensor,
                       done_idx: int = None) -> th.Tensor:

        # Preprocess the observation if needed
        features = self.extract_features(obs, adapt_trn=False)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def reset_prev_action(self, done_idx: int = None):
        if done_idx is not None:
            self.prev_actions[done_idx] = th.zeros(self.action_space.shape[0],
                                                   device=self.device)
        else:
            self.prev_actions = th.zeros_like(self.prev_actions)

    # def reset_eval_prev_action(self):
    #     self.prev_actions_eval = th.zeros(1, self.action_space.shape[0],
    #                         device=self.device)

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
