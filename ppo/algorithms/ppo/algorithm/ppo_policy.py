import torch
import torch.nn as nn  # Already likely imported, but ensure it exists
from torch.nn.parallel import DataParallel  # Explicit import for clarity
import numpy as np
from ppo.utils.util import update_linear_schedule
from ppo.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from ppo.algorithms.utils.util import check
from ppo.algorithms.ppo.algorithm.PPO import PPO

class PPO_Policy:
    """
    PPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu"), num_quants=1):
        self.device = device
        self.algorithm_name = args.algorithm_name
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks
        self.n_embd = args.n_embd
        self.use_image = args.use_image
        self.value_model_type = args.value_model_type
        
        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'

        obs_shape, obs_image_shape = get_shape_from_obs_space(obs_space)
        self.obs_dim = obs_shape[-1] if isinstance(obs_shape, (list, tuple)) else obs_shape
        self.obs_image_dim = obs_image_shape[1:] if obs_image_shape is not None else None

        self.act_dim = get_shape_from_act_space(act_space)

        if self.action_type == 'Discrete':
            # self.act_dim = act_space.n
            self.act_num = 1
        else:
            # self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.obs_dim_ = self.obs_dim
        self.num_quants = args.flow_num_samples if self.value_model_type == "flow" else num_quants

        self.transformer = PPO(self.obs_dim, 
                               self.act_dim,
                               n_embd=args.n_embd,
                               moe_policy=args.moe_policy,
                               device=device,
                               action_type=self.action_type,
                               num_experts=args.num_experts,
                               num_quants=self.num_quants,
                               use_image=self.use_image,
                               obs_image_shape=self.obs_image_dim,
                               value_model_type=self.value_model_type,
                               flow_solver_steps=args.flow_solver_steps,
                               flow_base_dist=args.flow_base_dist,
                               flow_target_ema=args.flow_target_ema)

        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=self.lr, eps=self.opti_eps,
                                          weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, obs, masks, obs_image=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        obs = obs.reshape(-1, self.obs_dim)

        if obs_image is not None and self.obs_image_dim is not None:
            obs_image = obs_image.reshape(-1, *self.obs_image_dim)
        else:
            obs_image = None

        actions, action_log_probs, values = self.transformer.get_actions(obs, obs_image)
        actions = actions.view(-1, self.act_num)        
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, self.num_quants)
    
        return values, actions, action_log_probs

    def get_values(self, obs, masks, obs_image=None):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        obs = obs.reshape(-1, self.obs_dim)

        if obs_image is not None and self.obs_image_dim is not None:
            obs_image = obs_image.reshape(-1, *self.obs_image_dim)
        else:
            obs_image = None

        values = self.transformer.get_values(obs, obs_image)
        values = values.view(-1, self.num_quants)

        return values

    def get_value_distribution(self, obs, masks, obs_image=None, taus=None, use_target_critic=False):
        obs = obs.reshape(-1, self.obs_dim)

        if obs_image is not None and self.obs_image_dim is not None:
            obs_image = obs_image.reshape(-1, *self.obs_image_dim)
        else:
            obs_image = None

        values = self.transformer.get_value_distribution(obs, obs_image=obs_image, taus=taus, use_target_critic=use_target_critic)
        return values.view(-1, self.num_quants)

    def get_value_flow_stats(self, obs, masks, obs_image=None, taus=None, use_target_critic=False):
        obs = obs.reshape(-1, self.obs_dim)

        if obs_image is not None and self.obs_image_dim is not None:
            obs_image = obs_image.reshape(-1, *self.obs_image_dim)
        else:
            obs_image = None

        quantiles, log_jacobian, path_length = self.transformer.get_value_flow_stats(
            obs,
            obs_image=obs_image,
            taus=taus,
            use_target_critic=use_target_critic,
        )
        return (
            quantiles.view(-1, self.num_quants),
            log_jacobian.view(-1, self.num_quants),
            path_length.view(-1, self.num_quants),
        )

    def evaluate_actions(self, obs, actions, masks, active_masks=None, obs_image=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param actions: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = obs.reshape(-1, self.obs_dim)
        actions = actions.reshape(-1, self.act_num)

        if obs_image is not None and self.obs_image_dim is not None:
            obs_image = obs_image.reshape(-1, *self.obs_image_dim)
        else:
            obs_image = None

        action_log_probs, values, entropy, gate_entropy = self.transformer(obs, actions, obs_image=obs_image)

        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, self.num_quants)
        entropy = entropy.view(-1, self.act_num)

        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy*active_masks).sum()/active_masks.sum()
        else:
            entropy = entropy.mean()

        if gate_entropy is not None:
            return values, action_log_probs, entropy, gate_entropy.mean()
        else:
            return values, action_log_probs, entropy, gate_entropy

    def act(self, obs, masks, obs_image=None):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        # this function is just a wrapper for compatibility
        _, actions, _ = self.get_actions(obs, masks, obs_image)

        return actions

    def save(self, save_dir, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        transformer_state_dict = torch.load(model_dir)
        self.transformer.load_state_dict(transformer_state_dict)
        # self.transformer.reset_std()

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()

    def update_target_critic(self):
        self.transformer.update_target_critic()
