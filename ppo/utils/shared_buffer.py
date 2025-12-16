import torch
import numpy as np
import torch.nn.functional as F
from ppo.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param obs_space: (gym.Space) observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, obs_space, act_space, env_name, use_value_entropy=True):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name
        self.env_name = env_name
        self.num_quants = args.num_quants
        
        obs_shape = get_shape_from_obs_space(obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if env_name == 'IsaacLab':
            obs_shape = (obs_shape[-1],)

        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1, *obs_shape), dtype=np.float32)
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1, self.num_quants), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1, 1), dtype=np.float32)

        act_shape = get_shape_from_act_space(act_space)
        print('act_shape after = ', act_shape)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1, act_shape), dtype=np.float32)

        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0
        self.use_value_entropy = use_value_entropy

        self.gamma_normalizer = ((1/args.gamma) ** torch.arange(args.episode_length, dtype=torch.float32)).unsqueeze(1).repeat(self.n_rollout_threads,1,1)
        self.gamma_normalizer = self.gamma_normalizer.detach().cpu().numpy()

        if self.num_quants > 1:
            self.quantile_spacing = 1.0 / (self.num_quants - 1)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, active_masks=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.obs[self.step + 1] = np.expand_dims(obs, axis=1).copy()
        self.actions[self.step] = np.expand_dims(actions, axis=1).copy()
        self.action_log_probs[self.step] = np.expand_dims(action_log_probs, axis=1).copy()
        self.value_preds[self.step] = np.expand_dims(value_preds, axis=1).copy()
        self.rewards[self.step] = np.expand_dims(rewards, axis=1).copy()
        self.masks[self.step + 1] = np.expand_dims(masks, axis=1).copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = np.expand_dims(bad_masks, axis=1).copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = np.expand_dims(active_masks, axis=1).copy()
        
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        self.value_preds[-1] = np.expand_dims(next_value, axis=1).copy()
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                if self.num_quants == 1:
                    delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = self.rewards[step] + self.wasserstein_like_distance(self.gamma * value_normalizer.denormalize(
                        self.value_preds[step + 1]) * self.masks[step + 1], value_normalizer.denormalize(self.value_preds[step]), step)
                
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                self.advantages[step] = gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            else:
                if self.num_quants == 1:
                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                                self.masks[step + 1] - self.value_preds[step]
                else:
                    delta = self.wasserstein_like_distance(self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                            self.masks[step + 1], self.value_preds[step], step)

                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                self.advantages[step] = (gae - gae.mean()) / (gae.std() + 1e-8) #gae
                self.returns[step] = gae + self.value_preds[step]

    def wasserstein_like_distance(self, icdf1, icdf2, k):
        """
        Compute the Wasserstein distance between each pair of ICDF functions.

        Parameters:
        icdf1 (torch.Tensor): Tensor of shape [2048, num_quantiles] representing the first set of ICDFs.
        icdf2 (torch.Tensor): Tensor of shape [2048, num_quantiles] representing the second set of ICDFs.

        Returns:
        torch.Tensor: Tensor of shape [2048, 1] representing the Wasserstein distance for each pair of ICDFs.
        """
        # Compute the Wasserstein distance
        # Wasserstein distance between two distributions is the area between their CDFs
        # For ICDFs, this can be approximated by the average absolute difference between the ICDF values
        # distances = torch.sum((1/64)*(icdf1 - icdf2), dim=1, keepdim=True)\             
        if self.use_value_entropy:
            del_icdf1 = (icdf1[:,:,1:] - icdf1[:,:,:-1])/self.quantile_spacing
            del_icdf2 = (icdf2[:,:,1:] - icdf2[:,:,:-1])/self.quantile_spacing

            # Guard against non-positive slopes (ICDF must be non-decreasing)
            # del_icdf1 = np.clamp(del_icdf1, min=1e-6)
            # del_icdf2 = np.clamp(del_icdf1, min=1e-6)
                        
            icdf1_mids = (icdf1[:,:,1:] + icdf1[:,:,:-1])/2
            icdf2_mids = (icdf2[:,:,1:] + icdf2[:,:,:-1])/2
            
            # q = np.exp(np.linspace(0, 1, self.num_quants))
            # q = q[1:] - q[:-1]
            # q = np.tile(q, (icdf1.shape[0], 1))
            # q = q[:, np.newaxis, :]
            
            # distances = np.sum(q*((icdf1_mids - icdf2_mids)+0.5*(self.gamma**(-k))*(np.log(del_icdf1+1e-6)-np.log(del_icdf2+1e-6))), axis=-1, keepdims=True)
            # distances = np.mean((icdf1_mids - icdf2_mids)+0.1*(self.gamma**(-k))*(np.log(del_icdf1+1e-6)-np.log(del_icdf2+1e-6)), axis=-1, keepdims=True)
            distances = np.mean((icdf1_mids - icdf2_mids)+1.0*(np.log(del_icdf1+1e-6)-np.log(del_icdf2+1e-6)), axis=-1, keepdims=True)
            # distances = np.mean((icdf1_mids - icdf2_mids), axis=-1, keepdims=True)
        
        else:
            distances = np.mean((icdf1 - icdf2), axis=-1, keepdims=True)
    
        return distances


    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, 1)

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]

        next_obs = self.obs[1:].reshape(-1, *self.obs.shape[2:])
        next_obs = next_obs[rows, cols]

        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]

        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            obs_batch = obs[indices].reshape(-1, *self.obs.shape[2:])
            next_obs_batch = next_obs[indices].reshape(-1, *self.obs.shape[2:])

            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])

            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            yield obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch,\
                   active_masks_batch, old_action_log_probs_batch, adv_targ, next_obs_batch
