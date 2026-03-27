import numpy as np
import torch

from ppo.utils.util import get_shape_from_act_space, get_shape_from_obs_space


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
        self.value_model_type = args.value_model_type
        self.num_quants = args.flow_num_samples if self.value_model_type == "flow" else args.num_quants
        self.dgae_epsilon = args.dgae_epsilon
        self.use_value_entropy = args.use_value_entropy
        self.use_image = args.use_image
        self.adv_expansion_coef = args.adv_expansion_coef
        self.adv_magnitude_coef = args.adv_magnitude_coef
        self.adv_use_path_length = args.adv_use_path_length
        self.flow_stats = {}

        obs_shape, obs_img_shape = get_shape_from_obs_space(obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if env_name == "IsaacLab":
            obs_shape = (obs_shape[-1],)
            if obs_img_shape is not None:
                obs_img_shape = obs_img_shape[1:]

        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1, *obs_shape), dtype=np.float32)

        if obs_img_shape is not None and self.use_image:
            self.obs_img = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1, *obs_img_shape), dtype=np.float32)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1, self.num_quants), dtype=np.float32
        )
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros((self.episode_length, self.n_rollout_threads, 1, 1), dtype=np.float32)

        act_shape = get_shape_from_act_space(act_space)
        print("act_shape after = ", act_shape)

        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, 1, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, 1, act_shape), dtype=np.float32)

        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

        self.q = np.exp(np.linspace(0, 1, self.num_quants))
        self.q = self.q[1:] - self.q[:-1]
        self.q = np.tile(self.q, (self.n_rollout_threads, 1))
        self.q = self.q[:, np.newaxis, :]

        self.gamma_normalizer = (
            ((1 / args.gamma) ** torch.arange(args.episode_length, dtype=torch.float32))
            .unsqueeze(1)
            .repeat(self.n_rollout_threads, 1, 1)
        )
        self.gamma_normalizer = self.gamma_normalizer.detach().cpu().numpy()

        if self.num_quants > 1:
            self.quantile_spacing = 1.0 / (self.num_quants - 1)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, active_masks=None, obs_img=None):
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
        if obs_img is not None and self.use_image:
            self.obs_img[self.step + 1] = np.expand_dims(obs_img, axis=1).copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.flow_stats = {}

    def chooseafter_update(self):
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        self.value_preds[-1] = np.expand_dims(next_value, axis=1).copy()
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                if self.num_quants == 1:
                    delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                        self.value_preds[step + 1]
                    ) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = self.rewards[step] + self.wasserstein_like_distance(
                        self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1],
                        value_normalizer.denormalize(self.value_preds[step]),
                        step,
                    )

                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                self.advantages[step] = gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            else:
                if self.num_quants == 1:
                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                else:
                    delta = self.wasserstein_like_distance(
                        self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1],
                        self.value_preds[step],
                        step,
                    )

                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                self.advantages[step] = (gae - gae.mean()) / (gae.std() + 1e-8)
                self.returns[step] = gae + self.value_preds[step]

    def compute_returns_flow(self, policy, target_policy, value_normalizer=None):
        with torch.no_grad():
            obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
            next_obs = self.obs[1:].reshape(-1, *self.obs.shape[3:])
            masks = self.masks[:-1].reshape(-1, *self.masks.shape[3:])
            next_masks = self.masks[1:].reshape(-1, *self.masks.shape[3:])
            rewards = self.rewards.reshape(-1, 1)

            if self.use_image:
                obs_img = self.obs_img[:-1].reshape(-1, *self.obs_img.shape[3:])
                next_obs_img = self.obs_img[1:].reshape(-1, *self.obs_img.shape[3:])
            else:
                obs_img = None
                next_obs_img = None

            current_q, current_log_j, current_path = policy.get_value_flow_stats(
                obs, masks, obs_image=obs_img, use_target_critic=False
            )
            target_next_q, target_next_log_j, target_next_path = target_policy.get_value_flow_stats(
                next_obs, next_masks, obs_image=next_obs_img, use_target_critic=True
            )
            bootstrap_q, _, _ = target_policy.get_value_flow_stats(
                self.obs[-1].reshape(-1, *self.obs.shape[3:]),
                self.masks[-1].reshape(-1, *self.masks.shape[3:]),
                obs_image=self.obs_img[-1].reshape(-1, *self.obs_img.shape[3:]) if self.use_image else None,
                use_target_critic=True,
            )

            device = current_q.device if torch.is_tensor(current_q) else torch.device("cpu")

            if value_normalizer is not None:
                current_q = value_normalizer.denormalize(current_q)
                target_next_q = value_normalizer.denormalize(target_next_q)
                bootstrap_q = value_normalizer.denormalize(bootstrap_q)

            if not torch.is_tensor(current_q):
                current_q = torch.as_tensor(current_q, dtype=torch.float32, device=device)
            else:
                current_q = current_q.to(device=device, dtype=torch.float32)
            if not torch.is_tensor(target_next_q):
                target_next_q = torch.as_tensor(target_next_q, dtype=torch.float32, device=current_q.device)
            else:
                target_next_q = target_next_q.to(device=current_q.device, dtype=torch.float32)
            if not torch.is_tensor(bootstrap_q):
                bootstrap_q = torch.as_tensor(bootstrap_q, dtype=torch.float32, device=current_q.device)
            else:
                bootstrap_q = bootstrap_q.to(device=current_q.device, dtype=torch.float32)
            if torch.is_tensor(current_log_j):
                current_log_j = current_log_j.to(device=current_q.device, dtype=torch.float32)
            if torch.is_tensor(target_log_j):
                target_log_j = target_log_j.to(device=current_q.device, dtype=torch.float32)
            if torch.is_tensor(current_path):
                current_path = current_path.to(device=current_q.device, dtype=torch.float32)
            if torch.is_tensor(target_path):
                target_path = target_path.to(device=current_q.device, dtype=torch.float32)

            tensor_dtype = torch.float32
            rewards_t = torch.as_tensor(rewards, dtype=tensor_dtype, device=current_q.device)
            next_masks_t = torch.as_tensor(next_masks, dtype=tensor_dtype, device=current_q.device)

            target_q = rewards_t + self.gamma * next_masks_t * target_next_q
            target_log_j = target_next_log_j
            target_path = target_next_path

            direction = (target_q - current_q).mean(dim=-1, keepdim=True)
            expansion = (target_log_j - current_log_j).mean(dim=-1, keepdim=True)
            magnitude = (target_q - current_q).abs().mean(dim=-1, keepdim=True)
            path_term = (target_path - current_path).mean(dim=-1, keepdim=True)

            mag_norm = magnitude / magnitude.mean().clamp_min(1e-6)
            signed_delta = direction + self.adv_expansion_coef * expansion
            if self.adv_use_path_length:
                signed_delta = signed_delta + self.adv_expansion_coef * path_term
            delta = signed_delta * (1.0 + self.adv_magnitude_coef * mag_norm)

            direction = direction.view(self.episode_length, self.n_rollout_threads, 1, 1).cpu().numpy()
            expansion = expansion.view(self.episode_length, self.n_rollout_threads, 1, 1).cpu().numpy()
            magnitude = magnitude.view(self.episode_length, self.n_rollout_threads, 1, 1).cpu().numpy()
            path_term = path_term.view(self.episode_length, self.n_rollout_threads, 1, 1).cpu().numpy()
            delta = delta.view(self.episode_length, self.n_rollout_threads, 1, 1).cpu().numpy()

            current_q_np = current_q.view(self.episode_length, self.n_rollout_threads, 1, self.num_quants).cpu().numpy()
            target_q_np = target_q.view(self.episode_length, self.n_rollout_threads, 1, self.num_quants).cpu().numpy()
            bootstrap_q_np = bootstrap_q.view(self.n_rollout_threads, 1, self.num_quants).cpu().numpy()

        self.value_preds[:-1] = current_q_np
        self.value_preds[-1] = bootstrap_q_np
        self.returns[:-1] = target_q_np

        gae = np.zeros((self.n_rollout_threads, 1, 1), dtype=np.float32)
        for step in reversed(range(self.episode_length)):
            gae = delta[step] + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.advantages[step] = gae

        self.flow_stats = {
            "adv_direction": float(direction.mean()),
            "adv_expansion": float(expansion.mean()),
            "adv_magnitude": float(magnitude.mean()),
            "adv_path": float(path_term.mean()),
            "jacobian_mean": float(current_log_j.mean().item()),
            "path_length_mean": float(current_path.mean().item()),
        }

    def wasserstein_like_distance(self, icdf1, icdf2, step):
        if self.use_value_entropy:
            del_icdf1 = (icdf1[:, :, 1:] - icdf1[:, :, :-1]) / self.quantile_spacing
            del_icdf2 = (icdf2[:, :, 1:] - icdf2[:, :, :-1]) / self.quantile_spacing

            icdf1_mids = (icdf1[:, :, 1:] + icdf1[:, :, :-1]) / 2
            icdf2_mids = (icdf2[:, :, 1:] + icdf2[:, :, :-1]) / 2

            distances = np.sum(
                self.q
                * (
                    (icdf1_mids - icdf2_mids)
                    + (self.dgae_epsilon / self.gamma**step) * (np.log(del_icdf1 + 1e-6) - np.log(del_icdf2 + 1e-6))
                ),
                axis=-1,
                keepdims=True,
            )
        else:
            distances = np.mean((icdf1 - icdf2), axis=-1, keepdims=True)

        return distances

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({}).".format(
                    n_rollout_threads,
                    episode_length,
                    n_rollout_threads * episode_length,
                    num_mini_batch,
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, 1)

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]

        next_obs = self.obs[1:].reshape(-1, *self.obs.shape[2:])
        next_obs = next_obs[rows, cols]

        if self.use_image:
            obs_img = self.obs_img[:-1].reshape(-1, *self.obs_img.shape[2:])
            obs_img = obs_img[rows, cols]

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
            obs_batch = obs[indices].reshape(-1, *self.obs.shape[2:])
            next_obs_batch = next_obs[indices].reshape(-1, *self.obs.shape[2:])

            if self.use_image:
                obs_img_batch = obs_img[indices].reshape(-1, *self.obs_img.shape[2:])

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

            if self.use_image:
                yield (
                    obs_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    active_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    next_obs_batch,
                    obs_img_batch,
                )
            else:
                yield (
                    obs_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    active_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    next_obs_batch,
                )
