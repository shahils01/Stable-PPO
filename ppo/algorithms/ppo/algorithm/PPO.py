import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from ppo.algorithms.utils.transformer_act import (
    continuous_autoregreesive_act,
    continuous_moe_act,
    continuous_moe_eval,
    continuous_parallel_act,
)
from ppo.algorithms.utils.util import check, init


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain("relu")
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class ObservationEncoder(nn.Module):
    """Encodes vector observations and, optionally, image observations."""

    def __init__(self, obs_dim, n_embd, use_image=False, obs_image_shape=None):
        super().__init__()
        self.obs_dim = obs_dim if obs_dim is not None else 0
        self.use_image = use_image and obs_image_shape is not None

        if self.use_image:
            c, h, w = self._infer_chw(obs_image_shape)
            self.image_shape = (c, h, w)
            self.cnn = nn.Sequential(
                init_(nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1), activate=True),
                nn.GELU(),
                init_(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), activate=True),
                nn.GELU(),
                init_(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), activate=True),
                nn.GELU(),
                nn.Flatten(),
            )
            conv_out_dim = self._get_conv_out_dim(self.image_shape)
            self.image_proj = nn.Sequential(
                nn.LayerNorm(conv_out_dim),
                init_(nn.Linear(conv_out_dim, n_embd), activate=True),
                nn.GELU(),
            )
            self.image_out_dim = n_embd
        else:
            self.cnn = None
            self.image_proj = None
            self.image_out_dim = 0

        self.output_dim = self.obs_dim + self.image_out_dim

    def _infer_chw(self, obs_shape):
        if obs_shape[0] <= 4:
            return obs_shape[0], obs_shape[1], obs_shape[2]
        return obs_shape[3], obs_shape[1], obs_shape[2]

    def _format_obs(self, obs):
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        if obs.shape[1] in (1, 3, 4):
            return obs
        return obs.permute(0, 3, 1, 2)

    def _get_conv_out_dim(self, image_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_shape)
            return self.cnn(dummy_input).view(1, -1).size(1)

    def forward(self, obs, obs_image=None):
        features = []
        if obs is not None:
            features.append(obs)

        if self.use_image and obs_image is not None:
            x = obs_image
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            x = self._format_obs(x)
            x = self.cnn(x)
            x = self.image_proj(x)
            features.append(x)

        if len(features) == 0:
            return None
        if len(features) == 1:
            return features[0]
        return torch.cat(features, dim=-1)


class QuantileCritic(nn.Module):
    def __init__(self, obs_dim, n_embd, num_quants, use_image=False, obs_image_shape=None):
        super().__init__()
        self.encoder = ObservationEncoder(obs_dim, n_embd, use_image, obs_image_shape)
        critic_input_dim = self.encoder.output_dim
        self.head = nn.Sequential(
            nn.LayerNorm(critic_input_dim),
            init_(nn.Linear(critic_input_dim, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, num_quants)),
        )

    def forward(self, obs, obs_image=None):
        features = self.encoder(obs, obs_image)
        return self.head(features)


class FlowValueCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        n_embd,
        num_quants,
        solver_steps,
        base_dist="normal",
        use_image=False,
        obs_image_shape=None,
    ):
        super().__init__()
        self.encoder = ObservationEncoder(obs_dim, n_embd, use_image, obs_image_shape)
        self.hidden_dim = self.encoder.output_dim
        self.num_quants = num_quants
        self.solver_steps = solver_steps
        self.base_dist = base_dist

        self.velocity_net = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + 2),
            init_(nn.Linear(self.hidden_dim + 2, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, 1)),
        )

        tau_grid = torch.linspace(0.0, 1.0, num_quants + 2)[1:-1]
        self.register_buffer("tau_grid", tau_grid)
        if base_dist != "normal":
            raise ValueError(f"Unsupported flow base distribution: {base_dist}")
        normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        self.register_buffer("base_grid", normal.icdf(tau_grid))

    def _get_taus_and_base(self, batch_size, device, taus=None):
        if taus is None:
            taus = self.tau_grid.unsqueeze(0).expand(batch_size, -1)
            base = self.base_grid.unsqueeze(0).expand(batch_size, -1)
        else:
            taus = taus.to(device=device, dtype=torch.float32)
            if taus.dim() == 1:
                taus = taus.unsqueeze(0).expand(batch_size, -1)
            taus = taus.clamp(1e-4, 1.0 - 1e-4)
            normal = Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
            base = normal.icdf(taus)
        return taus, base

    def _velocity(self, features, x, t):
        batch_size, num_samples = x.shape
        expanded_features = features.unsqueeze(1).expand(-1, num_samples, -1)
        net_input = torch.cat((expanded_features, x.unsqueeze(-1), t.unsqueeze(-1)), dim=-1)
        velocities = self.velocity_net(net_input.reshape(batch_size * num_samples, -1))
        return velocities.reshape(batch_size, num_samples)

    def _estimate_log_jacobian(self, transported, base):
        if transported.shape[1] <= 1:
            return torch.zeros_like(transported)

        base_delta = (base[:, 1:] - base[:, :-1]).clamp_min(1e-6)
        slope = (transported[:, 1:] - transported[:, :-1]) / base_delta
        slope = slope.clamp_min(1e-6)
        left = slope[:, :1]
        right = slope[:, -1:]
        if slope.shape[1] > 1:
            middle = 0.5 * (slope[:, :-1] + slope[:, 1:])
            jacobian = torch.cat((left, middle, right), dim=1)
        else:
            jacobian = torch.cat((left, right), dim=1)
        return torch.log(jacobian.clamp_min(1e-6))

    def forward_flow(self, obs, obs_image=None, taus=None, solver_steps=None):
        features = self.encoder(obs, obs_image)
        steps = int(solver_steps or self.solver_steps)
        batch_size = features.shape[0]
        _, base = self._get_taus_and_base(batch_size, features.device, taus)

        x = base.clone()
        path_length = torch.zeros_like(x)
        dt = 1.0 / max(steps, 1)

        for step_idx in range(steps):
            time_value = torch.full_like(x, step_idx * dt)
            velocity = self._velocity(features, x, time_value)
            x = x + dt * velocity
            path_length = path_length + dt * velocity.abs()

        log_jacobian = self._estimate_log_jacobian(x, base)
        return x, log_jacobian, path_length, base, features

    def forward(self, obs, obs_image=None, taus=None, solver_steps=None):
        quantiles, _, _, _, _ = self.forward_flow(obs, obs_image=obs_image, taus=taus, solver_steps=solver_steps)
        return quantiles

    def flow_matching_loss(self, obs, target_quantiles, obs_image=None, taus=None):
        target_quantiles = target_quantiles.detach()
        quantiles, _, _, base, features = self.forward_flow(obs, obs_image=obs_image, taus=taus)

        u = torch.rand_like(target_quantiles)
        xt = (1.0 - u) * base + u * target_quantiles
        target_velocity = target_quantiles - base
        pred_velocity = self._velocity(features, xt, u)

        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        endpoint_loss = F.mse_loss(quantiles, target_quantiles)

        if quantiles.shape[1] > 1:
            monotonicity_loss = F.relu(-(quantiles[:, 1:] - quantiles[:, :-1])).mean()
        else:
            monotonicity_loss = quantiles.new_zeros(())

        return {
            "flow_matching_loss": fm_loss,
            "flow_endpoint_loss": endpoint_loss,
            "flow_monotonicity_loss": monotonicity_loss,
            "pred_quantiles": quantiles,
        }


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, device, action_type="Discrete", use_image=False, obs_image_shape=None):
        super().__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type
        self.encoder = ObservationEncoder(obs_dim, n_embd, use_image, obs_image_shape)
        actor_input_dim = self.encoder.output_dim

        if action_type != "Discrete":
            log_std = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)

        print("action_dim = ", action_dim)
        print("obs_dim = ", obs_dim)

        self.mlp_ = nn.ModuleList()
        for _ in range(1):
            actor = nn.Sequential(
                nn.LayerNorm(actor_input_dim),
                init_(nn.Linear(actor_input_dim, n_embd), activate=True),
                nn.GELU(),
                nn.LayerNorm(n_embd),
                init_(nn.Linear(n_embd, n_embd), activate=True),
                nn.GELU(),
                nn.LayerNorm(n_embd),
                init_(nn.Linear(n_embd, action_dim)),
            )
            self.mlp_.append(actor)

    def zero_std(self, device):
        if self.action_type != "Discrete":
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, obs, obs_image=None):
        features = self.encoder(obs, obs_image)
        logit = self.mlp_[0](features)
        return logit


class GaussianExpert(nn.Module):
    def __init__(self, state_dim, action_dim, n_embd=64):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.LayerNorm(state_dim),
            init_(nn.Linear(state_dim, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, action_dim)),
        )

        log_std = torch.ones(action_dim)
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x):
        mu = self.mu_head(x)
        return mu


class MoE_GaussianPolicies(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, num_experts, use_image=False, obs_image_shape=None):
        super().__init__()

        self.num_experts = num_experts
        self.action_dim = action_dim
        self.encoder = ObservationEncoder(obs_dim, n_embd, use_image, obs_image_shape)
        expert_input_dim = self.encoder.output_dim

        self.experts = nn.ModuleList([GaussianExpert(expert_input_dim, action_dim, n_embd) for _ in range(num_experts)])

        self.gate = nn.Sequential(
            nn.Linear(expert_input_dim, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, num_experts),
        )

    def forward(self, obs, obs_image=None, temperature=2):
        features = self.encoder(obs, obs_image)
        gate_logits = self.gate(features)
        gate_weights = torch.softmax(gate_logits / temperature, dim=-1)

        mus = []
        sigmas = []
        for expert in self.experts:
            mu = expert(features)
            sigma = torch.sigmoid(expert.log_std) * 0.5
            mus.append(mu)
            sigmas.append(sigma)

        mus = torch.stack(mus, dim=1)
        sigmas = torch.stack(sigmas, dim=0)
        return mus, sigmas, gate_weights


class PPO(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        n_embd,
        moe_policy,
        device=torch.device("cpu"),
        action_type="Discrete",
        num_experts=5,
        num_quants=1,
        use_image=False,
        obs_image_shape=None,
        value_model_type="quantile",
        flow_solver_steps=8,
        flow_base_dist="normal",
        flow_target_ema=0.995,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.moe_policy = moe_policy
        self.use_image = use_image
        self.obs_image_shape = obs_image_shape
        self.value_model_type = value_model_type
        self.flow_target_ema = flow_target_ema

        if value_model_type == "flow":
            self.critic = FlowValueCritic(
                obs_dim,
                n_embd,
                num_quants=num_quants,
                solver_steps=flow_solver_steps,
                base_dist=flow_base_dist,
                use_image=use_image,
                obs_image_shape=obs_image_shape,
            )
            self.target_critic = copy.deepcopy(self.critic)
            for param in self.target_critic.parameters():
                param.requires_grad_(False)
        else:
            self.critic = QuantileCritic(obs_dim, n_embd, num_quants, use_image, obs_image_shape)
            self.target_critic = None

        if moe_policy:
            self.gmm_MoE_policy = MoE_GaussianPolicies(obs_dim, action_dim, n_embd, num_experts, use_image, obs_image_shape)
        else:
            self.actor = Actor(obs_dim, action_dim, n_embd, device, self.action_type, use_image, obs_image_shape)

        self.to(device)

    def _critic_module(self, use_target_critic=False):
        if use_target_critic and self.target_critic is not None:
            return self.target_critic
        return self.critic

    def zero_std(self):
        if self.action_type != "Discrete":
            self.actor.zero_std(self.device)

    def update_target_critic(self, ema=None):
        if self.target_critic is None:
            return

        momentum = self.flow_target_ema if ema is None else ema
        with torch.no_grad():
            for target_param, online_param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)

    def compute_flow_losses(self, obs, target_quantiles, obs_image=None, taus=None):
        if self.value_model_type != "flow":
            raise RuntimeError("Flow losses requested for a non-flow critic.")
        obs = check(obs).to(**self.tpdv)
        target_quantiles = check(target_quantiles).to(**self.tpdv)
        if obs_image is not None:
            obs_image = check(obs_image).to(**self.tpdv)
        return self.critic.flow_matching_loss(obs, target_quantiles, obs_image=obs_image, taus=taus)

    def forward(self, obs, action, gate_entropy=None, obs_image=None):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        if obs_image is not None:
            obs_image = check(obs_image).to(**self.tpdv)

        v_loc = self.critic(obs, obs_image)

        if self.moe_policy:
            mu, sigma, weight = self.gmm_MoE_policy(obs, obs_image)
            action_log, entropy, gate_entropy = continuous_moe_eval(mu, sigma, weight, action)
        else:
            if self.action_type == "Discrete":
                logits = self.actor(obs, obs_image)
                distri = Categorical(logits=logits)
                action = action.long().squeeze(-1)
                action_log = distri.log_prob(action).unsqueeze(-1)
                entropy = distri.entropy().unsqueeze(-1)
            else:
                batch_size = np.shape(obs)[0]
                action_log, entropy = continuous_parallel_act(
                    self.actor, obs, action, batch_size, self.action_dim, self.tpdv, obs_image=obs_image
                )

        return action_log, v_loc, entropy, gate_entropy

    def get_actions(self, obs, obs_image=None):
        obs = check(obs).to(**self.tpdv)
        if obs_image is not None:
            obs_image = check(obs_image).to(**self.tpdv)
        batch_size = np.shape(obs)[0]

        v_loc = self.critic(obs, obs_image)

        if self.moe_policy:
            mu, sigma, weight = self.gmm_MoE_policy(obs, obs_image)
            output_action, output_action_log = continuous_moe_act(mu, sigma, weight)
        else:
            if self.action_type == "Discrete":
                logits = self.actor(obs, obs_image)
                distri = Categorical(logits=logits)
                output_action = distri.sample().unsqueeze(-1)
                output_action_log = distri.log_prob(output_action.squeeze(-1)).unsqueeze(-1)
            else:
                output_action, output_action_log = continuous_autoregreesive_act(
                    self.actor, obs, batch_size, self.action_dim, self.tpdv, obs_image=obs_image
                )

        return output_action, output_action_log, v_loc

    def get_values(self, obs, obs_image=None, use_target_critic=False):
        obs = check(obs).to(**self.tpdv)
        if obs_image is not None:
            obs_image = check(obs_image).to(**self.tpdv)
        critic = self._critic_module(use_target_critic=use_target_critic)
        return critic(obs, obs_image)

    def get_value_distribution(self, obs, obs_image=None, taus=None, use_target_critic=False):
        obs = check(obs).to(**self.tpdv)
        if obs_image is not None:
            obs_image = check(obs_image).to(**self.tpdv)

        critic = self._critic_module(use_target_critic=use_target_critic)
        if self.value_model_type == "flow":
            return critic(obs, obs_image=obs_image, taus=taus)
        return critic(obs, obs_image)

    def get_value_flow_stats(self, obs, obs_image=None, taus=None, use_target_critic=False):
        obs = check(obs).to(**self.tpdv)
        if obs_image is not None:
            obs_image = check(obs_image).to(**self.tpdv)

        critic = self._critic_module(use_target_critic=use_target_critic)
        if self.value_model_type == "flow":
            quantiles, log_jacobian, path_length, _, _ = critic.forward_flow(obs, obs_image=obs_image, taus=taus)
            return quantiles, log_jacobian, path_length

        quantiles = critic(obs, obs_image)
        log_jacobian = torch.zeros_like(quantiles)
        path_length = torch.zeros_like(quantiles)
        return quantiles, log_jacobian, path_length
