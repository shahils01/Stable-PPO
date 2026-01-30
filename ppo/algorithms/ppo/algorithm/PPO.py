import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from ppo.algorithms.utils.util import check, init
from ppo.algorithms.utils.transformer_act import discrete_autoregreesive_act, discrete_decentralized_act
from ppo.algorithms.utils.transformer_act import discrete_parallel_act
from ppo.algorithms.utils.transformer_act import continuous_autoregreesive_act
from ppo.algorithms.utils.transformer_act import continuous_parallel_act
from ppo.algorithms.utils.transformer_act import continuous_moe_act, continuous_moe_eval


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

def _to_tpdv(x, tpdv):
    if isinstance(x, dict):
        return {k: v.to(**tpdv) for k, v in x.items()}
    return x.to(**tpdv)

class ObservationEncoder(nn.Module):
    """Small helper that turns either vector or image observations into feature vectors."""

    def __init__(self, obs_shape, n_embd):
        super().__init__()
        self.is_image = len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.n_embd = n_embd

        if self.is_image:
            c, h, w = self._infer_chw(obs_shape)
            self.image_shape = (c, h, w)
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            conv_out_dim = self._get_conv_out_dim(self.image_shape)
            self.proj = nn.Sequential(
                nn.LayerNorm(conv_out_dim),
                init_(nn.Linear(conv_out_dim, n_embd), activate=True),
                nn.GELU()
            )
            self.output_dim = n_embd
        else:
            self.cnn = None
            self.proj = None
            self.output_dim = obs_shape[0]

    def _infer_chw(self, obs_shape):
        # Assume channel-first if first dimension looks like a channel count.
        if obs_shape[0] <= 4:
            return obs_shape[0], obs_shape[1], obs_shape[2]
        return obs_shape[2], obs_shape[0], obs_shape[1]

    def _format_obs(self, obs):
        # Accept HWC or CHW; always return NCHW.
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        if obs.shape[1] in (1, 3, 4):
            return obs
        return obs.permute(0, 3, 1, 2)

    def _get_conv_out_dim(self, image_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_shape)
            return self.cnn(dummy_input).view(1, -1).size(1)

    def forward(self, obs):
        if not self.is_image:
            return obs

        x = obs
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = self._format_obs(x)
        x = self.cnn(x)
        x = self.proj(x)
        return x


class DictObservationEncoder(nn.Module):
    """Encode dict observations with a vector policy obs and a terrain map."""

    def __init__(self, obs_shape, n_embd):
        super().__init__()
        self.obs_shape = obs_shape
        self.n_embd = n_embd
        self.policy_shape = obs_shape["policy"]
        self.map_shape = obs_shape["tmap"]
        self.policy_dim = self.policy_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_dim = self._get_conv_out_dim()
        self.proj = nn.Sequential(
            nn.LayerNorm(conv_out_dim),
            init_(nn.Linear(conv_out_dim, n_embd), activate=True),
            nn.GELU()
        )
        self.output_dim = self.policy_dim + n_embd

    def _get_conv_out_dim(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.map_shape)
            return self.cnn(dummy_input).view(1, -1).size(1)

    def _format_map(self, tmap):
        if tmap.dim() == 2:
            tmap = tmap.unsqueeze(0)
        if tmap.dim() == 3:
            if tmap.shape[1] in (1, 3, 4):
                return tmap
            return tmap.unsqueeze(1)
        if tmap.dim() == 4:
            if tmap.shape[1] in (1, 3, 4):
                return tmap
            if tmap.shape[-1] in (1, 3, 4):
                return tmap.permute(0, 3, 1, 2)
        return tmap

    def forward(self, obs):
        policy = obs["policy"].float()
        tmap = obs["tmap"].float()
        if policy.dim() == 1:
            policy = policy.unsqueeze(0)
        tmap = self._format_map(tmap)
        tmap_feat = self.proj(self.cnn(tmap))
        return torch.cat([policy, tmap_feat], dim=-1)


class Critic(nn.Module):

    def __init__(self, obs_shape, n_embd, device, num_quants, terrain_map_shape=None):
        super(Critic, self).__init__()

        self.obs_shape = obs_shape
        self.n_embd = n_embd

        if isinstance(obs_shape, dict):
            self.encoder = DictObservationEncoder(obs_shape, n_embd)
        else:
            self.encoder = ObservationEncoder(obs_shape, n_embd)
        critic_input_dim = self.encoder.output_dim

        self.head_ = nn.ModuleList()
        for n in range(1):
            critic = nn.Sequential(nn.LayerNorm(critic_input_dim),
                                init_(nn.Linear(critic_input_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, num_quants)))

            self.head_.append(critic)

    def forward(self, obs):
        # obs: (batch, 1, obs_dim)                
        features = self.encoder(obs)
        v_loc = self.head_[0](features)
            
        return v_loc


class Actor(nn.Module):

    def __init__(self, obs_shape, action_dim, n_embd, device, action_type='Discrete', terrain_map_shape=None):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type
        if isinstance(obs_shape, dict):
            self.encoder = DictObservationEncoder(obs_shape, n_embd)
        else:
            self.encoder = ObservationEncoder(obs_shape, n_embd)
        actor_input_dim = self.encoder.output_dim

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
                        
        print('action_dim = ', action_dim)
        print('obs_shape = ', obs_shape)
        
        self.mlp_ = nn.ModuleList()
        for n in range(1):
            actor = nn.Sequential(nn.LayerNorm(actor_input_dim),
                                init_(nn.Linear(actor_input_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, action_dim)))

            self.mlp_.append(actor)

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, obs):
        features = self.encoder(obs)
        logit = self.mlp_[0](features)
        return logit


class GaussianExpert(nn.Module):
    def __init__(self, state_dim, action_dim, n_embd=64):
        super().__init__()
        self.mu_head = nn.Sequential(nn.LayerNorm(state_dim),
                                init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, action_dim)))
        # Separate heads for Mean and Log Std
        # self.mu_head = nn.Linear(n_embd, action_dim)
        # self.log_std_head = nn.Linear(n_embd, action_dim)

        log_std = torch.ones(action_dim)
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x):
        # features = self.net(x)
        mu = self.mu_head(x)
        
        # Clamp log_std to maintain numerical stability in PPO
        # log_std = self.log_std_head(features)
        # log_std = torch.clamp(log_std, min=-6, max=6) 
        
        return mu


class MoE_GaussianPolicies(nn.Module):
    def __init__(self, obs_shape, action_dim, n_embd, num_experts, terrain_map_shape=None):
        super().__init__()

        self.num_experts = num_experts
        self.action_dim = action_dim
        if isinstance(obs_shape, dict):
            self.encoder = DictObservationEncoder(obs_shape, n_embd)
        else:
            self.encoder = ObservationEncoder(obs_shape, n_embd)
        expert_input_dim = self.encoder.output_dim

        # Initialize Multiple Gaussian Policy Experts
        # We use nn.ModuleList to register them properly
        self.experts = nn.ModuleList([
            GaussianExpert(expert_input_dim, action_dim, n_embd) 
            for _ in range(num_experts)
        ])
        
        # Define the Weight Matrix (Gating Network)
        # This layer acts as the "weight matrix" that projects the state 
        # to a weight vector (logits) for combining policies.
        # Gating Network (The Router)
        self.gate = nn.Sequential(
            nn.Linear(expert_input_dim, n_embd), # Added a hidden layer for better routing
            nn.ReLU(),
            nn.Linear(n_embd, num_experts)
        )

    def forward(self, obs, temperature=2):
        """
        Returns a Mean and Std of all expert Gaussians.
        """
        features = self.encoder(obs)
        gate_logits = self.gate(features)
        gate_weights = torch.softmax(gate_logits/temperature, dim=-1) # [batch, num_experts]

        # --- B. Get Expert Parameters ---
        mus = []
        sigmas = []
        for expert in self.experts:
            mu = expert(features)
            sigma = torch.sigmoid(expert.log_std) * 0.5
            mus.append(mu)
            sigmas.append(sigma)

        # Stack to shape: [batch, num_experts, action_dim]
        mus = torch.stack(mus, dim=1)
        sigmas = torch.stack(sigmas, dim=0)

        # print('mus shape = ', mus.shape)
        # print('sigmas shape = ', sigmas.shape)

        return mus, sigmas, gate_weights


class PPO(nn.Module):

    def __init__(self, obs_shape, action_dim, n_embd, moe_policy, device=torch.device("cpu"), action_type='Discrete', num_experts=5, num_quants=1, terrain_map_shape=None):
        super(PPO, self).__init__()

        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.moe_policy = moe_policy
        self.obs_shape = obs_shape
   
        # Actor-Critic Networks
        self.critic = Critic(obs_shape, n_embd, device, num_quants, terrain_map_shape=terrain_map_shape)

        if moe_policy:
            self.gmm_MoE_policy = MoE_GaussianPolicies(obs_shape, action_dim, n_embd, num_experts, terrain_map_shape=terrain_map_shape)
        else:
            self.actor = Actor(obs_shape, action_dim, n_embd, device, self.action_type, terrain_map_shape=terrain_map_shape)

        # self.value_entropy_weight = torch.nn.Parameter(torch.ones(1)/2)
   
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.actor.zero_std(self.device)

    def forward(self, obs, action, gate_entropy=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)
        obs = _to_tpdv(check(obs), self.tpdv)
        action = check(action).to(**self.tpdv)

        v_loc = self.critic(obs)

        if self.moe_policy:
            mu, sigma, weight = self.gmm_MoE_policy(obs)
            action_log, entropy, gate_entropy = continuous_moe_eval(mu, sigma, weight, action)
        else:
            if isinstance(obs, dict):
                batch_size = obs["policy"].shape[0]
            else:
                batch_size = np.shape(obs)[0]
            if self.action_type == 'Discrete':
                action = action.long()
                action_log, entropy = discrete_parallel_act(self.actor, obs, action, batch_size, self.action_dim, self.tpdv)
            else:
                action_log, entropy = continuous_parallel_act(self.actor, obs, action, batch_size, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy, gate_entropy

    def get_actions(self, obs):
        obs = _to_tpdv(check(obs), self.tpdv)
        if isinstance(obs, dict):
            batch_size = obs["policy"].shape[0]
        else:
            batch_size = np.shape(obs)[0]

        v_loc = self.critic(obs)
        
        if self.moe_policy:
            mu, sigma, weight = self.gmm_MoE_policy(obs)
            output_action, output_action_log = continuous_moe_act(mu, sigma, weight)
        else:
            if self.action_type == "Discrete":
                output_action, output_action_log = discrete_decentralized_act(self.actor, obs, batch_size, self.action_dim, self.tpdv)
            else:
                output_action, output_action_log = continuous_autoregreesive_act(self.actor, obs, batch_size, self.action_dim, self.tpdv)

        return output_action, output_action_log, v_loc

    def get_values(self, obs):
        obs = _to_tpdv(check(obs), self.tpdv)
        v_tot = self.critic(obs)
        return v_tot
