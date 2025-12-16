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

class Critic(nn.Module):

    def __init__(self, obs_dim, n_embd, device, num_quants):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.n_embd = n_embd
        
        self.head_ = nn.ModuleList()
        for n in range(1):
            critic = nn.Sequential(nn.LayerNorm(obs_dim),
                                init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, num_quants)))

            self.head_.append(critic)

    def forward(self, obs):
        # obs: (batch, 1, obs_dim)                
        v_loc = self.head_[0](obs)
            
        return v_loc


class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, n_embd, device, action_type='Discrete'):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
                        
        print('action_dim = ', action_dim)
        print('obs_dim = ', obs_dim)
        
        self.mlp_ = nn.ModuleList()
        for n in range(1):
            actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, action_dim)))

            self.mlp_.append(actor)

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, obs):

        logit = self.mlp_[0](obs)
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
    def __init__(self, obs_dim, action_dim, n_embd, num_experts):
        super().__init__()

        self.num_experts = num_experts
        self.action_dim = action_dim

        # Initialize Multiple Gaussian Policy Experts
        # We use nn.ModuleList to register them properly
        self.experts = nn.ModuleList([
            GaussianExpert(obs_dim, action_dim, n_embd) 
            for _ in range(num_experts)
        ])
        
        # Define the Weight Matrix (Gating Network)
        # This layer acts as the "weight matrix" that projects the state 
        # to a weight vector (logits) for combining policies.
        # Gating Network (The Router)
        self.gate = nn.Sequential(
            nn.Linear(obs_dim, n_embd), # Added a hidden layer for better routing
            nn.ReLU(),
            nn.Linear(n_embd, num_experts)
        )

    def forward(self, obs):
        """
        Returns a Mean and Std of all expert Gaussians.
        """
        gate_logits = self.gate(obs)
        gate_weights = torch.softmax(gate_logits, dim=-1) # [batch, num_experts]

        # --- B. Get Expert Parameters ---
        mus = []
        sigmas = []
        for expert in self.experts:
            mu = expert(obs)
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

    def __init__(self, obs_dim, action_dim, n_embd, moe_policy, device=torch.device("cpu"), action_type='Discrete', num_experts=5, num_quants=1):
        super(PPO, self).__init__()

        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.moe_policy = moe_policy
   
        # Actor-Critic Networks
        self.critic = Critic(obs_dim, n_embd, device, num_quants)

        if moe_policy:
            self.gmm_MoE_policy = MoE_GaussianPolicies(obs_dim, action_dim, n_embd, num_experts)
        else:
            self.actor = Actor(obs_dim, action_dim, n_embd, device, self.action_type)

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
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        v_loc = self.critic(obs)

        if self.moe_policy:
            mu, sigma, weight = self.gmm_MoE_policy(obs)
            action_log, entropy, gate_entropy = continuous_moe_eval(mu, sigma, weight, action)
        else:
            batch_size = np.shape(obs)[0]
            if self.action_type == 'Discrete':
                action = action.long()
                action_log, entropy = discrete_parallel_act(self.actor, obs, action, batch_size, self.action_dim, self.tpdv)
            else:
                action_log, entropy = continuous_parallel_act(self.actor, obs, action, batch_size, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy, gate_entropy

    def get_actions(self, obs):
        obs = check(obs).to(**self.tpdv)
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
        obs = check(obs).to(**self.tpdv)
        v_tot = self.critic(obs)
        return v_tot



