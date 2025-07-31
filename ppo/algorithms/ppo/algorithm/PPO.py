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

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class Critic(nn.Module):

    def __init__(self, obs_dim, n_embd, device):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.n_embd = n_embd
        
        self.head_ = nn.ModuleList()
        for n in range(1):
            critic = nn.Sequential(nn.LayerNorm(obs_dim),
                                init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, 1)))

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


class PPO(nn.Module):

    def __init__(self, obs_dim, action_dim, n_embd, device=torch.device("cpu"), action_type='Discrete'):
        super(PPO, self).__init__()

        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.n_embd = n_embd
   
        # Actor-Critic Networks
        self.critic = Critic(obs_dim, n_embd, device)
        self.actor = Actor(obs_dim, action_dim, n_embd, device, self.action_type)

        # self.value_entropy_weight = torch.nn.Parameter(torch.ones(1)/2)
   
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.actor.zero_std(self.device)

    def forward(self, obs, action):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.actor, obs, action, batch_size, self.action_dim, self.tpdv)
        else:
            action_log, entropy = continuous_parallel_act(self.actor, obs, action, batch_size, self.action_dim, self.tpdv)

        v_loc = self.critic(obs)

        return action_log, v_loc, entropy

    def get_actions(self, obs):
        obs = check(obs).to(**self.tpdv)
        batch_size = np.shape(obs)[0]        
        
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_decentralized_act(self.actor, obs, batch_size, self.action_dim, self.tpdv)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.actor, obs, batch_size, self.action_dim, self.tpdv)

        v_loc = self.critic(obs)

        return output_action, output_action_log, v_loc

    def get_values(self, obs):
        obs = check(obs).to(**self.tpdv)
        v_tot = self.critic(obs)
        return v_tot



