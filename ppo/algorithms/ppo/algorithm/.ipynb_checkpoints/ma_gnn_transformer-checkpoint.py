import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from mat.algorithms.utils.util import check, init
from mat.algorithms.utils.transformer_act import discrete_autoregreesive_act
from mat.algorithms.utils.transformer_act import discrete_parallel_act
from mat.algorithms.utils.transformer_act import continuous_autoregreesive_act
from mat.algorithms.utils.transformer_act import continuous_parallel_act
from mat.algorithms.mat.algorithm.aero_gnn import AERO_GNN_Model as gnn

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class Encoder(nn.Module):

    def __init__(self, args, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, num_quants, device):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.ln = nn.LayerNorm(2*n_embd)
        self.head = nn.Sequential(init_(nn.Linear(2*n_embd, n_embd), activate=True, gain=1.0), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, n_embd), activate=True, gain=1.0), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, num_quants), gain=0.1))
        
        #self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         #init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)

        obs_embeddings = obs
        rep = self.ln(obs_embeddings)
                            
        v_loc, _ = torch.sort(self.head(rep), dim=-1)
                
        #print('v_loc = ', v_loc.shape)

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, args, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, device,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        self.n_agent = n_agent
        #self.edge_index = edge_index
        #self.policy_encoder = policy_encoder

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
                        
        print('n_agent = ', n_agent)
        print('action_dim = ', action_dim)
        
        self.mlp_ = nn.ModuleList()
        for n in range(n_agent):
            actor = nn.Sequential(nn.LayerNorm(2*n_embd),
                                init_(nn.Linear(2*n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, action_dim)))

            self.mlp_.append(actor)
            
        '''self.actor_embedding = nn.Sequential(init_(nn.Linear(action_dim*n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                             init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                             init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd))'''
            
        '''self.mlp_head = nn.ModuleList()
        for n in range(n_agent):
            actor_head = init_(nn.Linear(n_embd, action_dim), gain=0.001)
            self.mlp_head.append(actor_head)'''
            
        #self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
        #                                  init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        x = obs
        #x = F.gelu(x)

        logit = []
        for n in range(self.n_agent):
            logit_n = self.mlp_[n](x[:, n, :])
            logit.append(logit_n)
        logit = torch.stack(logit, dim=1)
            
        return logit


class MultiAgentGnnTransformer(nn.Module):

    def __init__(self, args, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False, num_quants=50):
        super(MultiAgentGnnTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        # state unused
        state_dim = 37
        
        # Edge Index
        #self.edge_index = self.ring_edge_index(n_agent)
        self.edge_index = np.array([[0,0,0,1,1,1,2,2,3,3,3,4,4,4,5,5],
                                    [0,1,3,0,1,2,1,2,0,3,4,3,4,5,4,5]])
        self.edge_index = torch.from_numpy(self.edge_index).long().to(device)
        
        # GNN
        self.obs_encoder = gnn(args, obs_dim, args.hid_dim, n_embd, n_agent)
        self.policy_encoder = gnn(args, action_dim*n_embd, args.hid_dim, n_embd, n_agent)
        
        self.encoder = Encoder(args, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, num_quants, device)
        self.decoder = Decoder(args, n_embd, action_dim, n_block, n_embd, n_head, n_agent, device,
                                   self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        
        #self.edge_index = np.array([[0,0,0,1,1,1,2,2,3,3,3,4,4,4,5,5],
        #                            [0,1,3,0,1,2,1,2,0,3,4,3,4,5,4,5]])
        #self.edge_index = np.array([[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9],
        #                            [1,9,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8,0]])
        #self.edge_index = np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],
        #                            [0,1,4,0,1,2,1,2,3,2,3,4,3,0,4]])
            
        self.to(device)
        
    def ring_edge_index(self, n_agents):
        edge_index = []
        for i in range(n_agents):
            # Only connect to right neighbor to avoid duplicates
            j = np.clip((i + 1),0,n_agents) % n_agents
            edge_index.append([i, i])
            edge_index.append([i, j])
            edge_index.append([j, i])
            
        edge_index = np.roll(edge_index, shift=1, axis=0)
    
        return np.array(edge_index).T

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        
        # print('v_loc shape = ', v_loc.shape)
        
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim,
                                                                             self.tpdv, deterministic)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(state, obs)
        return v_tot



