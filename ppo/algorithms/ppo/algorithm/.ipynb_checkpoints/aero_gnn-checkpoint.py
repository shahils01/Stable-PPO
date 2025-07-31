import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, ones
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add


class AERO_GNN_Model(MessagePassing):
    def __init__(self, args, in_channels, hid_channels, out_channels, num_agents):
        super().__init__(node_dim=0, aggr='add')

        self.args = args
        self.num_nodes = num_agents
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = self.args.num_heads
        self.hid_channels = hid_channels
        self.hid_channels_ = self.heads * self.hid_channels
        self.K = self.args.iterations
                
        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        self.dropout = nn.Dropout(self.args.dropout)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        self.dense_lins = nn.ModuleList()
        self.atts = nn.ParameterList()
        self.hop_atts = nn.ParameterList()
        self.hop_biases = nn.ParameterList()
        self.decay_weights = []
        
        # Dense layers for feature transformation
        self.dense_lins.append(Linear(self.in_channels, self.hid_channels_, bias=True, weight_initializer='glorot'))
        for _ in range(self.args.num_layers - 1): 
            self.dense_lins.append(Linear(self.hid_channels_, self.hid_channels_, bias=True, weight_initializer='glorot'))
        self.dense_lins.append(Linear(self.hid_channels_, self.out_channels, bias=True, weight_initializer='glorot'))
        
        # Edge Attention and Hop Attention Matrices 
        for k in range(self.K + 1): 
            self.atts.append(nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels)))
            self.hop_atts.append(nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels*2)))
            self.hop_biases.append(nn.Parameter(torch.Tensor(1, self.heads)))
            self.decay_weights.append(np.log((self.args.lambd_gnn / (k+1)) + (1 + 1e-6)))
        self.hop_atts[0] = nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels))
        self.atts = self.atts[1:]

    def reset_parameters(self):
        for lin in self.dense_lins: lin.reset_parameters()
        for att in self.atts: glorot(att) 
        for att in self.hop_atts: glorot(att) 
        for bias in self.hop_biases: ones(bias) 

    def hid_feat_init(self, x):
        # x shape: (batch_size, num_agents, in_channels)
        x = self.dropout(x)
        # Process all agents in all batches simultaneously
        x = x.view(-1, self.in_channels)  # (batch_size * num_agents, in_channels)
        x = self.dense_lins[0](x)
        
        for l in range(self.args.num_layers - 1):
            x = self.elu(x)
            x = self.dropout(x)
            x = self.dense_lins[l+1](x)
        
        # Reshape to (batch_size, num_agents, heads, hid_channels)
        x = x.view(-1, self.num_nodes, self.heads, self.hid_channels)
        return x

    def aero_propagate(self, h, edge_index):
        # h shape: (batch_size, num_agents, heads, hid_channels)
        batch_size = h.size(0)
        num_edges = edge_index.size(-1)
        self.k = 0
        
        # Create batch-aware edge index
        offset = torch.arange(0, batch_size * self.num_nodes, self.num_nodes, 
                            device=h.device).repeat_interleave(num_edges)
                
        # Expand edge_index for all batches
        edge_index_batch = edge_index.repeat(1, batch_size) + offset.unsqueeze(0)
        #edge_index_batch = edge_index.reshape(2,-1)
        #edge_index_batch = edge_index
        
        # Initial hop attention
        g = self.hop_att_pred(h, z_scale=None)
        z = h * g
        z_scale = z * self.decay_weights[0]

        for k in range(self.K):
            self.k = k + 1
            
            # Flatten for batch processing
            h_flat = h.reshape(-1, self.heads, self.hid_channels)  # (batch_size * num_agents, heads, hid_channels)
            z_scale_flat = z_scale.reshape(-1, self.heads, self.hid_channels)
            
            # Prepare edge features
            row, col = edge_index_batch
            z_scale_i = z_scale_flat[row]
            z_scale_j = z_scale_flat[col]
            
            # Compute attention coefficients
            a_ij = self.edge_att_pred(z_scale_i, z_scale_j, edge_index_batch)
            
            # Prepare messages
            x_j = h_flat[col]
            messages = a_ij.unsqueeze(-1) * x_j
            
            # Aggregate messages
            out = torch.zeros_like(h_flat)
            out = scatter_add(messages, row, dim=0, out=out)
            
            # Reshape back
            h = out.view(batch_size, self.num_nodes, self.heads, self.hid_channels)
            
            # Update z and z_scale
            g = self.hop_att_pred(h, z_scale)
            z += h * g
            z_scale = z * self.decay_weights[self.k]
        
        return z

    def node_classifier(self, z):
        # z shape: (batch_size, num_agents, heads, hid_channels)
        batch_size = z.size(0)
        num_agents = z.size(1)
        
        # Flatten for processing
        z = z.reshape(batch_size * num_agents, self.heads * self.hid_channels)
        z = self.elu(z)
        if self.args.add_dropout:
            z = self.dropout(z)
        z = self.dense_lins[-1](z)
        
        # Reshape back to (batch_size, num_agents, out_channels)
        z = z.view(batch_size, num_agents, self.out_channels)
        return z

    def forward(self, x, edge_index):
        # x shape: (batch_size, num_agents, in_channels)
        # edge_index shape: (2, num_edges)
        
        h0 = self.hid_feat_init(x)  # (batch_size, num_agents, heads, hid_channels)
        z_k_max = self.aero_propagate(h0, edge_index)  # (batch_size, num_agents, heads, hid_channels)
        z_star = self.node_classifier(z_k_max)  # (batch_size, num_agents, out_channels)
        
        return z_star

    def hop_att_pred(self, h, z_scale):
        # h shape: (batch_size, num_agents, heads, hid_channels) or similar
        if z_scale is None: 
            x = h
        else:
            x = torch.cat((h, z_scale), dim=-1)

        # Compute attention for all batches and agents simultaneously
        g = self.elu(x)
        g = (self.hop_atts[self.k] * g).sum(dim=-1, keepdim=True) + self.hop_biases[self.k].unsqueeze(-1)
        
        return g

    def edge_att_pred(self, z_scale_i, z_scale_j, edge_index_batch):
        # z_scale_i, z_scale_j shape: (batch_size * num_edges, heads, hid_channels)
        # edge_index_batch shape: (2, batch_size * num_edges)
        batch_size = z_scale_i.size(0)
        
        # edge attention (alpha_check_ij)
        a_ij = z_scale_i + z_scale_j
        a_ij = self.elu(a_ij)
        a_ij = (self.atts[self.k-1] * a_ij).sum(dim=-1)
        a_ij = self.softplus(a_ij) + 1e-6

        # symmetric normalization (alpha_ij)
        row, col = edge_index_batch[0], edge_index_batch[1]
        deg = scatter_add(a_ij, col, dim=0, dim_size=batch_size * self.num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        a_ij = deg_inv_sqrt[row] * a_ij * deg_inv_sqrt[col]        

        return a_ij