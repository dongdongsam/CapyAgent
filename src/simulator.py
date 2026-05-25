import torch
import torch.nn as nn
import json
import os

class HugeSynapse(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Extremely over-parameterized synapse
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias

class CapyElegansCore(nn.Module):
    def __init__(self, num_neurons, embedding_dim=1024):
        super().__init__()
        self.num_neurons = num_neurons
        self.embedding_dim = embedding_dim
        
        # Each neuron has a state vector (embedding)
        self.neuron_states = nn.Parameter(torch.zeros(num_neurons, embedding_dim))
        
        # LayerNorm for stability
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Connectome data
        with open("data/connectome.json", "r") as f:
            data = json.load(f)
            self.chemical_adj = torch.tensor(data["chemical"])
            self.electrical_adj = torch.tensor(data["electrical"])
            
        self.synapse_indices = torch.nonzero(self.chemical_adj + self.electrical_adj)
        self.num_synapses = self.synapse_indices.shape[0]
        
        print(f"Initializing {self.num_synapses} huge synapses...")
        
        self.synapse_weights = nn.Parameter(torch.randn(self.num_synapses, embedding_dim, embedding_dim) * 0.01)
        
    def forward(self, sensory_input, steps=5, plasticity=True):
        device = self.synapse_weights.device
        current_states = self.neuron_states.clone().to(device)
        # Add sensory input
        current_states = self.norm(current_states + sensory_input.to(device))
        
        synapse_indices = self.synapse_indices.to(device)
        
        for _ in range(steps):
            new_states = torch.zeros_like(current_states).to(device)
            
            src_indices = synapse_indices[:, 0]
            tgt_indices = synapse_indices[:, 1]
            
            # Batch matrix multiplication for all synapses
            messages = torch.bmm(current_states[src_indices].unsqueeze(1), self.synapse_weights)
            messages = messages.squeeze(1)
            
            # Accumulate messages at target neurons
            new_states.index_add_(0, tgt_indices, messages)
            
            # Hebbian-style plasticity: 
            if plasticity and self.training:
                with torch.no_grad():
                    pre_states = current_states[src_indices].unsqueeze(2)
                    post_states = new_states[tgt_indices].unsqueeze(1)
                    updates = torch.bmm(pre_states, post_states) * 1e-6
                    self.synapse_weights.data += updates
                    self.synapse_weights.data.clamp_(-1.0, 1.0)
            
            # Simple activation and state update
            current_states = torch.tanh(self.norm(new_states + current_states * 0.9))
            
        return current_states
