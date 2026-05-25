import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json

class SensoryEncoder(nn.Module):
    def __init__(self, neuron_names, embedding_dim=1024):
        super().__init__()
        self.neuron_names = neuron_names
        self.embedding_dim = embedding_dim
        
        # Identify sensory neurons
        self.sensory_indices = [i for i, n in enumerate(neuron_names) if n['type'] == 'sensory']
        self.num_sensory = len(self.sensory_indices)
        
        print(f"Encoder: Found {self.num_sensory} sensory neurons.")
        
        # Use a real (but small) transformer to encode the input text
        # To keep it "huge", we could use a bigger one, but for portability we'll use 'distilbert'
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Project BERT output to sensory neurons' embedding space
        # BERT output is 768. We need (Num_Sensory * Embedding_Dim)
        self.projection = nn.Linear(768, self.num_sensory * embedding_dim)
        
    def forward(self, text):
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).items()}
        outputs = self.bert(**inputs)
        
        # Use [CLS] token representation
        cls_rep = outputs.last_hidden_state[:, 0, :] # (1, 768)
        
        # Project to sensory space and scale up for impact
        sensory_vecs = self.projection(cls_rep) * 10.0 # (1, Num_Sensory * Dim)
        sensory_vecs = sensory_vecs.view(self.num_sensory, self.embedding_dim)
        
        # Create full neuron input tensor
        full_input = torch.zeros(len(self.neuron_names), self.embedding_dim).to(device)
        full_input[self.sensory_indices] = sensory_vecs
        
        return full_input

if __name__ == "__main__":
    with open("data/neurons.json", "r") as f:
        neurons = json.load(f)
    encoder = SensoryEncoder(neurons, embedding_dim=128)
    out = encoder("I am a happy capybara")
    print("Encoder output shape:", out.shape)
