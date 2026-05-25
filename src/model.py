import torch
import torch.nn as nn
import json
from .simulator import CapyElegansCore
from .encoder import SensoryEncoder
from .decoder import CapyGenerativeDecoder

class CapyElegansLLM(nn.Module):
    def __init__(self, embedding_dim=512, device=None): # Scaled up for "Huge" feel
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with open("data/neurons.json", "r") as f:
            self.neurons = json.load(f)
            
        self.encoder = SensoryEncoder(self.neurons, embedding_dim=embedding_dim).to(self.device)
        self.core = CapyElegansCore(len(self.neurons), embedding_dim=embedding_dim).to(self.device)
        
        # New Generative Decoder
        motor_indices = [i for i, n in enumerate(self.neurons) if n['type'] == 'motor']
        num_motor = len(motor_indices)
        self.decoder = CapyGenerativeDecoder(num_motor * embedding_dim, embedding_dim=embedding_dim).to(self.device)
        
    def forward(self, text, temperature=0.7):
        # 1. Encode text to sensory stimuli
        sensory_input = self.encoder(text) 
        
        # 2. Simulate connectome dynamics
        # Train=True enables plasticity during forward pass
        self.train() 
        final_states = self.core(sensory_input, steps=10, plasticity=True)
        
        # 3. Decode motor states to GENERATED Capybara text
        self.eval()
        motor_indices = [i for i, n in enumerate(self.neurons) if n['type'] == 'motor']
        thought = self.decoder.generate(final_states[motor_indices], temperature=temperature)
        
        return thought

if __name__ == "__main__":
    # Test
    model = CapyElegansLLM(embedding_dim=64)
    print("Model initialized.")
    print("Response:", model("The world is ending tomorrow!"))
