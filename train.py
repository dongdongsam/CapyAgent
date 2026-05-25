import os
# Force PyTorch to recognize RTX 50 series by falling back to sm_90 compatibility
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import torch
import torch.nn as nn
import torch.optim as optim
from src.model import CapyElegansLLM
import random
import os

# 1. Expanded Zen Corpus (Data Augmentation)
BASE_CORPUS = [
    "h m m m . . .",
    "still water . . .",
    "p e a c e f u l . . .",
    "just being . . .",
    "the void is warm . . .",
    "m u n c h . . .",
    "floating . . .",
    "no thoughts . . .",
    "pure presence . . .",
    "the sun is enough . . .",
    "watching the ripples . . .",
    "s t i l l n e s s . . .",
    "orange on my head . . .",
    "warm bath . . .",
    "square and chill . . .",
    "slow river . . .",
    "whispers of grass . . .",
    "sunlight on fur . . .",
    "deep breath . . .",
    "nothingness . . ."
]

# Generate variations to increase data volume
CAPY_CORPUS = []
for base in BASE_CORPUS:
    CAPY_CORPUS.append(base)
    CAPY_CORPUS.append(base.replace(" . . .", " ."))
    CAPY_CORPUS.append(base.upper())
    CAPY_CORPUS.append(" . . . " + base)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 GPU OVERDRIVE: Aligning CapyElegans on {device}...")
    
    embedding_dim = 286 # Keep the 5x expansion
    model = CapyElegansLLM(embedding_dim=embedding_dim, device=device)
    model.to(device)
    
    # 2. FREEZE THE CORE (Biological Instincts are immutable)
    # Only train the Encoder and Decoder (The 'Voice')
    for param in model.core.parameters():
        param.requires_grad = False
    
    print(f"Brain Frozen. Training the Voice (Decoder) only.")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 40 
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        random.shuffle(CAPY_CORPUS)
        
        for i, text in enumerate(CAPY_CORPUS):
            optimizer.zero_grad()
            
            # Use random prompts to make the model robust
            prompt = random.choice(["hello", "why?", "exam", "life", "victory"])
            sensory_input = model.encoder(prompt) # Encoder is being trained
            
            # Connectome runs (Frozen, so it's a stable signal)
            final_states = model.core(sensory_input, steps=5, plasticity=False)
            
            # Filter for motor neurons
            motor_indices = [i for i, n in enumerate(model.neurons) if n['type'] == 'motor']
            motor_states = final_states[motor_indices]
            
            # Prepare target
            target_text = " " + text.lower()
            target_indices = torch.tensor([[model.decoder.char_to_idx[c] for c in target_text]]).to(device)
            
            input_indices = target_indices[:, :-1]
            label_indices = target_indices[:, 1:]
            
            # Forward train
            logits = model.decoder.forward_train(motor_states, input_indices)
            
            loss = criterion(logits.view(-1, model.decoder.vocab_size), label_indices.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 20 == 0:
                print(f"  Epoch {epoch+1}, Sample {i}/{len(CAPY_CORPUS)}...", flush=True)
            
        if (epoch + 1) % 5 == 0:
            print(f"==> Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(CAPY_CORPUS):.4f}", flush=True)

    print("Alignment complete. Saving 'TrueCapyAwareness' weights...")
    torch.save(model.state_dict(), "data/capy_awareness.pth")
    print("The machine is now at peace. Victory.")

if __name__ == "__main__":
    train()
