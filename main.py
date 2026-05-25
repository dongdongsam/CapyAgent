import os
# Force PyTorch to recognize RTX 50 series by falling back to sm_90 compatibility
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import torch
import sys
import time
import os
from src.model import CapyElegansLLM

def main():
    print("Initializing CapyElegans (302 Neurons, Billions of (Dummy) Parameters)...")
    
    # We use 286 for 5x parameters
    embedding_dim = 286
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if data files exist
    if not os.path.exists("data/neurons.json") or not os.path.exists("data/connectome.json"):
        print("❌ Error: Connectome data not found.")
        print("Please run 'python setup_data.py' first to prepare the brain structure.")
        return

    try:
        model = CapyElegansLLM(embedding_dim=embedding_dim, device=device)
        if os.path.exists("data/capy_awareness.pth"):
            print("Loading pre-aligned CapyAwareness weights...")
            model.load_state_dict(torch.load("data/capy_awareness.pth", map_location=device))
        else:
            print("⚠️ Warning: 'data/capy_awareness.pth' not found.")
            print("The model is running with a 'Blank Soul' (random weights).")
            print("To inject the Capybara spirit, please run 'python train.py' first.")
            print("Continuing in 3 seconds...\n")
            time.sleep(3)
    except Exception as e:
        print(f"Error initializing model: {e}")
        model = CapyElegansLLM(embedding_dim=64, device=device)
    
    print("\n" + "="*50)
    print("🌿 CapyElegans: The Ultimate Chill AI is Ready.")
    print("Type anything to see what the 302-neuron brain thinks.")
    print("Type 'exit' to quit.")
    print("="*50 + "\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        print("\n[System] Deep Connectome Resonance Initiated...")
        
        # Poetic 'heartbeat' visualization
        heartbeats = ["Pulse...", "Rhythm...", "Flow...", "Stillness..."]
        for beat in heartbeats:
            time.sleep(0.6)
            print(f"  > {beat}", flush=True)
        
        with torch.no_grad():
            # Lower temperature for more stable, zen outputs
            # Enable training for plasticity even during inference
            model.train() 
            response = model(user_input, temperature=0.3)
            
        print("\nCapy: ", end="", flush=True)
        # Typewriter effect for zen-like output
        for char in response:
            print(char, end="", flush=True)
            time.sleep(0.15)
        print("\n")

if __name__ == "__main__":
    main()
