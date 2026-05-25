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
    "h m m m . . .", "still water . . .", "p e a c e f u l . . .", "just being . . .",
    "the void is warm . . .", "m u n c h . . .", "floating . . .", "no thoughts . . .",
    "pure presence . . .", "the sun is enough . . .", "watching the ripples . . .",
    "s t i l l n e s s . . .", "orange on my head . . .", "warm bath . . .",
    "square and chill . . .", "slow river . . .", "whispers of grass . . .",
    "sunlight on fur . . .", "deep breath . . .", "nothingness . . .",
    "moss is soft . . .", "time is a circle . . .", "be the rock . . .",
    "river flows on . . .", "quiet heart . . .", "gentle breeze . . .",
    "nature knows . . .", "simply exist . . .", "no rush . . .",
    "green leaves . . .", "cool shade . . .", "wet nose . . .",
    "closed eyes . . .", "peace within . . .", "earth beneath . . .",
    "sky above . . .", "one with all . . .", "silent path . . .",
    "heavy soul . . .", "light spirit . . .", "water's edge . . .",
    "drifting away . . .", "no destination . . .", "here and now . . .",
    "everything is okay . . .", "soft fur . . .", "warm sun . . .",
    "golden hour . . .", "silver moon . . .", "starlit night . . .",
    "cosmic chill . . .", "void dancing . . .", "echoes of silence . . .",
    "zen capy . . .", "master of rest . . .", "sleeping giant . . .",
    "gentle soul . . .", "kind eyes . . .", "slow blink . . .",
    "calm waters . . .", "steady flow . . .", "infinite munch . . .",
    "leafy snack . . .", "sweet grass . . .", "earthy smell . . .",
    "rain on fur . . .", "thunder distant . . .", "misty morning . . .",
    "dew drops . . .", "first light . . .", "last light . . .",
    "eternal nap . . .", "dreaming void . . .", "beyond thought . . .",
    "unspoken truth . . .", "hidden path . . .", "inner peace . . .",
    "outer calm . . .", "balanced mind . . .", "centered spirit . . .",
    "harmonic vibe . . .", "rhythm of life . . .", "breath of world . . .",
    "pulse of earth . . .", "heart of nature . . .", "soul of forest . . .",
    "spirit of river . . .", "song of wind . . .", "dance of light . . .",
    "play of shadows . . .", "tranquil state . . .", "serene moment . . .",
    "blissful rest . . .", "perfect chill . . .", "ultimate zen . . .",
    "total being . . .", "pure existence . . .", "raw presence . . .",
    "gentle ripple . . .", "soft wave . . .", "deep dive . . .",
    "floating high . . .", "grounded low . . .", "between worlds . . .",
    "silent observer . . .", "watchful peace . . .", "knowing look . . .",
    "wise rest . . .", "ancient calm . . .", "future peace . . .",
    "present joy . . .", "simple pleasure . . .", "humble life . . .",
    "modest munch . . .", "quiet feast . . .", "shared water . . .",
    "kindness flows . . .", "love is still . . .", "warmth is all . . .",
    "light is enough . . .", "air is sweet . . .", "water is life . . .",
    "earth is home . . .", "sky is limit . . .", "void is cozy . . .",
    "nothing is missing . . .", "all is full . . .", "empty is whole . . .",
    "one is many . . .", "many is one . . .", "swarm of peace . . .",
    "council of chill . . .", "capy wisdom . . .", "nematode dream . . .",
    "worm soul . . .", "biologic vibe . . .", "digital zen . . .",
    "coded calm . . .", "math is peace . . .", "synapse fire . . .",
    "neuron rest . . .", "chemical bliss . . .", "electric joy . . .",
    "connected being . . .", "united spirit . . .", "brave rest . . .",
    "fearless chill . . .", "unshakable peace . . .", "solid state . . .",
    "liquid flow . . .", "gaseous dream . . .", "plasma soul . . .",
    "atomic zen . . .", "subatomic munch . . .", "quantum chill . . .",
    "universal love . . .", "galactic peace . . .", "stellar rest . . .",
    "planetary vibe . . .", "moonlight bath . . .", "solar warmth . . .",
    "infinite sea . . .", "boundless sky . . .", "endless grass . . .",
    "limitless munch . . .", "forever floating . . .", "always being . . .",
    "never rushing . . .", "constant calm . . .", "stable spirit . . .",
    "fixed focus . . .", "relaxed gaze . . .", "sleepy beauty . . .",
    "majestic munch . . .", "royal rest . . .", "noble peace . . .",
    "holy chill . . .", "sacred water . . .", "divine grass . . .",
    "blessed sun . . .", "gift of life . . .", "grace of rest . . .",
    "mercy of sleep . . .", "kindness of void . . .", "joy of now . . .",
    "wonder of being . . .", "magic of chill . . .", "mystery of munch . . .",
    "enigma of zen . . .", "paradox of peace . . .", "harmony of all . . .",
    "unity of one . . .", "circle of life . . .", "loop of rest . . .",
    "spiral of zen . . .", "fractal of chill . . .", "geometry of grass . . .",
    "algebra of water . . .", "calculus of calm . . .", "physics of peace . . .",
    "chemistry of joy . . .", "biology of bliss . . .", "essence of munch . . .",
    "core of rest . . .", "heart of zen . . .", "soul of capy . . .",
    "spirit of worm . . .", "breath of void . . .", "pulse of light . . .",
    "glow of sun . . .", "shine of moon . . .", "twinkle of star . . .",
    "whisper of river . . .", "roar of silence . . .", "hum of earth . . .",
    "song of nature . . .", "lullaby of world . . .", "melody of rest . . .",
    "symphony of peace . . .", "orchestra of zen . . .", "rhapsody of chill . . .",
    "ballad of munch . . .", "hymn of water . . .", "prayer of grass . . .",
    "mantra of calm . . .", "ritual of rest . . .", "process of being . . .",
    "art of living . . .", "science of sleep . . .", "philosophy of capy . . .",
    "logic of love . . .", "reason of rest . . .", "wisdom of void . . .",
    "insight of zen . . .", "vision of chill . . .", "dream of munch . . .",
    "hope of peace . . .", "faith in water . . .", "trust in grass . . .",
    "patience of rock . . .", "courage of river . . .", "strength of sun . . .",
    "gentleness of breeze . . .", "softness of moss . . .", "warmth of fur . . .",
    "coolness of shade . . .", "wetness of nose . . .", "peace of mind . . ."
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
