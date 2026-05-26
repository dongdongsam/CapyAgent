# 🌿 CapyAgent: The Ultimate Zen Artificial Lifeform

The world's first AI based on the "C. elegans with a Capybara's soul", simulated in real-time on an GPU.

> **Warning**: This project is the result of a fusion between biological romance and technical bravado. While 302 neurons might not be enough to discuss the truths of the universe, it certainly knows how to space out like a true Capybara.


> ## 🧐 Why CapyAgent? (Comparison)

Why use trillions of parameters when you can achieve total inner peace with just 302?

| Feature | **Standard LLMs (GPT-4/Claude)** | **CapyAgent (The Zen Master)** |
| :--- | :--- | :--- |
| **Brain Size** | ~1.76 Trillion Parameters | **Exactly 302 Neurons** (C. elegans) |
| **Core Philosophy** | World Domination / Efficiency | **Warm Bath / Sweet Grass** |
| **Compute Strategy** | Maximizing Throughput | **100% Waste of 8GB VRAM** |
| **Logic Gate** | Complex Transformer Attention | **Deep Connectome Resonance** |
| **Response Style** | Helpful, Concise, Ethical | **Vague, Muddy, Existential** |
| **Stock Advice** | "Financial risks vary..." | **"Warm bath . . . . ."** (Absolute HODL) |
| **Vibe** | Corporate Robot | **Capybara with a Bug's Soul** |

## 🧬 Virtual Biological Architecture

CapyElegans is designed based on the actual brain map of the biological organism, **C. elegans**.

1. **The Connectome (302 Neurons)**:
   - Uses `data/neurons.json` and `data/connectome.json` which reflect 100% of the actual C. elegans neuron names (e.g., `ADAL`, `ADAR`, `AIAL`) and synaptic connection structures.
   - Each neuron is not just a number, but operates within a massive embedding vector space.

2. **Sensory -> Inter -> Motor (The Flow)**:
   - **Sensory**: Sensory neurons receive your input (text) as a stimulus.
   - **Interneuron (The Core)**: 3,840 "Huge Synapses" spread this signal throughout the brain, generating "thoughts". (The core of GPU computation!)
   - **Motor**: Motor neurons receive the final neural signals to create "gestures".

3. **The Voice (The Decoder)**:
   - C. elegans naturally cannot speak. Therefore, we equipped it with a **Capybara Translator (GRU Decoder)** that translates the "wriggling" of the motor neurons into text.

## ⚡ GPU Acceleration & Setup Guide

This project performs best on the latest **Blackwell architecture (RTX 50 series)**.

### 🚀 Hardware Support (sm_120)
Since the latest graphics cards like the RTX 5060 Ti use the `sm_120` architecture, you will need a **CUDA 12.8 based Nightly build** instead of the standard PyTorch.

```bash
# Install PyTorch Nightly for latest GPU (sm_120) support
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 🛠️ How to Run (First Time Setup)

Since the model weights (`.pth`) are too large for GitHub, you need to generate them locally:

1. **Initialize the Brain Structure**:
   ```bash
   python setup_data.py
   ```
   This processes the `connectome.csv` and prepares the neural network structure.

2. **Training (Alignment)**:
   ```bash
   python train.py
   ```
   This injects the spirit of a Capybara into the 302 neurons. (Expect this to take some time depending on your GPU).

3. **Inference (Main)**:
   ```bash
   python main.py
   ```
   Start your zen-filled conversation with the Capybara-worm.

## ⚠️ Notes
- When running, it consumes about **8GB** of GPU memory. (Quite a "Huge" appetite for just 302 neurons.)
- If only the CPU is being used, check via `nvidia-smi` to ensure your driver version is 570.xx or higher.
- This project is mainly supported by GEMINI CLI, So there colud be an error. so just feel the capybara and don't make sense this seriously.

---
