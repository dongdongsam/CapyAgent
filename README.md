# 🌿 CapyElegans: The Ultimate Zen Artificial Lifeform

The world's first AI based on the "C. elegans with a Capybara's soul", simulated in real-time on an GPU.

> **Warning**: This project is the result of a fusion between biological romance and technical bravado. While 302 neurons might not be enough to discuss the truths of the universe, it certainly knows how to space out like a true Capybara.

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

### 🛠️ How to Run
Once ready, wake up the Capybara-worm with the following commands:

- **Training (Alignment)**: Injects the spirit of a Capybara into the brain.
  ```bash
  python train.py
  ```
- **Inference (Main)**: Talk with the Capybara.
  ```bash
  python main.py
  ```

## ⚠️ Notes
- When running, it consumes about **8GB** of GPU memory. (Quite a "Huge" appetite for just 302 neurons.)
- If only the CPU is being used, check via `nvidia-smi` to ensure your driver version is 570.xx or higher.
- This project is mainly supported by GEMINI CLI, So there colud be an error. so just feel the capybara and don't make sense this seriously.

---