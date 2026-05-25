# CapyAgent Project Memory

## Architecture: Swarm Intelligence (v2)
- **Concept:** 10,000 lightweight C. elegans brains working in parallel.
- **Core (`src/simulator.py`):** `CapySwarmCore` manages 10,000 worms. Each worm has a unique set of scalar synapse weights (Horizontal Scaling).
- **Encoder (`src/encoder.py`):** BERT-based, outputs scalar sensory stimuli for the swarm.
- **Decoder (`src/decoder.py`):** Reads the motor neurons of **all 10,000 worms** (flattened) to generate text. This "Complex Readout" (approx. 1.14M features) acts as the swarm's collective intelligence layer.
- **Parameters:** Total ~382M. Decoder projection layer is the largest (~291M).
- **Philosophy:** Scalability via population size ("1 worm + 1 worm = 2 worms' intelligence").

## Key Files
- `src/model.py`: Orchestrates the swarm, encoder, and decoder.
- `train.py`: Aligning the swarm's collective output to "Capybara Zen" responses.
- `data/neurons.json`: Biological neuron data (304 neurons).
- `data/connectome.json`: Biological synapse connectivity (~3,840 connections).
