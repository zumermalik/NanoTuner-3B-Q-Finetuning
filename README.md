# NanoTuner-3B: QLoRA Fine-Tuning Pipeline

A modular, containerized framework designed to fine-tune 3B-parameter LLMs (like Phi-3, Llama-3.2) on constrained compute environments. 

## ⚡ The Philosophy
The 3B parameter class represents a "sweet spot" in modern AI engineering: small enough for edge deployment, yet capable enough for complex reasoning. 

This project was architected to solve the **"No Local Storage"** problem. It operates as an ephemeral training rig:
1. **Stream** data directly into memory.
2. **Quantize** the base model on-the-fly (4-bit NF4).
3. **Train** utilizing QLoRA adapters.
4. **Push** artifacts immediately to the Hugging Face Hub.

## 🏗️ Architecture

```bash
NanoTuner-3B/
├── configs/       # YAML-driven hyperparameter management
├── src/           # Decoupled logic (Data, Model, Training)
├── Dockerfile     # CUDA-optimized runtime environment
└── main.py        # Entry point