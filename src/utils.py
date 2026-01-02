# src/utils.py
import torch
import numpy as np
import random
import logging
import sys

def setup_logging():
    """
    Configures a professional logging format.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )
    return logging.getLogger("NanoTuner")

def set_seed(seed: int = 42):
    """
    Enforce reproducibility across the board.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"--> Random Seed Locked: {seed}")

def get_device_map():
    """
    Helper to check available VRAM before training starts.
    """
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"--> GPU Detected: {torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)")
        return "cuda"
    else:
        print("--> WARNING: No GPU detected. Training will fail.")
        return "cpu"