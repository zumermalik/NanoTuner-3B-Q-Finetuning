# main.py
import yaml
import os
import sys
from peft import LoraConfig
from huggingface_hub import login

# Import our custom modules
from src.loader import ModelLoader
from src.data import DataHandler
from src.trainer import NanoTrainer
from src.utils import setup_logging, set_seed, get_device_map

# Constant
CONFIG_PATH = "configs/default.yaml"

def main():
    # 1. Setup & Hygiene
    logger = setup_logging()
    logger.info("Starting NanoTuner-3B Pipeline...")
    
    # Check for Hugging Face Token (Critical for "No Local Storage")
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
        logger.info("Logged into Hugging Face Hub successfully.")
    else:
        logger.warning("HF_TOKEN not found in env. Push to Hub might fail!")

    # Load Configuration
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config file not found at {CONFIG_PATH}")
        sys.exit(1)
        
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Set reproducibility
    set_seed(42)
    device = get_device_map()

    # 2. Load Resources (Model & Data)
    logger.info(f"Loading Base Model: {cfg['model']['id']}")
    loader = ModelLoader(cfg)
    model, tokenizer = loader.load()

    logger.info("Loading and Formatting Dataset...")
    data_handler = DataHandler(cfg['model']['dataset_id'], tokenizer) 
    # Note: Make sure 'dataset_id' is in your yaml, or hardcode/pass it here
    dataset = data_handler.load()

    # 3. Define LoRA Configuration (The Adapter)
    logger.info(f"Configuring LoRA (Rank: {cfg['lora']['r']})...")
    peft_config = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        lora_dropout=cfg['lora']['dropout'],
        bias=cfg['lora']['bias'],
        task_type="CAUSAL_LM",
        target_modules=cfg['lora']['target_modules']
    )

    # 4. Initialize Trainer
    # This class handles the complexity of TrainingArguments and Hub pushing
    trainer = NanoTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        peft_config=peft_config,
        config=cfg
    )

    # 5. Execute
    logger.info(">>> BEGINNING FINE-TUNING <<<")
    trainer.train()
    
    logger.info(">>> PIPELINE COMPLETE <<<")

if __name__ == "__main__":
    main()