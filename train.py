# NanoTuner-3B-Q-Finetuning/train.py

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import os

# --- 1. Load Configuration ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- 2. Setup BitsAndBytes Configuration for 4-bit Quantization (QLoRA) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Optimized for weights normally distributed
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation on A10G/T4
    bnb_4bit_use_double_quant=True,
)

# --- 3. Load Model and Tokenizer ---
print(f"Loading base model: {config['model_id']}")
model = AutoModelForCausalLM.from_pretrained(
    config["model_id"],
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.config.use_cache = False # Disable cache for training
model = prepare_model_for_kbit_training(model) # Prepare model for QLoRA

tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
tokenizer.pad_token = tokenizer.eos_token # Essential for SFTTrainer padding

# --- 4. Load Dataset ---
# Assuming your dataset on the Hub is formatted with a single 'text' column for SFTTrainer
print(f"Loading dataset: {config['dataset_id']}")
dataset = load_dataset(config["dataset_id"], split="train")

# --- 5. PEFT (LoRA) Configuration ---
peft_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    target_modules=config["lora_target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 6. Training Arguments ---
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    warmup_ratio=config["warmup_ratio"],
    learning_rate=config["learning_rate"],
    logging_steps=config["logging_steps"],
    save_steps=config["save_steps"],
    optim=config["optim"],
    fp16=config["fp16"],
    bf16=config["bf16"],
    # Integration with Hugging Face Hub
    push_to_hub=True,
    hub_model_id=os.environ.get("HF_MODEL_ID", "your-nano-tuner-model"), # Use ENV or a placeholder
    hub_private_repo=False,
    report_to=["wandb"], # If you set up wandb logging
)

# --- 7. Initialize and Start Trainer ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=2048, # Adjust based on your GPU memory and model
)

print("Starting training...")
trainer.train()

# --- 8. Save and Push Final Adapter ---
trainer.save_model()
trainer.push_to_hub()
print("Training complete and adapter pushed to Hugging Face Hub!")
