import os
import torch
from transformers import TrainingArguments
from trl import SFTTrainer

class NanoTrainer:
    def __init__(self, model, tokenizer, dataset, peft_config, config):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.peft_config = peft_config
        self.cfg = config

    def _get_training_args(self):
        """
        Constructs the TrainingArguments object from the config.
        Optimized for 3B parameter models on consumer hardware.
        """
        # We assume A10G/T4/A100 support (bf16 or fp16)
        use_bf16 = torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16

        return TrainingArguments(
            output_dir="./results",
            
            # Batching & Gradient Accumulation
            per_device_train_batch_size=self.cfg['training']['batch_size'],
            gradient_accumulation_steps=self.cfg['training']['grad_accum'],
            
            # Learning Rate & Scheduler
            learning_rate=self.cfg['training']['learning_rate'],
            warmup_ratio=self.cfg['training']['warmup_ratio'],
            lr_scheduler_type="cosine",
            
            # Epochs & Logging
            num_train_epochs=self.cfg['training']['epochs'],
            logging_steps=10,
            report_to="tensorboard",  # Or "wandb" if configured
            
            # Memory Optimizations for QLoRA
            fp16=use_fp16,
            bf16=use_bf16,
            optim=self.cfg['training']['optimizer'],
            gradient_checkpointing=True, # Critical for saving VRAM
            group_by_length=True,        # Speeds up training by grouping similar length prompts
            
            # Hugging Face Hub Integration (The "No Local Storage" Solution)
            push_to_hub=self.cfg['hub']['push'],
            hub_model_id=self.cfg['hub']['model_id'],
            hub_private_repo=self.cfg['hub']['private'],
            hub_strategy="every_save",   # Pushes checkpoints as they happen
        )

    def train(self):
        print("--> Initializing SFTTrainer...")
        args = self._get_training_args()

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            dataset_text_field="text", # Ensure your data handler outputs this key
            max_seq_length=self.cfg['training']['context_window'],
            tokenizer=self.tokenizer,
            args=args,
            packing=False, # Set to True if you want to pack multiple examples into one sequence
        )

        print(f"--> Starting Training on {args.num_train_epochs} epochs...")
        trainer.train()

        # Final cleanup and push
        if self.cfg['hub']['push']:
            print("--> Pushing final adapter to Hub...")
            trainer.push_to_hub()
            print("--> Push complete.")