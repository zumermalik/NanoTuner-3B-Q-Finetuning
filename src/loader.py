import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

class ModelLoader:
    def __init__(self, config):
        self.config = config

    def load(self):
        """
        Loads the base model with 4-bit quantization and the tokenizer.
        """
        print(f"--> Loading Base Model: {self.config['model']['id']}")

        # 1. Define Quantization Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config['quantization']['load_in_4bit'],
            bnb_4bit_quant_type=self.config['quantization']['quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, self.config['model']['dtype']),
            bnb_4bit_use_double_quant=self.config['quantization']['use_double_quant'],
        )

        # 2. Load Model
        model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['id'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config['model']['trust_remote_code']
        )
        
        # Optimization for training stability
        model.config.use_cache = False 
        model = prepare_model_for_kbit_training(model)

        # 3. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['id'],
            trust_remote_code=True
        )
        # Fix padding issues common in Llama/Phi models
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fixed for fp16 training stability

        return model, tokenizer