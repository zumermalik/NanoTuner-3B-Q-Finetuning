from datasets import load_dataset

class DataHandler:
    def __init__(self, dataset_id, tokenizer):
        self.dataset_id = dataset_id
        self.tokenizer = tokenizer

    def format_prompts(self, examples):
        """
        Maps raw dataset examples to the specific instruction format.
        Adjust this based on whether you are using Alpaca, ChatML, etc.
        """
        # Example for a generic instruction dataset
        output_texts = []
        for i in range(len(examples['instruction'])):
            text = f"### User: {examples['instruction'][i]}\n\n### Assistant: {examples['output'][i]}"
            output_texts.append(text)
        return output_texts

    def load(self):
        print(f"--> Streaming Dataset: {self.dataset_id}")
        # Streaming=False for small datasets, True for massive ones
        ds = load_dataset(self.dataset_id, split="train") 
        return ds