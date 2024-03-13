from datasets import load_dataset
from transformers import AutoTokenizer
from llm_lab.utils.utils import get_dataloaders

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("model_name"))
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess(self, dataset_type: str = "hf"):
        if dataset_type == "hf":
            return self.preprocess_hf(
                train_split=self.config.get("dataset")["splits"]["train_split"], 
                eval_split=self.config.get("dataset")["splits"]["eval_split"],
                test_split=self.config.get("dataset")["splits"]["test_split"] if "test_split" in self.config.get("dataset")["splits"] else None
            )
        else:
            raise NotImplementedError
        
    
        
    def preprocess_hf(self, train_split, eval_split, test_split=None):
        dataset = load_dataset(self.config.get("dataset")["name"], "all")
        
        # Function to process each example, including encoding the labels
        def process_example(examples):
            # Concatenate question with each choice
            texts = [example['question'] + " " + choice for example in examples for choice in example['choices']]
            # Encode concatenated texts
            encoded_inputs = self.tokenizer.batch_encode_plus(texts, padding="max_length", truncation=True, max_length=self.config.get("max_length"), return_tensors="pt")
            
            return encoded_inputs
        
        # Function to process each example
        # def process_example(example):
        #     # Concatenate question with each choice
        #     texts = [example['question'] + " " + choice for choice in example['choices']]
        #     # Encode concatenated texts
        #     return self.tokenizer.batch_encode_plus(texts, padding="max_length", truncation=True, max_length=self.config.get("max_length"), return_tensors="pt")

        # Apply processing function to the dataset
        dataset = dataset.map(process_example, batched=True, batch_size=self.config.get("training_args")["per_device_train_batch_size"])  # Set batched=False if processing each example individually

        train_data = dataset[train_split]
        eval_data = dataset[eval_split]
        test_data = dataset[test_split] if test_split else None
        train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(self.config, self.tokenizer, train_data, eval_data, test_data)
        return self.tokenizer, train_dataloader, eval_dataloader, test_dataloader

