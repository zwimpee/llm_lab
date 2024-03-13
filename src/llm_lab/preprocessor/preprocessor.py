from datasets import load_dataset
from transformers import AutoTokenizer

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name"))
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess(self, dataset_type: str = "hf"):
        if dataset_type == "hf":
            return self.preprocess_hf(
                train_split=self.config.get("dataset")["splits"]["train_split"], 
                eval_split=self.config.get("dataset")["splits"]["eval_split"],
                test_split=self.config.get("dataset")["splits"].get("test_split")
            )
        else:
            raise NotImplementedError

    def preprocess_hf(self, train_split, eval_split, test_split=None):
        dataset = load_dataset(self.config.get("dataset")["name"], "all")

        def process_example_mmlu(example):
            # Format the question according to the required structure
            formatted_question = (f"The following are multiple choice questions\n"
                                  f"about {example['subject']}\n"
                                  f"{example['question']}\n"
                                  f"(A) {example['choices'][0]} "
                                  f"(B) {example['choices'][1]} "
                                  f"(C) {example['choices'][2]} "
                                  f"(D) {example['choices'][3]}\n"
                                  "Answer: ")

            encoded_input = self.tokenizer(formatted_question, padding="max_length", truncation=True, max_length=self.config.get("max_length"), return_tensors="pt")

            label = example['answer']
            
            return {'input_ids': encoded_input['input_ids'], 'attention_mask': encoded_input['attention_mask'], 'labels': label}

        dataset = dataset.map(process_example_mmlu, batched=False)

        train_dataset = dataset[train_split]
        eval_dataset = dataset[eval_split]
        test_dataset = dataset[test_split] if test_split else None

        return self.tokenizer, train_dataset, eval_dataset, test_dataset

