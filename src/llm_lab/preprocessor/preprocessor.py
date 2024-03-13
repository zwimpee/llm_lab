from datasets import load_dataset
from transformers import AutoTokenizer

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name"))
        # Ensure the tokenizer is correctly set up with a padding token, if it doesn't already have one
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def preprocess(self, dataset_type: str = "hf"):
        if dataset_type == "hf":
            train_data, eval_data, test_data = self.preprocess_hf(
                train_split=self.config.get("dataset")["splits"]["train_split"], 
                eval_split=self.config.get("dataset")["splits"]["eval_split"],
                test_split=self.config.get("dataset")["splits"].get("test_split")
            )
            # Correctly return the tokenizer along with the processed datasets
            return self.tokenizer, train_data, eval_data, test_data
        else:
            raise NotImplementedError

    def preprocess_hf(self, train_split, eval_split, test_split=None):
        dataset = load_dataset(self.config.get("dataset")["name"], "all")

        def process_example(example):
            # Format the question according to the required structure
            formatted_question = (f"The following multiple choice question is"
                                  f"about {example['subject']}\n"
                                  f"{example['question']}\n"
                                  f"(A) {example['choices'][0]} "
                                  f"(B) {example['choices'][1]} "
                                  f"(C) {example['choices'][2]} "
                                  f"(D) {example['choices'][3]}\n"
                                  "Answer: ")

            # Encode the formatted question, ensuring tensor output format
            encoded_input = self.tokenizer(formatted_question, padding="max_length", truncation=True, max_length=self.config.get("max_length"), return_tensors="pt")
            
            # Flatten the tensors to avoid nesting issues
            input_ids = encoded_input['input_ids'].squeeze()
            attention_mask = encoded_input['attention_mask'].squeeze()
            label = example['answer']
            
            return {
                'input_ids': input_ids, 
                'attention_mask': attention_mask, 
                'label': label
            }

        # Apply processing function to the dataset, ensure batched processing is correctly handled
        processed_dataset = dataset.map(process_example, batched=False)

        train_data = processed_dataset[train_split]
        eval_data = processed_dataset[eval_split]
        test_data = processed_dataset[test_split] if test_split else None

        return train_data, eval_data, test_data