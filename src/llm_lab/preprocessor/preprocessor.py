import logging
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name"))
        # Ensure the tokenizer is correctly set up with a padding token, if it doesn't already have one
        if self.tokenizer.pad_token is None:
            #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        dataset = load_dataset(self.config.get("dataset")["name"], *self.config.get("dataset").get("category"))

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

        train_data = processed_dataset[train_split].shuffle(seed=self.config.get("training_args").get("seed"))
        if max_train_examples := self.config.get("dataset").get("max_train_examples"):
            logger.info(f"Selecting {max_train_examples} examples from train split.")
            train_data = train_data.select(range(max_train_examples))
            
        eval_data = processed_dataset[eval_split].shuffle(seed=self.config.get("training_args").get("seed"))
        if max_eval_examples := self.config.get("dataset").get("max_eval_examples"):
            logger.info(f"Selecting {max_eval_examples} examples from eval split.")
            eval_data = eval_data.select(range(max_eval_examples))
        
        test_data = processed_dataset[test_split].shuffle(seed=self.config.get("training_args").get("seed")) if test_split else None
        if max_test_examples := self.config.get("dataset").get("max_test_examples") and test_data:
            logger.info(f"Selecting {max_test_examples} examples from test split.")
            test_data = test_data.select(range(max_test_examples))
            
        logger.info(f"{len(train_data)} train examples.")
        logger.info(f"{len(eval_data)} train examples.")
        logger.info(f"{len(test_data)} train examples.")

        return train_data, eval_data, test_data