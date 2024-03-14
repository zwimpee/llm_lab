from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding

import numpy as np
import evaluate

metric = evaluate.load("accuracy")



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class LLMLabTrainer:
    def __init__(self, model, tokenizer, config, train_data, val_data):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, 
            padding="max_length", 
            max_length=config.get("max_length"),
            pad_to_multiple_of=config.get("pad_to_multiple_of")
        )

        self.training_args = self.config.get("training_args", {})
        
        
        self.trainer = Trainer(
            model=self.model,
            tokenizer=tokenizer,
            data_collator=self.data_collator,
            args=TrainingArguments(**self.training_args),
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            compute_metrics=compute_metrics
        )

    def train(self):
        self.trainer.train()
        self.trainer.save_model(self.config.get("output_dir", "output"))
        self.trainer.save_state()
