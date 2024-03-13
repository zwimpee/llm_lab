from transformers import Trainer, TrainingArguments 

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class LLMLabTrainer:
    def __init__(self, model, tokenizer, config, train_dataloader, val_dataloader):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.training_args = self.config.get("training_args", {})
        
        
        self.trainer = Trainer(
            model=self.model,
            tokenizer=tokenizer,
            args=TrainingArguments(**self.training_args),
            train_dataset=self.train_dataloader,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics
        )

    def train(self):
        self.trainer.train()
        self.trainer.save_model(self.config.get("output_dir", "output"))
        self.trainer.save_state()
