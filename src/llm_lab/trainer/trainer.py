from transformers import Trainer, TrainingArguments 

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class LLMLabTrainer:
    def __init__(self, model, config, train_dataset, val_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        self.training_args = self.config.get("training_args", {})
        
        
        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(**self.training_args),
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics
        )

    def train(self):
        trainer_return1, trainer_return2, trainer_return3 = self.trainer.train()
        self.trainer.save_model(self.config.get("output_dir", "output"))
        self.trainer.save_state()
