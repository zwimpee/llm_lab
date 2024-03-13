import argparse
from llm_lab.config.config import Config
from llm_lab.preprocessor.preprocessor import Preprocessor
from llm_lab.trainer.trainer import LLMLabTrainer
from llm_lab.evaluator.evaluator import Evaluator
from llm_lab.utils.utils import initialize_model

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Lab Project")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration
    config = Config(args.config)

    model = initialize_model(config)
    

    # Preprocess data (you might need to adjust the parameters based on your config)
    preprocessor = Preprocessor(config)
    tokenizer, train_dataset, val_dataset, test_dataset = preprocessor.preprocess()

    # Initialize trainer
    trainer = LLMLabTrainer(
        model=model, tokenizer=tokenizer, config=config, train_dataset=train_dataset, val_dataset=val_dataset
    )
    
    # Train the model
    trainer.train()

    # Evaluate the model
    # evaluator = Evaluator(model=model, test_data_loader=None)
    # evaluation_metrics = evaluator.evaluate()

    # Further steps can include saving the model, logging metrics, etc.
    print("Training and evaluation complete.")

if __name__ == "__main__":
    main()
