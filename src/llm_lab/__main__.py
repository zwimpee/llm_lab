import argparse
from llm_lab.config.config import Config
from llm_lab.preprocessor.preprocessor import Preprocessor
from llm_lab.trainer.trainer import LLMLabTrainer
from llm_lab.evaluator.evaluator import Evaluator
from llm_lab.utils.utils import initialize_model, load_model

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
    train_data, val_data, test_data = preprocessor.preprocess()

    # Initialize trainer
    trainer = LLMLabTrainer(model=model, config=config, train_dataset=train_data, val_dataset=val_data)

    # Train the model
    trainer.train()

    # Evaluate the model
    # evaluator = Evaluator(model=model, test_data_loader=None)
    # evaluation_metrics = evaluator.evaluate()

    # Further steps can include saving the model, logging metrics, etc.
    print("Training and evaluation complete.")

if __name__ == "__main__":
    main()
