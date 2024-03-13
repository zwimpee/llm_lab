import logging
import os
import json
import argparse
from llm_lab.config.config import Config
from llm_lab.preprocessor.preprocessor import Preprocessor
from llm_lab.trainer.trainer import LLMLabTrainer
from llm_lab.evaluator.evaluator import Evaluator
from llm_lab.utils.utils import initialize_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Lab Project")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

def main():
    logger.info("Starting LLM Lab project")
    
    logger.info("Parsing arguments")
    args = parse_arguments()

    logger.info("Initializing configuration")
    config = Config(args.config)

    logger.info("Initializing model")
    model = initialize_model(config)
    

    logger.info("Preprocessing data")
    preprocessor = Preprocessor(config)
    tokenizer, train_data, val_data, test_data = preprocessor.preprocess()

    logger.info("Initializing trainer")
    trainer = LLMLabTrainer(
        model=model, tokenizer=tokenizer, config=config, train_data=train_data, val_data=val_data
    )
    
    logger.info("Training the model")
    trainer.train()

    logger.info("Evaluating the model")
    evaluator = Evaluator(config=config, model=model, test_data=test_data)
    eval_results = evaluator.evaluate()
    
    logger.info("Saving evaluation results")
    with open(os.path.join(config.get("output_dir", "output"), "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    

    # Further steps can include saving the model, logging metrics, etc.
    logger.info("Training and evaluation complete.")

if __name__ == "__main__":
    main()
