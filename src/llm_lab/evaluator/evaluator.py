import evaluate 


class Evaluator:
    def __init__(self, config, model, test_data):
        self.config = config
        self.model = model
        self.test_data = test_data
        self.evaluator = evaluate.evaluator(config.get("dataset")["task_type"])

    def evaluate(self):
        return self.evaluator.compute(
            model_or_pipeline=self.model,
            data=self.test_data_loader
        )
        
