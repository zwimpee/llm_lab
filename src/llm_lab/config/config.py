import json

class Config:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
    
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return json.load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)
