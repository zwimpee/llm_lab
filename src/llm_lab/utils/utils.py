from transformers import AutoModelForCausalLM

def initialize_model(config):
    return AutoModelForCausalLM.from_pretrained(config.get("model_name"))

def load_model():
    ...
