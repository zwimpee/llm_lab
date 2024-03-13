from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import get_peft_config, get_peft_model

def initialize_model(config):
    if config.get("model_type") == "CausalLM":
        model = AutoModelForCausalLM.from_pretrained(config.get("model_name"))
    elif config.get("model_type") == "SequenceClassification":
        model = AutoModelForSequenceClassification.from_pretrained(config.get("model_name"), num_labels=config.get("num_labels"))
    elif config.get("model_type") == "TokenClassification":
        raise NotImplementedError
    elif config.get("model_type") == "QuestionAnswering":
        raise NotImplementedError
    else:
        raise ValueError(f"Model type {config.get('model_type')} not recognized.")
    
    if config.get("peft"):
        peft_config = get_peft_config(config.get("peft").get("lora"))
        model = get_peft_model(model, peft_config)
        
    return model