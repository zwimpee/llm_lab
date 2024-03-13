from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

def initialize_model(config):
    if config.get("model_type") == "CausalLM":
        return AutoModelForCausalLM.from_pretrained(config.get("model_name"))
    elif config.get("model_type") == "SequenceClassification":
        return AutoModelForSequenceClassification.from_pretrained(config.get("model_name"), num_labels=config.get("num_labels"))
    elif config.get("model_type") == "TokenClassification":
        raise NotImplementedError
    elif config.get("model_type") == "QuestionAnswering":
        raise NotImplementedError
    else:
        raise ValueError(f"Model type {config.get('model_type')} not recognized.")