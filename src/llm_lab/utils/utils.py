from transformers import AutoModelForCausalLM
from transformers.data import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

def initialize_model(config):
    if config.get("model_type") == "CausalLM":
        return AutoModelForCausalLM.from_pretrained(config.get("model_name"))
    elif config.get("model_type") == "SequenceClassification":
        raise NotImplementedError
    elif config.get("model_type") == "TokenClassification":
        raise NotImplementedError
    elif config.get("model_type") == "QuestionAnswering":
        raise NotImplementedError
    else:
        raise ValueError(f"Model type {config.get('model_type')} not recognized.")

def get_dataloaders(config, tokenizer, train_dataset, val_dataset, test_dataset):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=config.get("pad_to_multiple_of")
    )

    train_dataloader = DataLoader(
        train_dataset, collate_fn=data_collator, batch_size=config.get("training_args")["per_device_train_batch_size"]
    )

    val_dataloader = DataLoader(
        val_dataset, collate_fn=data_collator, batch_size=config.get("training_args")["per_device_eval_batch_size"]
    )

    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=config.get("training_args")["per_device_eval_batch_size"]
    )

    return train_dataloader, val_dataloader, test_dataloader