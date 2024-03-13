from transformers import AutoModelForCausalLM
from transformers.data import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

def initialize_model(config):
    return AutoModelForCausalLM.from_pretrained(config.get("model_name"))

def get_dataloaders(config, tokenizer, train_dataset, val_dataset, test_dataset):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    train_dataloader = DataLoader(
        train_dataset, collate_fn=data_collator, batch_size=config.get("training_args")["batch_size"]
    )

    val_dataloader = DataLoader(
        val_dataset, collate_fn=data_collator, batch_size=config.get("training_args")["batch_size"]
    )

    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=config.get("training_args")["batch_size"]
    )

    return train_dataloader, val_dataloader, test_dataloader