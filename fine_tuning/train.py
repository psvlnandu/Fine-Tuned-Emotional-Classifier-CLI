# fine_tuning/train.py

import yaml
from transformers import AutoTokenizer, TrainingArguments, Trainer
from scripts.data_loader import load_and_preprocess_data
from scripts.trainer import get_model, get_trainer
import wandb

def main():

    wandb.init(project="huggingface", entity="psvlnandu-clarkson-university")
    learning_rate = wandb.config.learning_rate
    batch_size = wandb.config.per_device_train_batch_size

    # Load configuration from YAML file
    with open('fine_tuning/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Convert learning_rate to float
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    # Convert num_train_epochs and batch sizes to int
    config['training']['num_train_epochs'] = int(config['training']['num_train_epochs'])
    config['training']['per_device_train_batch_size'] = int(config['training']['per_device_train_batch_size'])
    config['training']['per_device_eval_batch_size'] = int(config['training']['per_device_eval_batch_size'])
    

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model, device = get_model(config['model_name'], num_labels=6)

    # Load and preprocess data
    ds_encoded = load_and_preprocess_data(config['dataset_name'], tokenizer)
    ds_encoded.set_format("torch")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        learning_rate=wandb.config.learning_rate,
        per_device_train_batch_size=wandb.config.per_device_train_batch_size,
        num_train_epochs=wandb.config.num_train_epochs, # Make sure this is also in your sweep config
        per_device_eval_batch_size=wandb.config.per_device_eval_batch_size, # Make sure this is also in your sweep config
        weight_decay=wandb.config.weight_decay,
    )

    # Get the trainer and start training
    trainer = get_trainer(model, tokenizer, ds_encoded, training_args)
    trainer.train()

    # Save the trained model
    trainer.save_model(config['output_dir'])

if __name__ == "__main__":
    main()