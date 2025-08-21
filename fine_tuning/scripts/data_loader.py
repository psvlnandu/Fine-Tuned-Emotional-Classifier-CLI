# fine_tuning/scripts/data_loader.py

from datasets import load_dataset

def load_and_preprocess_data(dataset_name, tokenizer):
    ds = load_dataset(dataset_name)
    ds_encoded = ds.map(
        lambda examples: tokenizer(examples['text'], padding=True, truncation=True),
        batched=True
    )
    return ds_encoded

# Note: The `ds.set_format` and pandas-related analysis from your notebook
# are for exploration and can be left out of the production script.