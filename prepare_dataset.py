import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Load the train CSV file
df = pd.read_csv("train.csv")

# We're focusing on classification here
df = df[["classification_prompt", "classification_label"]]

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer (Gemma)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# Tokenization function
def tokenize(example):
    return tokenizer(
        example["classification_prompt"],
        text_target=example["classification_label"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Apply tokenization
tokenized_dataset = dataset.map(tokenize, batched=True)

# Save tokenized dataset for training
tokenized_dataset.save_to_disk("tokenized_train")

