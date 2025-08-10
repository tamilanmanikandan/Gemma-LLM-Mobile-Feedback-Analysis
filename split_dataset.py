import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("prompt_dataset.csv")

# Stratified split for classification task
train_val, test = train_test_split(df, test_size=0.15, stratify=df['classification_label'], random_state=42)
train, val = train_test_split(train_val, test_size=0.1765, stratify=train_val['classification_label'], random_state=42)  # 0.1765 × 0.85 ≈ 0.15

# Save splits
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

print("✅ Dataset split complete. Files saved:")
print("- train.csv")
print("- val.csv")
print("- test.csv")

