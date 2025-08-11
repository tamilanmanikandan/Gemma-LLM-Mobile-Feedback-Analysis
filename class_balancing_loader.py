import pandas as pd
import json
from collections import Counter

# Load preprocessed datasets
google_df = pd.read_csv("google_preprocessed.csv")
apple_df = pd.read_csv("apple_preprocessed.csv")

# Rename Apple 'review' column to match Google's 'content' column
apple_df = apple_df.rename(columns={'review': 'content'})

# Keep only necessary columns
google_df = google_df[['clean_content', 'content', 'label']]
apple_df = apple_df[['clean_content', 'content', 'label']]

# Optional: Add platform column
google_df['platform'] = 'Google'
apple_df['platform'] = 'Apple'
LABEL_COL = "label"
# Merge both datasets
df = pd.concat([google_df, apple_df], ignore_index=True)

# --- ENCODE LABELS (consistent integer mapping) ---
if df[LABEL_COL].dtype == object:
    label2id = {label: idx for idx, label in enumerate(sorted(df[LABEL_COL].unique()))}
    id2label = {v: k for k, v in label2id.items()}
    df[LABEL_COL] = df[LABEL_COL].map(label2id)
else:
    id2label = {int(l): str(l) for l in df[LABEL_COL].unique()}
    label2id = {v: k for k, v in id2label.items()}

# --- CALCULATE CLASS WEIGHTS (inverse frequency) ---
counts = Counter(df[LABEL_COL])
num_samples, num_classes = len(df), len(counts)
class_weights = {cls: num_samples / (num_classes * count) for cls, count in counts.items()}

# --- SAVE OUTPUT FILES ---
with open("class_weights.json", "w") as f: json.dump(class_weights, f, indent=4)
with open("label2id.json", "w") as f: json.dump(label2id, f, indent=4)
with open("id2label.json", "w") as f: json.dump(id2label, f, indent=4)

print("Class weights saved to class_weights.json")
print("Label mappings saved to label2id.json and id2label.json")

