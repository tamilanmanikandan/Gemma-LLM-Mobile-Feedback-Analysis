import pandas as pd

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

# Merge both datasets
df = pd.concat([google_df, apple_df], ignore_index=True)

# --- Label Mapping ---
label_map = {
    0: "feature request",
    1: "bug report",
    2: "general feedback"
}

# Ensure labels are mapped to string form
if df['label'].dtype != int:
    reverse_map = {v.lower(): k for k, v in label_map.items()}
    df['label'] = df['label'].str.lower().map(reverse_map)

# --- Best Classification Prompt ---
def create_classification_prompt(review):
    return f"""You are an AI assistant trained to analyze mobile app user feedback.

Your task is to classify the following review into one of these categories:
- Feature Request
- Bug Report
- General Feedback

Review: "{review}"

Category:"""

# --- Best Summarization Prompt ---
def create_summarization_prompt(review):
    return f"""You are an AI assistant.

Summarize the following mobile app user review in one concise sentence.

Review: "{review}"

Summary:"""

# Apply prompts
df['classification_prompt'] = df['clean_content'].apply(create_classification_prompt)
df['classification_label'] = df['label'].map(label_map)

df['summarization_prompt'] = df['clean_content'].apply(create_summarization_prompt)
df['summarization_target'] = df['content']

# Save final prompt dataset
df.to_csv("prompt_dataset.csv", index=False)
print("✅ prompt_dataset.csv created successfully.")