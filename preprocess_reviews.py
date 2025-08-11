import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load datasets
google_df = pd.read_csv("google_reviews.csv")
apple_df = pd.read_csv("apple_reviews.csv")

# Select relevant columns
google_df = google_df[['reviewId', 'content', 'score', 'thumbsUpCount', 'at', 'replyContent']]
apple_df = apple_df[['title', 'review', 'rating', 'date']]

# Define stopwords
stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\\S+|www\\.\\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(filtered)

# Apply cleaning
google_df['clean_content'] = google_df['content'].apply(clean_text)
apple_df['clean_content'] = apple_df['review'].apply(clean_text)

# Simple rule-based labeling
def label_review(text):
    if any(word in text for word in ['add', 'please add', 'feature', 'it would be nice']):
        return 'Feature Request'
    elif any(word in text for word in ['crash', 'error', 'bug', 'not working']):
        return 'Bug Report'
    else:
        return 'General Feedback'

google_df['label'] = google_df['clean_content'].apply(label_review)
apple_df['label'] = apple_df['clean_content'].apply(label_review)

# Remove short/empty reviews
google_df = google_df[google_df['clean_content'].str.len() > 10]
apple_df = apple_df[apple_df['clean_content'].str.len() > 10]

# Export cleaned data
google_df.to_csv("google_preprocessed.csv", index=False)
apple_df.to_csv("apple_preprocessed.csv", index=False)

