import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load preprocessed dataset
google_df = pd.read_csv("google_preprocessed.csv")
apple_df = pd.read_csv("apple_preprocessed.csv")
df = pd.concat([google_df, apple_df], ignore_index=True)
# Basic overview
print(df.info())
print(df['label'].value_counts())

# 📈 1. Sentiment Score Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='score', palette='Blues')
plt.title('Review Score Distribution')
plt.savefig("score_distribution.png")
plt.close()

# 🔠 2. Review Length Distribution
df['length'] = df['clean_content'].apply(lambda x: len(x.split()))
plt.figure(figsize=(6,4))
sns.histplot(df['length'], bins=30)
plt.title('Review Length Distribution')
plt.savefig("review_length_distribution.png")
plt.close()

# 🧾 3. Label Distribution
plt.figure(figsize=(5,4))
sns.countplot(data=df, x='label', palette='pastel')
plt.title('Label Distribution')
plt.savefig("label_distribution.png")
plt.close()

# 🗓️ 4. Time Trends
df['at'] = pd.to_datetime(df['at'], errors='coerce')
df.set_index('at', inplace=True)
df['count'] = 1
weekly_counts = df.resample('W').sum()['count']
weekly_counts.plot(figsize=(8,4), title='Weekly Review Volume')
plt.savefig("weekly_volume_trend.png")
plt.close()

# 📉 5. Word Cloud
text = ' '.join(df['clean_content'].dropna().values)
wc = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Reviews')
plt.savefig("wordcloud.png")
plt.close()

# ❗ 6. Missing Values
print("\nMissing values:\n", df.isnull().sum())

