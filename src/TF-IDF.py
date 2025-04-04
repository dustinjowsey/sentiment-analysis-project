import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_csv("train.csv", header=None, names=["comment", "label"])
test_df = pd.read_csv("test.csv", header=None, names=["comment", "label"])
print(train_df['comment'].isna().sum())
print(test_df['comment'].isna().sum())
train_df = train_df.dropna(subset=['comment'])
test_df = test_df.dropna(subset=['comment'])

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)

tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2),
    lowercase=True,
    strip_accents='unicode'
)

X_train = tfidf_vectorizer.fit_transform(train_df['comment'])
X_test = tfidf_vectorizer.transform(test_df['comment'])

print("TF-IDF Train shape:", X_train.shape)
print("TF-IDF Test shape:", X_test.shape)

feature_names = tfidf_vectorizer.get_feature_names_out()

df_tfidf_train = pd.DataFrame(X_train.toarray(), columns=feature_names)
df_tfidf_test = pd.DataFrame(X_test.toarray(), columns=feature_names)
df_tfidf_train['label'] = train_df['label'].values
df_tfidf_test['label'] = test_df['label'].values
df_tfidf_train.to_csv("TF-IDF_train.csv", index=False)
df_tfidf_test.to_csv("TF-IDF_test.csv", index=False)
print("Completion of TF-IDF data processing")

def plot_top_tfidf_words(df, top_n=20):
    tfidf_scores = df.drop(columns=['label']).mean().sort_values(ascending=False)[:top_n]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=tfidf_scores.values, y=tfidf_scores.index, palette='viridis')
    plt.title(f"Top {top_n} Words by Average TF-IDF Score (Train Set)")
    plt.xlabel("Average TF-IDF Score")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.show()

plot_top_tfidf_words(df_tfidf_train, top_n=20)
plot_top_tfidf_words(df_tfidf_test, top_n=20)
