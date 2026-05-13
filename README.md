# Sarcasm-Detection-nlp
#Machine Learning project to detect sarcasm using NLP
#1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# LOAD DATASET
data = []
with open("C:/dataset/Sarcasm_Headlines_Dataset.json", "r") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df.head()
#CLEAN TEXT
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean"] = df["headline"].apply(clean_text)
#Top 50 sarcastic headlines
top_sarcastic = df[df["is_sarcastic"]==1].head(50)
top_sarcastic
#Headline length distribution
df["length"] = df["headline"].apply(lambda x: len(x.split()))

plt.figure(figsize=(10,5))
sns.histplot(df[df.is_sarcastic==1]["length"], kde=True, label="Sarcastic", color="red")
sns.histplot(df[df.is_sarcastic==0]["length"], kde=True, label="Not Sarcastic", color="blue")
plt.legend()
plt.title("Headline Length Distribution")
plt.show()
#Most frequent sarcastic words
from collections import Counter

sarcastic_words = " ".join(df[df.is_sarcastic==1]["clean"]).split()
word_freq = Counter(sarcastic_words).most_common(30)

freq_df = pd.DataFrame(word_freq, columns=["Word", "Count"])

plt.figure(figsize=(10,6))
sns.barplot(data=freq_df, x="Count", y="Word")
plt.title("Most Common Sarcastic Words")
plt.show()
#BEST MODEL CONFUSION MATRIX (SVM)
cm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SVM (Best Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
