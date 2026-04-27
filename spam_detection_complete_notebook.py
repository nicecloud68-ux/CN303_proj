# ==============================
# 📩 Spam Detection Project (Final Fixed)
# ==============================

# 1. Imports
import pandas as pd
import numpy as np
import re
import string
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from scipy.sparse import hstack
import matplotlib.pyplot as plt

# ==============================
# 2. Load Data (FIXED PATH)
# ==============================

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "spam.csv")

df = pd.read_csv(file_path, encoding='latin-1')

# Normalize column names
df.columns = [col.lower() for col in df.columns

# Handle different dataset formats
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

elif 'label' in df.columns and 'message' in df.columns:
    df = df[['label', 'message']]

elif 'category' in df.columns and 'message' in df.columns:
    df = df[['category', 'message']]
    df.columns = ['label', 'message']

else:
    print("⚠️ Columns found:", df.columns)
    raise Exception("Unknown dataset format")

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ==============================
# 3. Feature Engineering
# ==============================

def extract_features(text):
    return {
        'num_urls': len(re.findall(r'http\S+', text)),
        'num_numbers': len(re.findall(r'\d+', text)),
        'num_exclamation': text.count('!'),
        'num_uppercase': sum(1 for c in text if c.isupper()),
        'msg_length': len(text)
    }

extra_features_df = df['message'].apply(lambda x: pd.Series(extract_features(x)))
df = pd.concat([df, extra_features_df], axis=1)

# ==============================
# 4. Text Cleaning
# ==============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', 'url', text)
    text = re.sub(r'\d+', 'num', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

X_text = df['message'].apply(clean_text)
y = df['label']

# ==============================
# 5. Train/Test Split
# ==============================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

extra_cols = ['num_urls', 'num_numbers', 'num_exclamation', 'num_uppercase', 'msg_length']
X_extra = df[extra_cols]

X_train_extra, X_test_extra = train_test_split(
    X_extra, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 6. TF-IDF + Combine Features
# ==============================

tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

X_train = hstack([X_train_tfidf, X_train_extra])
X_test = hstack([X_test_tfidf, X_test_extra])

# ==============================
# 7. Models
# ==============================

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True)
}

results = []

# ==============================
# 8. Training & Evaluation
# ==============================

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, acc, prec, rec, f1])

# ==============================
# 9. Results
# ==============================

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
print("\nModel Results:\n")
print(results_df.sort_values(by="F1", ascending=False))

# ==============================
# 10. Best Model
# ==============================

best_model_name = results_df.sort_values(by="F1", ascending=False).iloc[0]['Model']
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")

# ==============================
# 11. Confusion Matrix
# ==============================

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ==============================
# 12. Cross Validation
# ==============================

cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
print("\nCross-Validation F1 Score:", cv_scores.mean())

# ==============================
# ✅ DONE
# ==============================
