########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

########## 2. Define text preprocessing methods ##########

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

stop_words = stopwords.words('english')

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r'"', "", string)
    return string.strip().lower()

########## 3. Data loading and processing ##########

project = 'pytorch'
path = f'D:/datasets/{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)
pd_all['text'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

data = pd_all[['text', 'class']].rename(columns={'class': 'sentiment'}).fillna('')
data['text'] = data['text'].apply(remove_html).apply(remove_emoji).apply(remove_stopwords).apply(clean_str)

########## 4. Training and evaluation ##########

REPEAT = 10

# Metrics for LinearSVC
accuracies_svc, precisions_svc, recalls_svc, f1_scores_svc, aucs_svc = [], [], [], [], []

# Metrics for Naive Bayes
accuracies_nb, precisions_nb, recalls_nb, f1_scores_nb, aucs_nb = [], [], [], [], []

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X = vectorizer.fit_transform(data['text']).toarray()
y = data['sentiment']

params = {'var_smoothing': np.logspace(-12, 0, 13)}

for seed in range(REPEAT):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    smote = SMOTE(random_state=seed)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Linear SVC with SMOTE
    clf_svc = LinearSVC(random_state=seed)
    clf_svc.fit(X_train_balanced, y_train_balanced)
    y_pred_svc = clf_svc.predict(X_test)

    accuracies_svc.append(accuracy_score(y_test, y_pred_svc))
    precisions_svc.append(precision_score(y_test, y_pred_svc, average='macro'))
    recalls_svc.append(recall_score(y_test, y_pred_svc, average='macro'))
    f1_scores_svc.append(f1_score(y_test, y_pred_svc, average='macro'))
    aucs_svc.append(roc_auc_score(y_test, clf_svc.decision_function(X_test)))

    # Gaussian Naive Bayes with GridSearchCV and old AUC
    clf_nb = GridSearchCV(GaussianNB(), params, cv=5, scoring='roc_auc')
    clf_nb.fit(X_train, y_train)
    best_nb = clf_nb.best_estimator_
    y_pred_nb = best_nb.predict(X_test)

    accuracies_nb.append(accuracy_score(y_test, y_pred_nb))
    precisions_nb.append(precision_score(y_test, y_pred_nb, average='macro'))
    recalls_nb.append(recall_score(y_test, y_pred_nb, average='macro'))
    f1_scores_nb.append(f1_score(y_test, y_pred_nb, average='macro'))
    aucs_nb.append(roc_auc_score(y_test, y_pred_nb))

# Linear SVC results
print("=== Linear SVC ===")
print(f"Average Accuracy:      {np.mean(accuracies_svc):.4f}")
print(f"Average Precision:     {np.mean(precisions_svc):.4f}")
print(f"Average Recall:        {np.mean(recalls_svc):.4f}")
print(f"Average F1 score:      {np.mean(f1_scores_svc):.4f}")
print(f"Average AUC:           {np.mean(aucs_svc):.4f}")

# Naive Bayes results
print("\n=== Naive Bayes ===")
print(f"Average Accuracy:      {np.mean(accuracies_nb):.4f}")
print(f"Average Precision:     {np.mean(precisions_nb):.4f}")
print(f"Average Recall:        {np.mean(recalls_nb):.4f}")
print(f"Average F1 score:      {np.mean(f1_scores_nb):.4f}")
print(f"Average AUC:           {np.mean(aucs_nb):.4f}")

# Create DataFrame to save results
df_log = pd.DataFrame({
    'Model': ['LinearSVC', 'NaiveBayes'],
    'Accuracy': [np.mean(accuracies_svc), np.mean(accuracies_nb)],
    'Precision': [np.mean(precisions_svc), np.mean(precisions_nb)],
    'Recall': [np.mean(recalls_svc), np.mean(recalls_nb)],
    'F1': [np.mean(f1_scores_svc), np.mean(f1_scores_nb)],
    'AUC': [np.mean(aucs_svc), np.mean(aucs_nb)],
    'AUC_List': [str(aucs_svc), str(aucs_nb)]
})

# Save to CSV
df_log.to_csv(f'{project}_results.csv', index=False)
print(f"\nResults saved to: {project}_results.csv")
