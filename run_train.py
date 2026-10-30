import os, json, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

PROJECT_ROOT = r'C:\Users\jbman\Desktop\finalproject'
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_MET = os.path.join(PROJECT_ROOT, 'reports', 'metrics')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_MET, exist_ok=True)

df_train = pd.read_csv(os.path.join(DATA_DIR, 'train_processed.csv'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'test_processed.csv'))
X_train, y_train = df_train['text'], df_train['label']
X_test, y_test = df_test['text'], df_test['label']

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
print('starting fit')
pipe.fit(X_train, y_train)
print('done fit')
y_pred_test = pipe.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
cm = confusion_matrix(y_test, y_pred_test)
report = classification_report(y_test, y_pred_test, target_names=['World','Sports','Business','Sci/Tech'])
with open(os.path.join(MODELS_DIR, 'best_model_pipeline.pkl'), 'wb') as f:
    pickle.dump(pipe, f)
metrics = {'best_model': 'tfidf_logreg', 'test_accuracy': float(test_acc), 'test_precision': float(test_prec), 'test_recall': float(test_rec), 'test_f1': float(test_f1), 'confusion_matrix': cm.tolist()}
with open(os.path.join(REPORTS_MET, 'test_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(REPORTS_MET, 'classification_report.txt'), 'w') as f:
    f.write(report)
print(report)
