import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_MET = os.path.join(PROJECT_ROOT, 'reports', 'metrics')
MLRUNS_DIR = os.path.join(PROJECT_ROOT, 'mlruns')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_MET, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")
mlflow.set_experiment("news_article_classification")


def load_data():
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'train_processed.csv'))
    df_val = pd.read_csv(os.path.join(DATA_DIR, 'val_processed.csv'))
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test_processed.csv'))
    return df_train, df_val, df_test


def train_models():
    df_train, df_val, df_test = load_data()
    X_train, y_train = df_train['text'], df_train['label']
    X_val, y_val = df_val['text'], df_val['label']
    X_test, y_test = df_test['text'], df_test['label']
    
    results = []
    models_to_try = [
        ('tfidf_logreg', Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])),
        ('tfidf_nb', Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
            ('clf', MultinomialNB())
        ])),
    ]
    
    best_model = None
    best_score = -1
    best_name = None
    
    for name, pipe in models_to_try:
        with mlflow.start_run(run_name=name):
            mlflow.log_param('model_type', name)
            mlflow.log_param('vectorizer', 'tfidf')
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
            mlflow.log_metric('val_accuracy', acc)
            mlflow.log_metric('val_precision', prec)
            mlflow.log_metric('val_recall', rec)
            mlflow.log_metric('val_f1', f1)
            mlflow.sklearn.log_model(pipe, name, input_example=X_train.head(2).to_frame())
            results.append({
                'name': name, 'val_accuracy': acc, 'val_f1': f1, 
                'val_precision': prec, 'val_recall': rec
            })
            if f1 > best_score:
                best_score = f1
                best_model = pipe
                best_name = name
    
    # Evaluate best on test
    y_pred_test = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
    cm = confusion_matrix(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, target_names=['World', 'Sports', 'Business', 'Sci/Tech'])
    
    # Save artifacts
    model_path = os.path.join(MODELS_DIR, 'best_model_pipeline.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    metrics = {
        'best_model': best_name,
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'all_results': results
    }
    with open(os.path.join(REPORTS_MET, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(REPORTS_MET, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    print('Best model:', best_name)
    print('Test metrics:', test_acc, test_f1)
    print(report)
    return best_model, metrics


if __name__ == '__main__':
    train_models()
