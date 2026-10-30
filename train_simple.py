import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_MET = os.path.join(PROJECT_ROOT, 'reports', 'metrics')
REPORTS_FIG = os.path.join(PROJECT_ROOT, 'reports', 'figures')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_MET, exist_ok=True)
os.makedirs(REPORTS_FIG, exist_ok=True)


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
    
    # Try Logistic Regression
    print("Training Logistic Regression...")
    pipe_lr = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=2000, random_state=42))
    ])
    pipe_lr.fit(X_train, y_train)
    y_pred_val_lr = pipe_lr.predict(X_val)
    acc_lr = accuracy_score(y_val, y_pred_val_lr)
    prec_lr, rec_lr, f1_lr, _ = precision_recall_fscore_support(y_val, y_pred_val_lr, average='weighted')
    results.append({'name': 'tfidf_logreg', 'val_accuracy': acc_lr, 'val_f1': f1_lr, 'val_precision': prec_lr, 'val_recall': rec_lr})
    print(f"LR val f1: {f1_lr:.4f}")
    
    # Try Naive Bayes
    print("Training Naive Bayes...")
    pipe_nb = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('clf', MultinomialNB())
    ])
    pipe_nb.fit(X_train, y_train)
    y_pred_val_nb = pipe_nb.predict(X_val)
    acc_nb = accuracy_score(y_val, y_pred_val_nb)
    prec_nb, rec_nb, f1_nb, _ = precision_recall_fscore_support(y_val, y_pred_val_nb, average='weighted')
    results.append({'name': 'tfidf_nb', 'val_accuracy': acc_nb, 'val_f1': f1_nb, 'val_precision': prec_nb, 'val_recall': rec_nb})
    print(f"NB val f1: {f1_nb:.4f}")
    
    # Select best
    if f1_lr >= f1_nb:
        best_model = pipe_lr
        best_name = 'tfidf_logreg'
    else:
        best_model = pipe_nb
        best_name = 'tfidf_nb'
    
    # Evaluate best on test
    print(f"Evaluating {best_name} on test set...")
    y_pred_test = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
    cm = confusion_matrix(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, target_names=['World', 'Sports', 'Business', 'Sci/Tech'])
    
    # Save artifacts
    model_path = os.path.join(MODELS_DIR, 'best_model_pipeline.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Saved model to {model_path}")
    
    metrics = {
        'best_model': best_name,
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'confusion_matrix': cm.tolist(),
        'all_results': results
    }
    with open(os.path.join(REPORTS_MET, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(REPORTS_MET, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    print('\n=== RESULTS ===')
    print('Best model:', best_name)
    print(f'Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}')
    print(report)
    return best_model, metrics


if __name__ == '__main__':
    train_models()
