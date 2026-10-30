import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
REPORTS_FIG = os.path.join(PROJECT_ROOT, 'reports', 'figures')
REPORTS_MET = os.path.join(PROJECT_ROOT, 'reports', 'metrics')

os.makedirs(REPORTS_FIG, exist_ok=True)
os.makedirs(REPORTS_MET, exist_ok=True)


def load_ag_news():
    """Load AG News CSV (headerless: class, title, description)"""
    train_path = os.path.join(DATA_DIR, 'ag_news_train.csv')
    test_path = os.path.join(DATA_DIR, 'ag_news_test.csv')
    df_train = pd.read_csv(train_path, header=None, names=['label', 'title', 'description'])
    df_test = pd.read_csv(test_path, header=None, names=['label', 'title', 'description'])
    return df_train, df_test


def prepare_data(test_size=0.1, random_state=42):
    df_train_full, df_test = load_ag_news()
    # Combine title and description
    df_train_full['text'] = df_train_full['title'].fillna('') + ' ' + df_train_full['description'].fillna('')
    df_test['text'] = df_test['title'].fillna('') + ' ' + df_test['description'].fillna('')
    # Split train/val
    df_train, df_val = train_test_split(df_train_full, test_size=test_size, stratify=df_train_full['label'], random_state=random_state)
    # Save processed
    df_train.to_csv(os.path.join(DATA_DIR, 'train_processed.csv'), index=False)
    df_val.to_csv(os.path.join(DATA_DIR, 'val_processed.csv'), index=False)
    df_test.to_csv(os.path.join(DATA_DIR, 'test_processed.csv'), index=False)
    # Save metadata
    meta = {
        'train_shape': df_train.shape,
        'val_shape': df_val.shape,
        'test_shape': df_test.shape,
        'class_dist_train': df_train['label'].value_counts().to_dict(),
        'class_dist_val': df_val['label'].value_counts().to_dict(),
        'class_dist_test': df_test['label'].value_counts().to_dict(),
        'label_map': {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
    }
    with open(os.path.join(REPORTS_MET, 'data_prep_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('Data prepared:', meta)
    return df_train, df_val, df_test


if __name__ == '__main__':
    prepare_data()
