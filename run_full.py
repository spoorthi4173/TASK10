import os, sys, json, pickle, time, subprocess
from pathlib import Path

PROJECT_ROOT = r'C:\Users\jbman\Desktop\finalproject'
os.chdir(PROJECT_ROOT)

# Setup dirs
for d in ['data','src','models','reports/figures','reports/metrics','tests','mlruns']:
    os.makedirs(d, exist_ok=True)

# Download data
import requests
data_dir = os.path.join(PROJECT_ROOT,'data')
urls = [
    ('https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv', 'ag_news_train.csv'),
    ('https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv', 'ag_news_test.csv'),
]
for url,name in urls:
    if not os.path.exists(os.path.join(data_dir,name)):
        r = requests.get(url, timeout=60)
        open(os.path.join(data_dir,name),'wb').write(r.content)
        print('down',name,len(r.content))

# Prepare data
import pandas as pd
from sklearn.model_selection import train_test_split
df_train_full = pd.read_csv(os.path.join(data_dir,'ag_news_train.csv'), header=None, names=['label','title','description'])
df_test = pd.read_csv(os.path.join(data_dir,'ag_news_test.csv'), header=None, names=['label','title','description'])
df_train_full['text'] = df_train_full['title'].fillna('')+' '+df_train_full['description'].fillna('')
df_test['text'] = df_test['title'].fillna('')+' '+df_test['description'].fillna('')
df_train, df_val = train_test_split(df_train_full, test_size=0.1, stratify=df_train_full['label'], random_state=42)
df_train.to_csv(os.path.join(data_dir,'train_processed.csv'), index=False)
df_val.to_csv(os.path.join(data_dir,'val_processed.csv'), index=False)
df_test.to_csv(os.path.join(data_dir,'test_processed.csv'), index=False)
print('prepared')

# Train
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
MODELS_DIR = os.path.join(PROJECT_ROOT,'models')
REPORTS_MET = os.path.join(PROJECT_ROOT,'reports','metrics')
X_train,y_train = df_train['text'], df_train['label']
X_test,y_test = df_test['text'], df_test['label']
pipe = Pipeline([('tfidf',TfidfVectorizer(max_features=10000,ngram_range=(1,2))),('clf',LogisticRegression(max_iter=1000,random_state=42))])
print('fit...')
pipe.fit(X_train,y_train)
print('predict...')
y_pred = pipe.predict(X_test)
test_acc = accuracy_score(y_test,y_pred)
test_prec,test_rec,test_f1,_ = precision_recall_fscore_support(y_test,y_pred,average='weighted')
cm = confusion_matrix(y_test,y_pred)
report = classification_report(y_test,y_pred,target_names=['World','Sports','Business','Sci/Tech'])
with open(os.path.join(MODELS_DIR,'best_model_pipeline.pkl'),'wb') as f: pickle.dump(pipe,f)
metrics = {'best_model':'tfidf_logreg','test_accuracy':float(test_acc),'test_precision':float(test_prec),'test_recall':float(test_rec),'test_f1':float(test_f1),'confusion_matrix':cm.tolist()}
with open(os.path.join(REPORTS_MET,'test_metrics.json'),'w') as f: json.dump(metrics,f,indent=2)
open(os.path.join(REPORTS_MET,'classification_report.txt'),'w').write(report)
print(report)
print('DONE')
