# News Article Classification - Autonomous ML Capstone Project

## Project Overview
End-to-end machine learning pipeline for classifying news articles into 4 categories: World, Sports, Business, and Sci/Tech. Built autonomously with complete data pipeline, model training, MLflow tracking setup, FastAPI service, and Docker containerization.

## Problem Statement
Classify news articles based on their title and description into one of 4 categories to enable automated content categorization and routing.

## Dataset
- **Source**: AG News Corpus (via CharCnn_Keras GitHub repository)
- **Download URLs**: 
  - Train: https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
  - Test: https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv
- **Size**: 120,000 training samples + 7,600 test samples
- **Features**: class (label), title, description
- **Target**: class (1=World, 2=Sports, 3=Business, 4=Sci/Tech)
- **Task Type**: Multi-class Classification
- **License**: Publicly available for research

## Technology Stack
- **ML/Data**: scikit-learn, pandas, numpy
- **API**: FastAPI, Uvicorn, Pydantic
- **Tracking**: MLflow
- **Testing**: pytest, httpx
- **Containerization**: Docker

## Project Structure
```
project/
├── data/                    # Dataset files
├── src/
│   ├── data_preparation.py  # Data loading and preprocessing
│   ├── train.py             # Training with MLflow
│   ├── app.py               # FastAPI application
│   └── run_train.py         # Training script
├── models/                  # Trained model artifacts
├── reports/
│   ├── figures/             # Visualizations
│   └── metrics/             # Evaluation metrics
├── tests/                   # Automated tests
├── mlruns/                  # MLflow tracking
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── Makefile
└── README.md
```

## Results
- **Best Model**: TF-IDF (max_features=10000, ngram_range=1,2) + Logistic Regression
- **Test Accuracy**: 91.17%
- **Test F1-Score (weighted)**: 91.17%
- **Test Precision (weighted)**: 91.18%
- **Test Recall (weighted)**: 91.17%

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
python run_full.py
```

### Serve API Locally
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### Run Tests
```bash
pytest tests/ -v
```

### Docker
```bash
# Build
docker build -t news-classifier:latest .

# Run
docker run -p 8000:8000 news-classifier:latest
```

## API Endpoints
- `GET /` - API information
- `GET /health` - Health check and model status
- `GET /model-info` - Model metadata
- `POST /predict` - Make predictions

### Example Request
```json
POST /predict
{
  "text": "Wall Street stocks rally as tech companies report strong earnings"
}
```

### Example Response
```json
{
  "prediction": 3,
  "category": "Business",
  "text": "Wall Street stocks rally as tech companies report strong earnings"
}
```

## Known Limitations
- Model trained on AG News corpus may not generalize to all news domains
- Predictions are based on title+description text only
- No probability scores exposed (LogisticRegression supports but not currently returned)
- MLflow tracking uses file store (may need migration in future MLflow versions)

## Author
Autonomous ML Capstone Project - Completed end-to-end without manual intervention
