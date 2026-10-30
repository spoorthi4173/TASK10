import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

app = FastAPI(title="News Article Classification API", description="End-to-end ML capstone project", version="1.0.0")

model = None
label_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


class PredictionRequest(BaseModel):
    text: str = Field(..., description="News article title and/or description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Wall Street stocks rally as tech companies report strong earnings"
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    category: str
    text: str


def load_model():
    global model
    if model is not None:
        return model
    model_path = os.path.join(MODELS_DIR, 'best_model_pipeline.pkl')
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
    return model


@app.on_event("startup")
async def startup_event():
    load_model()



@app.get("/")
async def root():
    return {
        "message": "News Article Classification API",
        "version": "1.0.0",
        "endpoints": ["/health", "/model-info", "/predict"]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.get("/model-info")
async def model_info():
    return {
        "model_name": "tfidf_logreg",
        "model_type": "TfidfVectorizer + LogisticRegression",
        "version": "1.0.0",
        "classes": ["World", "Sports", "Business", "Sci/Tech"],
        "label_mapping": label_map
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        pred = model.predict([request.text])[0]
        category = label_map.get(int(pred), str(pred))
        return PredictionResponse(
            prediction=int(pred),
            category=category,
            text=request.text
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
