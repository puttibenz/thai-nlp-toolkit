from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="Thai NLP Multi-task API", version="1.0.0")

class PredictRequest(BaseModel):
    text: str
    tasks: List[str]

class PredictResponse(BaseModel):
    # TODO: Refine the schema matching the spec
    # { "ner": ..., "sentiment": ..., "qa": ... }
    predictions: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    # TODO: Load pipeline instance
    pass

@app.post("/predict")
async def predict(request: PredictRequest):
    # TODO: Run pipeline inference
    return {"status": "ok", "predictions": {}}
