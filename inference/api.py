# inference/api.py
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from .pipeline import ThaiNLPPipeline

log = logging.getLogger(__name__)

# ── Global pipeline instance ──────────────────────────────────────────────────
pipeline: Optional[ThaiNLPPipeline] = None

SUPPORTED_TASKS = {"ner", "sentiment", "qa"}
MODEL_DIR       = os.environ.get("MODEL_DIR", "./outputs")


# ── Lifespan (แทน on_event deprecated ใน FastAPI 0.93+) ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    log.info(f"loading model from {MODEL_DIR}...")
    try:
        pipeline = ThaiNLPPipeline(model_dir=MODEL_DIR, device="auto")
        log.info("pipeline ready ✓")
    except Exception as e:
        log.error(f"ไม่สามารถโหลด model ได้: {e}")
        # ไม่ raise — ให้ server ขึ้นได้ แล้ว return 503 เมื่อถูกเรียก
    yield
    # cleanup (ถ้าต้องการ)
    pipeline = None


app = FastAPI(
    title="Thai NLP Multi-task API",
    version="1.0.0",
    description="NER, Sentiment Analysis, และ Question Answering สำหรับภาษาไทย",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text:     str
    tasks:    List[str]
    question: Optional[str] = None   # สำหรับ QA
    context:  Optional[str] = None   # สำหรับ QA

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, tasks):
        invalid = set(tasks) - SUPPORTED_TASKS
        if invalid:
            raise ValueError(
                f"task ไม่รู้จัก: {invalid} — รองรับแค่ {SUPPORTED_TASKS}"
            )
        if not tasks:
            raise ValueError("ต้องระบุ tasks อย่างน้อย 1 task")
        return tasks

    @field_validator("text")
    @classmethod
    def validate_text(cls, text):
        if not text or not text.strip():
            raise ValueError("text ต้องไม่ว่าง")
        if len(text) > 5000:
            raise ValueError("text ยาวเกินไป (max 5000 chars)")
        return text


class NERToken(BaseModel):
    token: str
    label: str

class SentimentResult(BaseModel):
    label:      str
    confidence: float
    scores:     Dict[str, float]

class QAResult(BaseModel):
    answer:     str
    start:      int
    end:        int
    confidence: float

class PredictResponse(BaseModel):
    ner:       Optional[List[NERToken]]    = None
    sentiment: Optional[SentimentResult]  = None
    qa:        Optional[QAResult]         = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — ใช้ใน Docker HEALTHCHECK และ load balancer"""
    return {
        "status":   "ok" if pipeline is not None else "model_not_loaded",
        "model_dir": MODEL_DIR,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="model ยังไม่พร้อม — โปรดรอหรือตรวจสอบ MODEL_DIR",
        )

    try:
        results = pipeline.predict(
            text=request.text,
            tasks=request.tasks,
            question=request.question,
            context=request.context,
        )
    except Exception as e:
        log.error(f"inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"inference error: {str(e)}")

    return PredictResponse(**results)


@app.get("/tasks")
async def list_tasks():
    """แสดง tasks ที่รองรับพร้อม input format"""
    return {
        "supported_tasks": list(SUPPORTED_TASKS),
        "usage": {
            "ner":       {"required": ["text"], "optional": []},
            "sentiment": {"required": ["text"], "optional": []},
            "qa":        {"required": ["text", "question", "context"], "optional": []},
        }
    }