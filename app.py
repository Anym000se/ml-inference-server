"""Text classification inference server.

Wraps a HuggingFace sequence-classification model in a FastAPI service with
the operational pieces a production deployment needs: eager model loading,
input validation, bounded batching, latency metrics, structured logging and
separate liveness/readiness probes.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from transformers import pipeline

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
MAX_CHARS = int(os.getenv("MAX_CHARS", "2000"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "32"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("inference")


def log_event(event: str, **fields: Any) -> None:
    """Emit one JSON object per line so logs are machine-parseable."""
    log.info(json.dumps({"event": event, **fields}))


class Metrics:
    """In-process counters. A real deployment would export these to Prometheus."""

    def __init__(self) -> None:
        self.requests = 0
        self.failures = 0
        self.latencies_ms: list[float] = []

    def observe(self, duration_ms: float, failed: bool = False) -> None:
        self.requests += 1
        if failed:
            self.failures += 1
        # Bounded so memory does not grow without limit under sustained load.
        self.latencies_ms.append(duration_ms)
        if len(self.latencies_ms) > 1000:
            self.latencies_ms.pop(0)

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(int(len(ordered) * p), len(ordered) - 1)
        return round(ordered[index], 2)


metrics = Metrics()
state: dict[str, Any] = {"classifier": None, "ready": False}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once at startup rather than per request.

    Readiness stays false until loading finishes, so an orchestrator will not
    route traffic to a pod that cannot serve it yet.
    """
    started = time.perf_counter()
    log_event("model_loading", model=MODEL_NAME)
    try:
        state["classifier"] = pipeline("sentiment-analysis", model=MODEL_NAME)
        state["ready"] = True
        log_event("model_loaded", seconds=round(time.perf_counter() - started, 2))
    except Exception as exc:
        log_event("model_load_failed", error=str(exc))
        raise
    yield
    state["ready"] = False
    log_event("shutdown")


app = FastAPI(title="Text Classification Inference Server", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        """Reject empty or oversized input before it reaches the model.

        Unbounded input is both a stability risk (memory, latency) and an
        untrusted-input surface. Rejecting early is cheaper than truncating late.
        """
        cleaned = []
        for text in values:
            stripped = text.strip()
            if not stripped:
                raise ValueError("texts must not contain empty strings")
            if len(stripped) > MAX_CHARS:
                raise ValueError(f"text exceeds MAX_CHARS ({MAX_CHARS})")
            cleaned.append(stripped)
        return cleaned


class Prediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float
    model: str


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="model not ready")

    started = time.perf_counter()
    try:
        raw = state["classifier"](request.texts)
    except Exception as exc:
        metrics.observe((time.perf_counter() - started) * 1000, failed=True)
        log_event("inference_failed", error=str(exc), batch_size=len(request.texts))
        # Deliberately generic: internal errors should not leak to the caller.
        raise HTTPException(status_code=500, detail="inference failed") from exc

    duration_ms = (time.perf_counter() - started) * 1000
    metrics.observe(duration_ms)
    log_event("inference_ok", batch_size=len(request.texts), latency_ms=round(duration_ms, 2))

    return PredictResponse(
        predictions=[Prediction(label=r["label"], score=round(float(r["score"]), 4)) for r in raw],
        latency_ms=round(duration_ms, 2),
        model=MODEL_NAME,
    )


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Liveness: is the process up? Used to decide whether to restart."""
    return JSONResponse({"status": "alive"})


@app.get("/readyz")
def readyz() -> JSONResponse:
    """Readiness: can it serve traffic? Used to decide whether to route."""
    if not state["ready"]:
        return JSONResponse({"status": "not ready"}, status_code=503)
    return JSONResponse({"status": "ready", "model": MODEL_NAME})


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    return "\n".join([
        f"inference_requests_total {metrics.requests}",
        f"inference_failures_total {metrics.failures}",
        f"inference_latency_ms_p50 {metrics.percentile(0.50)}",
        f"inference_latency_ms_p95 {metrics.percentile(0.95)}",
    ])
