# ML Inference Server

A small FastAPI service that wraps a HuggingFace sequence-classification model
behind an HTTP API, with the operational scaffolding a production deployment
actually needs.

The model itself is deliberately uninteresting — a stock sentiment classifier.
The point of this project is the **engineering around** the model: how it is
loaded, how input is validated, how failures are handled, how the service
reports whether it is healthy, and how it is observed once it is running.

## Why these design decisions

**The model is loaded once at startup, not per request.**
Loading happens in the FastAPI `lifespan` handler. Loading per request would add
seconds of latency to every call and duplicate the model in memory under
concurrency.

**Liveness and readiness are separate endpoints.**
`/healthz` answers "is the process alive?" — an orchestrator uses it to decide
whether to restart the container. `/readyz` answers "can it serve traffic?" and
returns 503 until the model has finished loading. Conflating the two causes a
classic failure: traffic gets routed to a pod that is up but cannot yet answer,
or a slow-loading pod gets killed and restarted in a loop.

**Input is validated and bounded before it reaches the model.**
Batch size is capped at `MAX_BATCH` and individual inputs at `MAX_CHARS`.
Unbounded input is both a stability problem — memory and tail latency — and an
untrusted-input surface. Rejecting early with a 422 is cheaper and safer than
truncating late or letting the model absorb it.

**Internal errors are not returned to the caller.**
Inference failures are logged in full server-side but the client receives a
generic 500. Error messages are an information-disclosure path, particularly
when the input is untrusted.

**Logs are structured, one JSON object per line.**
Human-readable log strings are difficult to aggregate or alert on. Each event
carries the fields needed to diagnose it — batch size, latency, error.

**Latency is recorded as a bounded rolling window.**
`/metrics` exposes request counts, failure counts and p50/p95 latency in
Prometheus text format. The latency buffer is capped at 1000 samples so memory
does not grow without limit under sustained load. In a real deployment these
would be exported to a proper metrics backend rather than held in-process.

## Running it

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

The first start downloads the model, which takes a minute or two. `/readyz`
returns 503 until that completes.

### Docker

```bash
docker build -t ml-inference-server .
docker run -p 8000:8000 ml-inference-server
```

## API

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["this service is well documented", "the deploy failed again"]}'
```

```json
{
  "predictions": [
    {"label": "POSITIVE", "score": 0.9998},
    {"label": "NEGATIVE", "score": 0.9995}
  ],
  "latency_ms": 41.27,
  "model": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

Returns `422` for invalid input, `503` if the model is not loaded,
`500` on inference failure.

### `GET /healthz`
Liveness probe. Returns 200 while the process is running.

### `GET /readyz`
Readiness probe. Returns 503 until the model is loaded, then 200.

### `GET /metrics`
Prometheus-format counters and latency percentiles.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `distilbert-base-uncased-finetuned-sst-2-english` | Model to load |
| `MAX_CHARS` | `2000` | Maximum characters per input |
| `MAX_BATCH` | `32` | Maximum inputs per request |

## Tests

```bash
pytest -q
```

Tests cover the validation boundaries and the readiness gating, using a stubbed
classifier so they run without downloading the model.

## What I would add next

- Request timeouts and a concurrency limit, so a slow batch cannot exhaust the
  worker pool
- Model version reported alongside the model name, so a prediction can be traced
  back to the exact artefact that produced it
- Authentication on `/predict`, with `/metrics` bound to an internal interface only
- Batching across concurrent requests to improve throughput under load
