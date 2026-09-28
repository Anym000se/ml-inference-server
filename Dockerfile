FROM python:3.12-slim

WORKDIR /app

# Dependencies first so the layer caches when only source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Bake the model into the image so container start does not depend on
# network access, and so every replica serves an identical artefact.
ENV MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
RUN python -c "from transformers import pipeline; import os; pipeline('sentiment-analysis', model=os.environ['MODEL_NAME'])"

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/readyz').status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
